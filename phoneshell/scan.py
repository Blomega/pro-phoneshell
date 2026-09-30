"""Scroll scan: you scroll the phone by hand, the Mac reads everything that passes.

The page shows the live screen. While a scan runs, a worker thread watches the
same MJPEG stream the page shows, waits for the screen to change and then settle
(a scroll has stopped), grabs one full-resolution screenshot, reads it with
macOS Vision, and stitches the new lines into one running document in reading
order. Nothing is ever sent to the phone: the stream and `/screenshot` are both
sessionless on WebDriverAgent, so a scan never evicts the agent's session and
never injects a touch while your thumb is on the glass.

Why settle-then-capture instead of reading every frame: a frame caught mid
scroll is motion-blurred and OCRs into near-duplicates of the lines around it,
which the dedupe then has to guess about. One sharp frame per resting position
is both cheaper and cleaner.

The scan is a server-side job. The page only starts, stops and renders it, so a
reload or a closed tab loses nothing: state and every captured frame are on disk
under runtime/scans/<id>/.
"""
from __future__ import annotations

import base64
import difflib
import io
import json
import logging
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Callable, Iterator

import httpx
import numpy as np
from PIL import Image

from .config import RUNTIME, Config
from .perception import ocr

log = logging.getLogger("phoneshell.scan")

SCANS = RUNTIME / "scans"
SCANS.mkdir(parents=True, exist_ok=True)

# Motion detection runs on a tiny grayscale thumbnail: mean absolute difference
# on a 0-255 scale. Measured on a synthetic scroll: a 1-row nudge of text is
# ~2-4, a keyboard caret blink is <0.5, a real scroll step is 10+.
THUMB = (48, 104)
CHANGED = 3.0        # vs the last CAPTURED frame: new content worth reading
STILL = 1.2          # vs the previous frame: the screen has stopped moving
SETTLE_FRAMES = 2    # consecutive still frames before we call it settled
# The status bar (clock, battery) and the home indicator change on their own and
# are never content. Fractions of the screen height.
TOP_CUT = 0.055
BOTTOM_CUT = 0.015
FUZZY = 0.9          # OCR reads the same line slightly differently across frames
STICKY = 0.003       # same text within this fraction of screen height = did not move


def _norm(text: str) -> str:
    return re.sub(r"[\W_]+", " ", text.lower()).strip()


def _thumb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize(THUMB, Image.BILINEAR), dtype=np.float32)


def _diff(a: np.ndarray | None, b: np.ndarray | None) -> float:
    if a is None or b is None or a.shape != b.shape:
        return 255.0
    return float(np.abs(a - b).mean())


def mjpeg_frames(url: str, stop: threading.Event, timeout: float = 10.0) -> Iterator[bytes]:
    """JPEG payloads out of a multipart MJPEG stream, split on the JPEG markers
    rather than on the multipart boundary, which WDA does not always announce."""
    with httpx.Client(timeout=httpx.Timeout(timeout, read=timeout)) as client:
        with client.stream("GET", url) as resp:
            buf = b""
            for chunk in resp.iter_bytes():
                if stop.is_set():
                    return
                buf += chunk
                while True:
                    start = buf.find(b"\xff\xd8")
                    if start < 0:
                        buf = b""
                        break
                    end = buf.find(b"\xff\xd9", start + 2)
                    if end < 0:
                        buf = buf[start:]
                        break
                    yield buf[start:end + 2]
                    buf = buf[end + 2:]


@dataclass
class Line:
    text: str
    frame: int          # the frame it was first read in
    confidence: float
    y: float = 0.0      # centre, as a fraction of screen height, in that frame
    chrome: bool = False  # sticky header / tab bar: seen at the same spot while the page moved
    gap: bool = False     # no overlap with the screen before it: something may be missing above
    edge: bool = False    # first or last content line of its frame: may be half hidden
    seen: int = 1         # how many frames read it


@dataclass
class Frame:
    n: int
    at: float
    app: str
    new_lines: int
    source: str         # "screenshot" (full res) or "stream" (fallback)


@dataclass
class Scan:
    id: str
    label: str
    started: float
    status: str = "running"          # running -> stopped | failed
    reason: str = ""
    ended: float | None = None
    frames: list[Frame] = field(default_factory=list)
    lines: list[Line] = field(default_factory=list)
    # Cause -> count, so a run that silently dropped half its frames says why.
    events: dict[str, int] = field(default_factory=dict)
    extract: dict = field(default_factory=dict)
    # Where the last frame sits in `lines`, so the next one is aligned near it.
    window: list[int] = field(default_factory=lambda: [0, 0])

    @property
    def dir(self) -> Path:
        return SCANS / self.id

    def summary(self) -> dict:
        return {"id": self.id, "label": self.label, "started": self.started, "ended": self.ended,
                "status": self.status, "reason": self.reason, "frames": len(self.frames),
                "lines": len(self.content()), "extract_status": self.extract.get("status", "")}

    def to_json(self) -> dict:
        return asdict(self)

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "scan.json.tmp"
        tmp.write_text(json.dumps(self.to_json(), ensure_ascii=False))
        tmp.replace(self.dir / "scan.json")

    @classmethod
    def load(cls, sid: str) -> "Scan | None":
        path = SCANS / sid / "scan.json"
        if not path.exists():
            return None
        d = json.loads(path.read_text())
        d["frames"] = [Frame(**f) for f in d.get("frames", [])]
        d["lines"] = [Line(**l) for l in d.get("lines", [])]
        s = cls(**d)
        if s.status == "running":
            # The process that owned it is gone: say so instead of pretending.
            s.status, s.reason, s.ended = "stopped", "server restarted during the scan", s.ended or time.time()
        return s

    def content(self) -> list[Line]:
        return [l for l in self.lines if not l.chrome]

    def text(self) -> str:
        return "\n".join(("\n[... gap: scrolled past faster than it could be read ...]\n" if l.gap else "") + l.text
                         for l in self.content())


def _digits(key: str) -> str:
    return "".join(re.findall(r"\d+", key))


def _overlaps(a: str, b: str) -> bool:
    """Same text, or one read contains the other ("Home Search" vs "Home")."""
    return bool(a) and bool(b) and (a == b or f" {a} " in f" {b} " or f" {b} " in f" {a} ")


def _solid(line: Line) -> bool:
    return not line.chrome and (not line.edge or line.seen > 1)


def same_line(a: str, b: str) -> bool:
    """Two normalised OCR reads of one line. Numbers must agree exactly: feeds are
    full of templated lines ("41 likes" / "14 likes") that differ ONLY there."""
    if a == b:
        return True
    if min(len(a), len(b)) < 8 or _digits(a) != _digits(b):
        return False
    return difflib.SequenceMatcher(None, a, b, autojunk=False).ratio() >= FUZZY


def merge_frame(scan: Scan, frame: list[tuple[str, float, float]], frame_n: int) -> int:
    """Stitch one frame's content lines (top to bottom: text, confidence, y) into
    scan.lines. Returns how many lines were added.

    This is sequence alignment, not a set of seen strings. A set cannot tell a
    second "Reply" under a second comment from the first one, and a feed is
    mostly lines like that. So the frame is aligned, in order, against the part
    of the document around where the previous frame sat; lines that align are
    already known, and the rest are inserted between their aligned neighbours.
    That places new lines correctly whether you scrolled down or back up.
    """
    doc = scan.lines
    content_idx = [i for i, l in enumerate(doc) if not l.chrome]
    ws, we = scan.window
    span = 3 * max(len(frame), 10)
    region = [i for i in content_idx if ws - span <= i <= we + span]
    rkeys = [_norm(doc[i].text) for i in region]

    fkeys: list[str] = []
    for text, _c, _y, _e in frame:
        key = _norm(text)
        # Snap an OCR variant onto the spelling already in the region, so the
        # exact-match aligner below sees them as equal.
        fkeys.append(next((k for k in rkeys if same_line(key, k)), key))

    sm = difflib.SequenceMatcher(None, rkeys, fkeys, autojunk=False)
    pairs = [(region[blk.a + t], blk.b + t) for blk in sm.get_matching_blocks() for t in range(blk.size)]
    # Lines that repeat all down a feed ("Reply", "2h", "Like") say nothing about
    # WHERE we are: three of them line up with any three others. A screen is
    # placed only if at least one line it matched is unique in the region.
    if not any(rkeys.count(fkeys[j]) == 1 for _d, j in pairs):
        pairs = []
    matched = {j: d for d, j in pairs}

    slots: dict[int, list[Line]] = {}
    added = 0
    gap = not pairs and bool(content_idx)
    for d in matched.values():
        doc[d].seen += 1
    # A line only ever read at the edge of a screen, that a later screen shows
    # in full view between two lines it DID align, was a half-hidden fragment
    # (a line sliding under a sticky header reads as "rrupiy"). Drop it.
    if len(matched) >= 2:
        lo, hi = min(matched.values()), max(matched.values())
        hit = set(matched.values())
        stale = {i for i in range(lo + 1, hi) if i not in hit and doc[i].edge and doc[i].seen == 1
                 and not doc[i].chrome}
        if stale:
            keep = [i for i in range(len(doc)) if i not in stale]
            remap = {old: new for new, old in enumerate(keep)}
            doc[:] = [doc[i] for i in keep]
            matched = {j: remap[d] for j, d in matched.items()}

    for j, (text, conf, y, edge) in enumerate(frame):
        if j in matched or not fkeys[j]:
            continue
        before = [matched[k] for k in matched if k < j]
        after = [matched[k] for k in matched if k > j]
        if before:
            pos = max(before) + 1
        elif after:
            pos = min(after)
        else:
            pos = we + 1 if content_idx else len(doc)
        # Between two lines some earlier screen read in full view, there is
        # nothing left to discover: an edge line landing there is a fragment of
        # a line sliding under the header or the tab bar.
        if edge and 0 < pos < len(doc) and _solid(doc[pos - 1]) and _solid(doc[pos]):
            continue
        line = Line(text=text, frame=frame_n, confidence=round(conf, 3), y=round(y, 4), gap=gap, edge=edge)
        gap = False
        slots.setdefault(pos, []).append(line)
        added += 1

    if added:
        out: list[Line] = []
        for i in range(len(doc) + 1):
            out.extend(slots.get(i, []))
            if i < len(doc):
                out.append(doc[i])
        doc[:] = out
        # Matched indices shifted by the insertions before them.
        shift = lambda d: d + sum(len(v) for k, v in slots.items() if k <= d)
        matched = {j: shift(d) for j, d in matched.items()}
    mine = list(matched.values()) + [i for i, l in enumerate(doc) if l.frame == frame_n]
    if mine:
        scan.window = [min(mine), max(mine)]
    return added


def reading_order(boxes: list, height: float) -> list:
    """Top to bottom, and left to right within a visual row."""
    rows: list[list] = []
    for b in sorted(boxes, key=lambda b: b.cy):
        if rows and abs(b.cy - rows[-1][0].cy) < max(4.0, 0.5 * min(b.h, rows[-1][0].h)):
            rows[-1].append(b)
        else:
            rows.append([b])
    return [b for row in rows for b in sorted(row, key=lambda b: b.x)]


class Scanner:
    """Runs one scan at a time. Owned by the server process."""

    def __init__(self, cfg_loader: Callable[[], Config] = Config.load,
                 mjpeg_url: str | None = None, wda_url: str | None = None):
        self._cfg_loader = cfg_loader
        self._mjpeg_url = mjpeg_url
        self._wda_url = wda_url
        self._lock = threading.Lock()
        self.current: Scan | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.last_frame_at = 0.0
        self._prev_boxes: list[tuple[str, float]] = []

    # --- lifecycle -------------------------------------------------------
    def start(self, label: str = "") -> Scan:
        with self._lock:
            if self.current and self.current.status == "running":
                return self.current     # one open scan at a time: a double click is not a second scan
            sid = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
            scan = Scan(id=sid, label=label.strip() or time.strftime("Scan %b %d %H:%M"), started=time.time())
            (scan.dir / "frames").mkdir(parents=True, exist_ok=True)
            scan.save()
            self.current = scan
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(scan, self._stop), daemon=True,
                                            name=f"scan-{sid}")
            self._thread.start()
            return scan

    def stop(self) -> Scan | None:
        with self._lock:
            scan = self.current
            if scan and scan.status == "running":
                self._stop.set()
                scan.status, scan.ended = "stopped", time.time()
                scan.save()
            return scan

    def get(self, sid: str) -> Scan | None:
        if self.current and self.current.id == sid:
            return self.current
        return Scan.load(sid)

    @staticmethod
    def history(limit: int = 30) -> list[dict]:
        out = []
        for p in sorted(SCANS.iterdir(), reverse=True):
            if (p / "scan.json").exists():
                s = Scan.load(p.name)
                if s:
                    out.append(s.summary())
            if len(out) >= limit:
                break
        return out

    @staticmethod
    def delete(sid: str) -> bool:
        path = SCANS / sid
        if not re.fullmatch(r"[\w-]+", sid) or not path.is_dir():
            return False
        shutil.rmtree(path)
        return True

    # --- worker ----------------------------------------------------------
    def _urls(self) -> tuple[str, str]:
        cfg = self._cfg_loader()
        return (self._mjpeg_url or cfg.mjpeg_url, self._wda_url or cfg.wda_base_url)

    def _run(self, scan: Scan, stop: threading.Event) -> None:
        mjpeg, wda = self._urls()
        failures = 0
        try:
            while not stop.is_set():
                try:
                    self._watch(scan, stop, mjpeg, wda)
                    failures = 0
                except httpx.HTTPError as e:
                    failures += 1
                    self._event(scan, f"stream: {type(e).__name__}")
                    if failures >= 5:
                        raise RuntimeError(
                            "cannot read the phone's screen stream. Is the phone connected and the bridge up "
                            "(bin/phoneshell up)?") from e
                    stop.wait(1.5)
        except Exception as e:  # noqa: BLE001 - the page must learn why, whatever it was
            log.exception("scan %s failed", scan.id)
            scan.status, scan.reason, scan.ended = "failed", str(e), time.time()
            scan.save()

    def _event(self, scan: Scan, cause: str) -> None:
        scan.events[cause] = scan.events.get(cause, 0) + 1

    def _watch(self, scan: Scan, stop: threading.Event, mjpeg: str, wda: str) -> None:
        # A reader thread keeps only the newest frame. Reading inline would let
        # frames pile up in the socket while OCR runs, and the loop would then
        # judge "settled" on a screen that is already a second old.
        latest: list = [0, None, None]        # seq, jpeg bytes, error
        cond = threading.Condition()

        def reader() -> None:
            try:
                for jpeg in mjpeg_frames(mjpeg, stop):
                    with cond:
                        latest[0] += 1
                        latest[1] = jpeg
                        cond.notify()
            except Exception as e:  # noqa: BLE001
                with cond:
                    latest[2] = e
                    cond.notify()

        threading.Thread(target=reader, daemon=True, name=f"scan-read-{scan.id}").start()
        last_captured: np.ndarray | None = None
        prev: np.ndarray | None = None
        still, seen = 0, 0
        self._prev_boxes = []
        with httpx.Client(timeout=8.0) as http:
            while not stop.is_set():
                with cond:
                    cond.wait_for(lambda: latest[0] != seen or latest[2] is not None or stop.is_set(), timeout=5)
                    if latest[2] is not None:
                        raise latest[2]
                    if latest[0] == seen:
                        if stop.is_set():
                            return
                        raise httpx.ReadTimeout("no frame from the screen stream for 5s")
                    seen, jpeg = latest[0], latest[1]
                try:
                    img = Image.open(io.BytesIO(jpeg))
                    img.load()
                except Exception:  # noqa: BLE001
                    self._event(scan, "stream: undecodable frame")
                    continue
                t = _thumb(img)
                still = still + 1 if _diff(t, prev) < STILL else 0
                prev = t
                if still >= SETTLE_FRAMES and _diff(t, last_captured) >= CHANGED:
                    self._capture(scan, img, http, wda)
                    last_captured = t

    def _capture(self, scan: Scan, stream_img: Image.Image, http: httpx.Client, wda: str) -> None:
        n = len(scan.frames) + 1
        source = "screenshot"
        raw: bytes | None = None
        try:
            r = http.get(f"{wda}/screenshot")
            r.raise_for_status()
            raw = base64.b64decode(r.json()["value"])
            img = Image.open(io.BytesIO(raw))
            img.load()
        except Exception as e:  # noqa: BLE001 - fall back to the (smaller) stream frame, and count it
            self._event(scan, f"screenshot fallback: {type(e).__name__}")
            img, raw, source = stream_img, None, "stream"
        app = ""
        try:
            info = http.get(f"{wda}/wda/activeAppInfo", timeout=1.5).json().get("value") or {}
            app = info.get("name") or info.get("bundleId") or ""
        except Exception:  # noqa: BLE001
            pass

        # 0.5: below it Vision is reading a line half covered by something, and
        # returns strings like "_rA_-Il_-__".
        boxes = ocr.read_boxes(raw if raw is not None else img, fast=False, min_confidence=0.5)
        if not ocr.AVAILABLE:
            raise RuntimeError("macOS Vision is not available to Python (pip install pyobjc-framework-Vision)")
        h = img.height
        kept = reading_order([b for b in boxes if h * TOP_CUT < b.cy < h * (1 - BOTTOM_CUT)], h)
        read = [(b.text, b.confidence, b.cy / h) for b in kept]
        # Sticky chrome: the same text at the same height as in the previous
        # capture, although the screen changed enough to be captured. A frame
        # where MOST lines stayed put did not scroll (an image loaded, a video
        # played), so nothing in it is called sticky.
        stay = [any(_overlaps(_norm(t), k) and abs(y - py) < STICKY for k, py in self._prev_boxes)
                for t, _c, y in read]
        self._prev_boxes = [(_norm(t), y) for t, _c, y in read]
        if read and sum(stay) / len(read) > 0.7:
            stay = [False] * len(read)
        for (t, _c, y), sticky in zip(read, stay):
            if sticky:
                for line in scan.lines:
                    if not line.chrome and _overlaps(_norm(line.text), _norm(t)) and abs(line.y - y) < STICKY:
                        line.chrome = True
        content = [r for r, sticky in zip(read, stay) if not sticky]
        last = len(content) - 1
        added = merge_frame(scan, [(t, c, y, i in (0, last)) for i, (t, c, y) in enumerate(content)], n)

        thumb = img.convert("RGB")
        thumb.thumbnail((720, 1560))
        thumb.save(scan.dir / "frames" / f"{n:04d}.jpg", quality=80)
        scan.frames.append(Frame(n=n, at=time.time(), app=app, new_lines=added, source=source))
        if added == 0:
            self._event(scan, "frame had nothing new")
        self.last_frame_at = time.time()
        scan.save()


# --- structuring: an LLM turns the raw lines into rows, as a queued job ------

EXTRACT_PROMPT = """You are given text read by OCR from a phone screen while someone scrolled through it, in reading order.
Instruction from the user: {instruction}

Return ONLY a JSON object, no prose, no code fence:
{{"columns": ["..."], "rows": [{{"<column>": "<value>"}}], "notes": "<one short sentence on anything you could not read or had to skip>"}}
Rules: one row per repeated item (post, product, contact, message, listing...). Use the text exactly as read; do not invent values; leave a cell "" when it is not there. Drop interface chrome (tab bar labels, buttons like Follow/Share) unless asked for.

OCR TEXT:
{text}
"""


def run_extract(scan: Scan, instruction: str, model: str = "sonnet") -> None:
    """Blocking. Call from a worker thread; progress lives on scan.extract."""
    scan.extract = {"status": "running", "instruction": instruction, "started": time.time()}
    scan.save()
    claude = shutil.which("claude") or str(Path.home() / ".local/bin/claude")
    try:
        if not Path(claude).exists():
            raise RuntimeError("the `claude` CLI is not installed on this Mac")
        text = scan.text()
        if not text.strip():
            raise RuntimeError("nothing has been read yet: scroll through something first")
        prompt = EXTRACT_PROMPT.format(instruction=instruction or "extract every item as a table", text=text[:180_000])
        proc = subprocess.run([claude, "-p", "--model", model, "--output-format", "text"],
                              input=prompt, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout).strip()[-400:] or f"claude exited {proc.returncode}")
        out = proc.stdout.strip()
        m = re.search(r"\{.*\}", out, re.S)
        if not m:
            raise RuntimeError("the model did not return JSON: " + out[:200])
        data = json.loads(m.group(0))
        rows = data.get("rows") or []
        cols = data.get("columns") or (list(rows[0].keys()) if rows else [])
        scan.extract = {"status": "done", "instruction": instruction, "columns": cols, "rows": rows,
                        "notes": data.get("notes", ""), "started": scan.extract["started"], "ended": time.time()}
    except Exception as e:  # noqa: BLE001
        scan.extract = {"status": "failed", "instruction": instruction, "reason": str(e)[:500],
                        "started": scan.extract.get("started"), "ended": time.time()}
    scan.save()
