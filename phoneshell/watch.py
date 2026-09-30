"""Watch and learn: do a task on the phone once by hand, and it becomes a macro.

iOS gives no other process your touches, so the watcher does not see taps. It
sees SCREENS: every time the screen comes to rest it keeps a keyframe (a
screenshot, the accessibility tree, the app in front). Afterwards a model reads
the keyframes in order and names the action that must have happened between
each pair ("the Pad Thai row on frame 6 was tapped: frame 7 is its page").

Every learned step is then checked by the same code replay uses: a tap target
must be findable with find_by_text on the frame it came from, the anchor must be
on that frame, the expectation on the next. A step that fails a check is shown
to the person as unverified rather than silently saved.

Everything here is sessionless on WebDriverAgent (MJPEG, /screenshot, /source,
/wda/activeAppInfo). Creating a session would switch apps under the person's
thumb; watching must never do that.
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import shutil
import subprocess
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field
from pathlib import Path

import httpx
import numpy as np
from PIL import Image

from .agent.macros import Macro, Step
from .config import RUNTIME, Config
from .perception.tree import Element, condense, find_by_text, flatten
from .wda.client import WDAClient

log = logging.getLogger("phoneshell.watch")

DEMOS = RUNTIME / "watch"
DEMOS.mkdir(parents=True, exist_ok=True)

THUMB = (48, 104)
# Any rest after motion is worth a look: MJPEG only sends frames when pixels
# change, so noise never reaches here, and a few typed letters move the whole
# thumbnail by well under 1.0. The tree comparison in _keyframe drops real repeats.
CHANGED = 0.15
SETTLE_GAP = 0.5     # WDA's MJPEG goes silent while nothing moves: silence this long = at rest
MAX_FRAMES = 150
ACTIONS = {"open_app", "tap", "type", "scroll_to", "swipe", "press", "alert", "open_url"}


def _thumb(img: Image.Image) -> np.ndarray:
    return np.asarray(img.convert("L").resize(THUMB, Image.BILINEAR), dtype=np.float32)


def _diff(a, b) -> float:
    if a is None or b is None:
        return 255.0
    return float(np.abs(a - b).mean())


def _mjpeg(url: str, stop: threading.Event, handle: list):
    with httpx.Client(timeout=httpx.Timeout(10.0, read=None)) as client:
        with client.stream("GET", url) as resp:
            handle.append(resp)
            buf = b""
            for chunk in resp.iter_bytes():
                if stop.is_set():
                    return
                buf += chunk
                while True:
                    a = buf.find(b"\xff\xd8")
                    if a < 0:
                        buf = b""
                        break
                    b = buf.find(b"\xff\xd9", a + 2)
                    if b < 0:
                        buf = buf[a:]
                        break
                    yield buf[a:b + 2]
                    buf = buf[b + 2:]


@dataclass
class Keyframe:
    n: int
    t: float
    app: str
    bundle: str
    w: float
    h: float
    elements: list[dict]


@dataclass
class Demo:
    id: str
    label: str
    started: float
    status: str = "watching"      # watching -> learning -> ready | failed ; ready -> saved
    reason: str = ""
    ended: float | None = None
    frames: list[Keyframe] = field(default_factory=list)
    web_taps: list[dict] = field(default_factory=list)   # taps made through the page: exact ground truth
    draft: dict = field(default_factory=dict)
    saved_as: str = ""

    @property
    def dir(self) -> Path:
        return DEMOS / self.id

    def summary(self) -> dict:
        return {"id": self.id, "label": self.label, "status": self.status, "reason": self.reason,
                "started": self.started, "ended": self.ended, "frames": len(self.frames),
                "steps": len(self.draft.get("steps", [])), "saved_as": self.saved_as}

    def save(self) -> None:
        self.dir.mkdir(parents=True, exist_ok=True)
        tmp = self.dir / "demo.json.tmp"
        tmp.write_text(json.dumps(asdict(self), ensure_ascii=False))
        tmp.replace(self.dir / "demo.json")

    @classmethod
    def load(cls, did: str) -> "Demo | None":
        path = DEMOS / did / "demo.json"
        if not re.fullmatch(r"[\w-]+", did) or not path.exists():
            return None
        d = json.loads(path.read_text())
        d["frames"] = [Keyframe(**f) for f in d.get("frames", [])]
        demo = cls(**d)
        if demo.status in ("watching", "learning"):
            demo.status, demo.reason = "failed", "the server restarted while this was in progress"
        return demo


def elements_of(frame: Keyframe) -> list[Element]:
    """Rebuild Elements from a keyframe, so checks run the exact code replay runs."""
    out = []
    for e in frame.elements:
        out.append(Element(idx=e["id"], type=e["type"], label=e.get("label", ""), name=e.get("name", ""),
                           value=e.get("value", ""), placeholder=e.get("placeholder", ""),
                           identifier=e.get("identifier", ""), x=e["x"], y=e["y"], w=e["w"], h=e["h"],
                           accessible=e.get("accessible", False), focused=e.get("focused", False),
                           traits=e.get("traits", ""), children_text=e.get("children_text", [])))
    return out


class Watcher:
    """One demonstration at a time. Owned by the server process."""

    def __init__(self, cfg: Config | None = None):
        self.cfg = cfg or Config.load()
        self.current: Demo | None = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None

    # --- lifecycle -------------------------------------------------------
    def start(self, label: str = "") -> Demo:
        with self._lock:
            if self.current and self.current.status == "watching":
                return self.current
            did = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
            demo = Demo(id=did, label=label.strip() or "demo", started=time.time())
            (demo.dir / "frames").mkdir(parents=True, exist_ok=True)
            demo.save()
            self.current = demo
            self._stop = threading.Event()
            self._thread = threading.Thread(target=self._run, args=(demo, self._stop), daemon=True,
                                            name=f"watch-{did}")
            self._thread.start()
            return demo

    def stop(self, learn: bool = True) -> Demo | None:
        """Stop watching and queue the learning pass. Returns at once."""
        with self._lock:
            demo = self.current
            if not demo or demo.status != "watching":
                return demo
            self._stop.set()
            demo.ended = time.time()
            if not learn:
                demo.status = "ready"
                demo.save()
                return demo
            demo.status = "learning"
            demo.save()
        watcher = self._thread

        def after_last_frame() -> None:
            if watcher is not None:
                watcher.join(timeout=30)   # a keyframe may still be being written
            learn_demo(demo)

        threading.Thread(target=after_last_frame, daemon=True, name=f"learn-{demo.id}").start()
        return demo

    def note_web_tap(self, x: float, y: float) -> None:
        demo = self.current
        if demo and demo.status == "watching":
            demo.web_taps.append({"t": time.time(), "x": x, "y": y, "after_frame": len(demo.frames)})

    def get(self, did: str) -> Demo | None:
        if self.current and self.current.id == did:
            return self.current
        return Demo.load(did)

    @staticmethod
    def history(limit: int = 30) -> list[dict]:
        out = []
        for p in sorted(DEMOS.iterdir(), reverse=True):
            d = Demo.load(p.name) if p.is_dir() else None
            if d:
                out.append(d.summary())
            if len(out) >= limit:
                break
        return out

    # --- worker ----------------------------------------------------------
    def _run(self, demo: Demo, stop: threading.Event) -> None:
        base = self.cfg.wda_base_url
        try:
            with httpx.Client(timeout=15.0) as http:
                last = self._keyframe(demo, http, base)
                while not stop.is_set() and len(demo.frames) < MAX_FRAMES:
                    try:
                        last = self._watch(demo, stop, http, base, last)
                    except httpx.HTTPError as e:
                        log.info("stream dropped: %s", e)
                        if not _alive(http, base):
                            raise RuntimeError("the phone stopped answering; is it still plugged in?") from e
                        stop.wait(1.0)
                if len(demo.frames) >= MAX_FRAMES and demo.status == "watching":
                    threading.Thread(target=self.stop, daemon=True).start()
        except Exception as e:  # noqa: BLE001
            log.exception("watch %s failed", demo.id)
            if demo.status == "watching":
                demo.status, demo.reason, demo.ended = "failed", str(e), time.time()
                demo.save()

    def _watch(self, demo: Demo, stop: threading.Event, http: httpx.Client, base: str, last):
        latest: list = [0, None, None]
        cond = threading.Condition()
        gone = threading.Event()
        handle: list = []

        def reader() -> None:
            try:
                for jpeg in _mjpeg(self.cfg.mjpeg_url, gone, handle):
                    with cond:
                        latest[0] += 1
                        latest[1] = jpeg
                        cond.notify()
            except Exception as e:  # noqa: BLE001
                with cond:
                    latest[2] = e
                    cond.notify()

        threading.Thread(target=reader, daemon=True).start()
        seen, moved = 0, False
        try:
            while not stop.is_set():
                with cond:
                    cond.wait_for(lambda: latest[0] != seen or latest[2] is not None or stop.is_set(),
                                  timeout=SETTLE_GAP)
                    if latest[2] is not None:
                        raise latest[2]
                    fresh = latest[0] != seen
                    if fresh:
                        seen, jpeg = latest[0], latest[1]
                if fresh:
                    moved = True
                    continue
                if moved and not stop.is_set():
                    moved = False
                    t = _thumb(Image.open(io.BytesIO(jpeg)))
                    if _diff(t, last) >= CHANGED:
                        kept = self._keyframe(demo, http, base)
                        if kept is not None:
                            last = kept
            return last
        finally:
            gone.set()
            for r in handle:
                try:
                    r.close()
                except Exception:  # noqa: BLE001
                    pass

    def _heal(self) -> bool:
        """iOS sometimes withdraws the runner's permission to see the screen
        (error 41, "Not authorized for performing UI testing actions"): after the
        phone locks, or both volume buttons are held. Only a runner restart
        clears it, so do that rather than failing the demo."""
        from . import device
        log.warning("runner lost UI permission; restarting it")
        return device.recycle_runner(self.cfg.device.udid, self.cfg.wda.runner_bundle_id,
                                     self.cfg.wda.port, self.cfg.wda.mjpeg_port).ok

    def _keyframe(self, demo: Demo, http: httpx.Client, base: str):
        """Screenshot + tree + app of the screen at rest. Returns its thumbnail."""
        r = http.get(f"{base}/screenshot")
        if r.status_code == 500 and "Not authorized" in r.text and self._heal():
            r = http.get(f"{base}/screenshot")
        r.raise_for_status()
        raw = base64.b64decode(r.json()["value"])
        img = Image.open(io.BytesIO(raw))
        img.load()
        src = http.get(f"{base}/source", params={"format": "json",
                                                  "excluded_attributes": WDAClient.DEFAULT_EXCLUDED_ATTRS})
        src.raise_for_status()
        tree = src.json().get("value") or {}
        flat = flatten(tree)
        sw = flat[0].w if flat and flat[0].w else 430.0
        sh = flat[0].h if flat and flat[0].h else 932.0
        els = condense(flat, sw, sh, max_elements=150, status_bar_h=sh * 0.06)
        app = bundle = ""
        try:
            info = http.get(f"{base}/wda/activeAppInfo", timeout=3).json().get("value") or {}
            app, bundle = info.get("name") or "", info.get("bundleId") or ""
        except Exception:  # noqa: BLE001
            pass
        # Skip a keyframe that reads the same as the previous one (an animation
        # settled into the same screen): it would only add noise to learning.
        sig = [(e.type, e.text, e.value, round(e.cy / 20)) for e in els]
        if demo.frames:
            prev = demo.frames[-1]
            if prev.bundle == bundle and [(e["type"], e["text"], e.get("value", ""), round((e["y"] + e["h"] / 2) / 20))
                                          for e in prev.elements] == sig:
                return _thumb(img)
        n = len(demo.frames) + 1
        shot = img.convert("RGB")
        shot.thumbnail((600, 1300))
        shot.save(demo.dir / "frames" / f"{n:04d}.jpg", quality=82)
        demo.frames.append(Keyframe(
            n=n, t=time.time(), app=app, bundle=bundle, w=sw, h=sh,
            elements=[{"id": i, "type": e.type, "text": e.text, "label": e.label, "name": e.name,
                       "value": e.value, "placeholder": e.placeholder, "identifier": e.identifier,
                       "x": round(e.x, 1), "y": round(e.y, 1), "w": round(e.w, 1), "h": round(e.h, 1),
                       "accessible": e.accessible, "focused": e.focused, "traits": e.traits,
                       "children_text": e.children_text[:6]} for i, e in enumerate(els)]))
        demo.save()
        return _thumb(img)


def _alive(http: httpx.Client, base: str) -> bool:
    try:
        return http.get(f"{base}/status", timeout=4).status_code == 200
    except httpx.HTTPError:
        return False


# --- learning -------------------------------------------------------------

LEARN_PROMPT = """You are turning a recorded demonstration on an iPhone into a replayable macro.

The person did this task by hand: "{label}".
Below are keyframes: the screen each time it came to rest, in order. For each you get the app and the
on-screen elements as `[id] Type "text" value=... @(x,y) WxH`. Screenshots are at {dir}/frames/NNNN.jpg
(frame 1 = 0001.jpg); open them with the Read tool whenever the element list is not enough to tell what
changed. {web_taps}

Work out the actions that took the phone from each frame to the next, and write the CLEAN route: drop
mistakes the person undid (tapped the wrong thing then went back), drop pure browsing that did not lead
anywhere, keep everything that mattered.

Actions (exact vocabulary):
- open_app  params {{"name": "<app display name>"}}           (use this for the first step if an app was opened)
- tap       target_text "<text of the element tapped, as it appears in the element list>"
- type      target_text "<label or placeholder of the field>", params {{"text": "<what was typed>", "submit": true|false}}
- scroll_to params {{"text": "<text that was scrolled to>"}}    (only when a later tap target was below the fold)
- press     params {{"button": "home"|"back"}}
- alert     params {{"action": "accept"|"dismiss", "button": "<button text>"}}

For every step also give:
- from_frame / to_frame: the keyframe numbers before and after it.
- anchor: a short, STABLE text on from_frame that identifies that screen (a title or section name; never a price,
  time, count or anything that changes day to day). "" if nothing stable.
- expect: a short stable text on to_frame that proves the step worked, or "".
- confirm: true if the step spends money, places an order, pays, books, subscribes, sends a message, posts or
  deletes. When in doubt, true.
- why: one short sentence of evidence (what changed between the frames).

target_text must be a substring of that element's text in from_frame's list, short and distinctive
(e.g. "Pad Thai", not the whole row). If what was tapped has no text (an icon), use its identifier or label
from the list; if it has none at all, say so in why and use "".

Inputs: if a value is clearly a choice that would change next time (a search term, a dish, an address, a
recipient, a message body), name it as an input, put its demo value in "inputs", and write it as
{{input_name}} inside the steps that use it (in target_text, params.text, anchor or expect).

Return ONLY JSON, no prose, no code fence:
{{"name": "<short-kebab-case-name>", "description": "<one line: what this does>", "app": "<main app>",
  "inputs": {{"<input>": "<demo value>"}},
  "steps": [{{"action": "...", "target_text": "...", "params": {{}}, "anchor": "...", "expect": "...",
             "confirm": false, "from_frame": 1, "to_frame": 2, "why": "..."}}],
  "notes": "<anything uncertain>"}}

KEYFRAMES:
{frames}
"""


def _frame_text(f: Keyframe) -> str:
    rows = []
    for e in f.elements:
        bits = [f"[{e['id']}]", e["type"]]
        if e["text"]:
            bits.append(json.dumps(e["text"][:80], ensure_ascii=False))
        if e.get("value") and e["value"] != e["text"]:
            bits.append(f"value={e['value'][:40]!r}")
        if e.get("placeholder") and not e.get("value"):
            bits.append(f"placeholder={e['placeholder'][:40]!r}")
        if e.get("focused"):
            bits.append("FOCUSED")
        bits.append(f"@({int(e['x'] + e['w'] / 2)},{int(e['y'] + e['h'] / 2)}) {int(e['w'])}x{int(e['h'])}")
        rows.append(" ".join(bits))
    return f"--- frame {f.n} | app: {f.app or f.bundle} | +{f.t:.1f}s\n" + "\n".join(rows)


# A landmark that is different tomorrow ("Good afternoon", "3 items", "฿145",
# "12:40") would make replay stop on a screen that is actually right.
_VOLATILE = re.compile(r"\bgood (morning|afternoon|evening|night)\b|\d", re.I)


def check_steps(demo: Demo, steps: list[dict], inputs: dict) -> list[dict]:
    """Verify each learned step against the recorded frames with replay's own
    matching. Adds `checks` (problems found) and `verified` to every step."""
    frames = {f.n: f for f in demo.frames}
    out = []
    for s in steps:
        s = dict(s)
        problems = []
        if s.get("action") not in ACTIONS:
            problems.append(f"unknown action {s.get('action')!r}")
        fill = lambda t: re.sub(r"\{(\w+)\}", lambda m: str(inputs.get(m.group(1), m.group(0))), t or "")
        f_from, f_to = frames.get(s.get("from_frame")), frames.get(s.get("to_frame"))
        if f_from is None:
            problems.append("from_frame does not exist")
        else:
            els = elements_of(f_from)
            if s.get("action") == "tap":
                target = fill(s.get("target_text", ""))
                if not target:
                    problems.append("no text to find the button by")
                elif not find_by_text(els, target):
                    problems.append(f"{target!r} is not a tappable element on frame {f_from.n}")
            anchor = fill(s.get("anchor", ""))
            if anchor and _VOLATILE.search(anchor):
                problems.append(f"anchor {anchor!r} changes with time or numbers; dropped")
                s["anchor"] = anchor = ""
            if anchor and not find_by_text(els, anchor, clickable_only=False):
                problems.append(f"anchor {anchor!r} is not on frame {f_from.n}; dropped")
                s["anchor"] = ""
        expect = fill(s.get("expect", ""))
        if expect and _VOLATILE.search(expect):
            problems.append(f"expect {expect!r} changes with time or numbers; dropped")
            s["expect"] = expect = ""
        if expect and f_to is not None and not find_by_text(elements_of(f_to), expect, clickable_only=False):
            problems.append(f"expect {expect!r} is not on frame {f_to.n}; dropped")
            s["expect"] = ""
        s["checks"] = problems
        # Dropped anchors/expects are fixed in place, not blockers.
        s["verified"] = not any("dropped" not in p for p in problems)
        out.append(s)
    return out


def learn_demo(demo: Demo, model: str = "sonnet") -> None:
    """Blocking: runs on a worker thread. Result lands in demo.draft."""
    try:
        if len(demo.frames) < 2:
            raise RuntimeError("only one screen was seen, so there is nothing to learn. Watch again and do the task.")
        claude = shutil.which("claude") or str(Path.home() / ".local/bin/claude")
        if not Path(claude).exists():
            raise RuntimeError("the `claude` CLI is not installed on this Mac")
        web = ""
        if demo.web_taps:
            web = ("Some taps were made through the web page, so their exact point is known (points, not pixels): "
                   + "; ".join(f"after frame {t['after_frame']}: ({int(t['x'])},{int(t['y'])})" for t in demo.web_taps))
        prompt = LEARN_PROMPT.format(label=demo.label, dir=demo.dir, web_taps=web,
                                     frames="\n\n".join(_frame_text(f) for f in demo.frames))
        proc = subprocess.run([claude, "-p", "--model", model, "--output-format", "text",
                               "--allowedTools", "Read", "--add-dir", str(demo.dir)],
                              input=prompt, capture_output=True, text=True, timeout=900, cwd=str(demo.dir))
        if proc.returncode != 0:
            raise RuntimeError((proc.stderr or proc.stdout).strip()[-400:] or f"claude exited {proc.returncode}")
        m = re.search(r"\{.*\}", proc.stdout, re.S)
        if not m:
            raise RuntimeError("the model did not return JSON: " + proc.stdout[:200])
        draft = json.loads(m.group(0))
        inputs = {str(k): str(v) for k, v in (draft.get("inputs") or {}).items()}
        draft["inputs"] = inputs
        draft["steps"] = check_steps(demo, draft.get("steps") or [], inputs)
        demo.draft = draft
        demo.status = "ready"
    except Exception as e:  # noqa: BLE001
        log.exception("learning %s failed", demo.id)
        demo.status, demo.reason = "failed", str(e)[:500]
    demo.save()


def _generalise(text: str, inputs: dict) -> str:
    """A landmark that contains an input's demo value ("Pad Thai Ban Khun Mae"
    while restaurant = "Ban Khun Mae") would only ever match the demo's choice.
    Make it the input itself, so it follows whatever this run asked for."""
    for k, v in sorted(inputs.items(), key=lambda kv: -len(kv[1] or "")):   # most specific first
        if v and "{" + k + "}" != text and v.lower() in text.lower():
            return "{" + k + "}"
    return text


def to_macro(demo: Demo, name: str, description: str, steps: list[dict], inputs: dict) -> Macro:
    name = re.sub(r"[^\w-]+", "-", name.strip().lower()).strip("-") or demo.id
    steps = [{**s, "anchor": _generalise(s.get("anchor", ""), inputs),
              "expect": _generalise(s.get("expect", ""), inputs),
              "target_text": _generalise(s.get("target_text", ""), inputs)} for s in steps]
    macro_steps = [Step(action=s["action"], params=dict(s.get("params") or {}), anchor=s.get("anchor", ""),
                        target_text=s.get("target_text", ""), expect=s.get("expect", ""),
                        note=s.get("why", ""), confirm=bool(s.get("confirm")))
                   for s in steps]
    bundle = next((f.bundle for f in reversed(demo.frames) if f.bundle and f.bundle != "com.apple.springboard"), "")
    return Macro(name=name, description=description, bundle_id=bundle, steps=macro_steps,
                 created=time.time(), inputs={str(k): str(v) for k, v in inputs.items()}, source="demo")
