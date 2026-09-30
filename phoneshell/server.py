"""The local app: a live view of the phone, click-to-control, and a chat that
drives the agent.

The brain is the `claude` CLI in print mode with the phoneshell MCP server
attached, so this runs on the subscription already on this Mac and needs no API
key. Swap BRAIN_CMD for the Anthropic SDK if you would rather pay per token.

One wrinkle worth knowing: WebDriverAgent allows exactly one session, and
creating a new one tears down the old one. The agent's MCP process and this
server therefore take turns; the client recreates its session transparently when
it finds it has been evicted, which costs one extra round trip and nothing else.
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import shutil
import time
from pathlib import Path

import httpx
import signal
import threading
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from starlette.background import BackgroundTask
import tempfile
import zipfile

from .actions import Phone
from .apps import installed_apps
from .bringup import Bridge, checks_for, diagnose, scan as scan_devices
from .config import Config, ROOT, RUNTIME
from .farm import Farm
from .crawl import (CRAWLS, DENY_BUNDLES, Crawler, Plan as CrawlPlan, is_user_app,
                    earlier_scans, list_runs, new_run_dir, plan_from_prompt, shot_filenames)
from .lock import DeviceBusy, device_lock
from .safety import Guard
from .scan import Scanner, run_extract
from .wda.client import WDAError, WDAUnreachable

log = logging.getLogger("phoneshell.server")

UI_DIR = Path(__file__).parent / "ui"
MCP_CONFIG = RUNTIME / "mcp.json"

ALLOWED_TOOLS = [
    "mcp__phoneshell__phone_status",
    "mcp__phoneshell__phone_observe",
    "mcp__phoneshell__phone_tap",
    "mcp__phoneshell__phone_long_press",
    "mcp__phoneshell__phone_type",
    "mcp__phoneshell__phone_swipe",
    "mcp__phoneshell__phone_scroll_to",
    "mcp__phoneshell__phone_open_app",
    "mcp__phoneshell__phone_open_url",
    "mcp__phoneshell__phone_press",
    "mcp__phoneshell__phone_alert",
    "mcp__phoneshell__phone_list_apps",
    "mcp__phoneshell__phone_wait_for",
    "mcp__phoneshell__phone_macro_list",
    "mcp__phoneshell__phone_macro_run",
    "mcp__phoneshell__phone_macro_save",
    "mcp__phoneshell__phone_gesture",
    "mcp__phoneshell__phone_dismiss_popup",
    "mcp__phoneshell__phone_memory",
    "mcp__phoneshell__phone_popup_help",
    "mcp__phoneshell__phone_do",
]

SYSTEM_APPEND = """
You are driving the user's real iPhone through the phoneshell tools.
If a popup, promo sheet or onboarding card is in the way, call phone_dismiss_popup
once rather than hunting for the X yourself; if it cannot clear it, phone_popup_help
lists every popup shape and the moves that close it. For anything beyond tap/type/swipe
(pinch, rotate, long press, row swipe, Control Centre, app switcher, undo) use
phone_gesture. Check phone_macro_list first: a saved macro is a route that already worked, and
replaying one is far more reliable than working the same path out again. When you
finish a task that was worth doing, offer to save it with phone_macro_save.
Prefer phone_do when you already know the route: it runs several steps in one
call and stops where it goes wrong, which is far faster than observe-then-tap
for every single step. Fall back to observe, act, observe again when exploring. Say what you are doing in one
short line before each action, in plain language, as if narrating over someone's
shoulder. Never invent screen contents you have not observed. If a tool refuses
an action because it looks irreversible, stop and ask the user in one sentence.
"""

app = FastAPI(title="phoneshell")

# The collect page is also published at appscan.blolabel.ai, and a page served from
# there has to reach this process on the loopback of the Mac holding the phone.
# Two separate rules stand in the way and both have to be answered: ordinary CORS,
# and Chrome's Private Network Access preflight, which asks a private-address
# server to opt in explicitly before a public page may talk to it.
#
# The origin list is exact, never "*": this server can drive a phone, so anything
# that can call it can drive the phone.
ALLOWED_ORIGINS = [
    "https://appscan.blolabel.ai",
    "http://127.0.0.1:8765", "http://localhost:8765",
    "http://127.0.0.1:8766", "http://localhost:8766",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["content-type"],
    # A download is fetched as a blob, and the file name rides in this header.
    expose_headers=["content-disposition"],
)


@app.middleware("http")
async def private_network_access(request, call_next):
    """Answer Chrome's private-network preflight.

    Without this header a page on https://appscan.blolabel.ai cannot even ask
    127.0.0.1 a question: the preflight is refused before the request is made, and
    the page looks like the helper is not running when it is.
    """
    if (request.method == "OPTIONS"
            and request.headers.get("access-control-request-private-network") == "true"):
        from starlette.responses import Response
        origin = request.headers.get("origin", "")
        headers = {"Access-Control-Allow-Private-Network": "true"}
        if origin in ALLOWED_ORIGINS:
            headers |= {
                "Access-Control-Allow-Origin": origin,
                "Access-Control-Allow-Methods": "GET, POST, OPTIONS",
                "Access-Control-Allow-Headers": "content-type",
                "Access-Control-Max-Age": "600",
            }
        return Response(status_code=204, headers=headers)
    response = await call_next(request)
    if request.headers.get("origin") in ALLOWED_ORIGINS:
        response.headers["Access-Control-Allow-Private-Network"] = "true"
    return response


SPEND = {"tasks": 0, "usd": 0.0, "turns": 0}
_cfg = Config.load()
_phone: Phone | None = None
_guard = Guard(_cfg)


def _passcode_stored() -> bool:
    from .secrets import has_secret
    try:
        return has_secret(Config.load().device.udid or "default")
    except Exception:
        return False


def _locked_now() -> bool:
    """Surfaced in the UI because a run against a locked phone costs full price
    and achieves nothing: it is the cheapest credit leak there is."""
    try:
        return bool(phone().wda.is_locked())
    except Exception:
        return False


def phone() -> Phone:
    global _phone
    if _phone is None:
        _phone = Phone(_cfg)
    return _phone


def write_mcp_config() -> Path:
    MCP_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    MCP_CONFIG.write_text(json.dumps({
        "mcpServers": {
            "phoneshell": {
                "command": str(ROOT / ".venv" / "bin" / "python"),
                "args": ["-m", "phoneshell.mcp_server"],
                "cwd": str(ROOT),
            }
        }
    }, indent=1))
    return MCP_CONFIG


@app.get("/")
def index() -> FileResponse:
    return FileResponse(UI_DIR / "index.html")


@app.get("/api/status")
def status() -> dict:
    cfg = Config.load()
    try:
        st = phone().wda.status()
        # Do not refresh geometry on every poll: it costs a session round trip,
        # and WebDriverAgent allows exactly one session, so a 4s poll would keep
        # evicting the agent's session while it works.
        geo = phone().wda.geometry()
        info = {}
        try:
            info = phone().wda.active_app_info()
        except WDAError:
            pass
        return {
            "connected": True,
            "device": st.get("device"),
            "ios": st.get("os", {}).get("version"),
            "wda": st.get("build", {}).get("version"),
            "transport": cfg.wda.transport,
            "point_size": [geo.point_w, geo.point_h],
            "scale": geo.scale,
            "bundle_id": info.get("bundleId"),
            "app": phone().name_for_bundle(str(info.get("bundleId") or "")),
            "mjpeg": "/stream.mjpeg",
            "mode": cfg.session.mode,
            "human_active": phone().coexist.human_has_the_phone(),
            "human_reason": phone().coexist.presence.last_reason,
            "yields": phone().coexist.presence.yields,
            "memory": cfg.memory.enabled,
            "memory_stats": phone().memory.stats(),
            "spend_usd": round(SPEND["usd"], 4),
            "spend_tasks": SPEND["tasks"],
            "locked": _locked_now(),
            "passcode_stored": _passcode_stored(),
        }
    except WDAUnreachable as exc:
        return {"connected": False, "error": str(exc),
                "hint": "not connected to a phone yet"}
    except Exception as exc:
        # A locked phone makes SpringBoard stop answering accessibility queries,
        # which surfaced here as a 500 and took the whole UI down. Status must
        # always answer: it is the thing that TELLS you the phone is locked.
        return {
            "connected": True, "degraded": True, "locked": _locked_now(),
            "error": str(exc)[:200],
            "hint": "the phone is locked; unlock it to carry on",
            "spend_usd": round(SPEND["usd"], 4), "spend_tasks": SPEND["tasks"],
            "mode": Config.load().session.mode, "memory": Config.load().memory.enabled,
            "mjpeg": "/stream.mjpeg",
        }


@app.post("/api/passcode")
async def set_passcode(payload: dict) -> dict:
    """Store the device passcode in the macOS Keychain, from the local app.

    This exists because the terminal prompt cannot be used from every context.
    The value arrives over loopback only, is never logged, never echoed back and
    never written to a file: it goes straight into the login Keychain, and the
    only thing that ever reads it is the unlock path that taps the digits on the
    phone's own keypad.
    """
    from .secrets import delete_secret, set_secret
    cfg = Config.load()
    account = cfg.device.udid or "default"
    if payload.get("forget"):
        return {"ok": delete_secret(account), "stored": False}
    code = str(payload.get("passcode") or "").strip()
    if not code:
        return {"ok": False, "error": "no passcode given"}
    if not code.isdigit():
        return {"ok": False, "error": "only a numeric passcode can be typed on the keypad"}
    if not set_secret(account, code):
        return {"ok": False, "error": "the Keychain refused to store it"}
    result = {"ok": True, "stored": True, "account": account}
    try:
        p = phone()
        if p.wda.is_locked():
            outcome = p.ensure_unlocked()
            result["tested"] = outcome.ok
            result["detail"] = outcome.detail or outcome.error
        else:
            result["detail"] = "stored; it will be used the next time the phone locks"
    except Exception as exc:
        result["detail"] = f"stored, but could not test it now: {exc}"
    return result


@app.post("/api/mode")
async def set_mode(payload: dict) -> dict:
    """takeover: the agent drives. shared: it yields the moment you touch the phone."""
    mode = str(payload.get("mode", "")).lower()
    if mode not in {"takeover", "shared"}:
        return {"ok": False, "error": "mode must be takeover or shared"}
    cfg = Config.load()
    cfg.session.mode = mode
    cfg.save()
    p = phone()
    p.cfg.session.mode = mode
    if mode == "shared":
        p.coexist.start()
    else:
        p.coexist.stop()
    return {"ok": True, "mode": mode}


@app.post("/api/memory")
async def memory_toggle(payload: dict) -> dict:
    """Turn app memory on or off, or clear it."""
    cfg = Config.load()
    if "enabled" in payload:
        cfg.memory.enabled = bool(payload["enabled"])
        cfg.save()
        phone().cfg.memory.enabled = cfg.memory.enabled
        phone().memory.enabled = cfg.memory.enabled
    if payload.get("forget"):
        phone().memory.forget(payload.get("app") or None)
    return {"ok": True, "enabled": cfg.memory.enabled, "stats": phone().memory.stats()}


@app.get("/stream.mjpeg")
async def stream() -> StreamingResponse:
    """Proxy WebDriverAgent's MJPEG stream so the browser can show it."""
    cfg = Config.load()
    url = cfg.mjpeg_url

    async def pump():
        async with httpx.AsyncClient(timeout=None) as client:
            try:
                async with client.stream("GET", url) as resp:
                    async for chunk in resp.aiter_bytes():
                        yield chunk
            except (httpx.HTTPError, asyncio.CancelledError):
                return

    return StreamingResponse(pump(), media_type="multipart/x-mixed-replace; boundary=BoundaryString")


@app.get("/api/observe")
def observe(marks: bool = False) -> dict:
    obs = phone().observe(step=int(time.time()) % 100000, force_som=marks)
    return {
        "app": obs.app,
        "bundle_id": obs.bundle_id,
        "som": obs.som,
        "alert": obs.alert,
        "text": obs.as_text(),
        "elements": [
            {"id": e.idx, "type": e.type, "text": e.text,
             "x": e.x, "y": e.y, "w": e.w, "h": e.h,
             "input": e.is_input, "enabled": e.enabled}
            for e in obs.elements
        ],
        "image": obs.image_b64,
    }


@app.post("/api/tap")
async def tap(payload: dict) -> dict:
    """Tap a point given in the phone's POINT space (the UI converts from pixels)."""
    x, y = float(payload["x"]), float(payload["y"])
    _guard.audit("manual_tap", x=x, y=y)
    phone().tap_point(x, y)
    return {"ok": True}


@app.post("/api/gesture")
async def gesture(payload: dict) -> dict:
    kind = payload.get("kind")
    p = phone()
    if kind == "home":
        p.home()
    elif kind == "back":
        p.back()
    elif kind == "swipe":
        p.swipe(payload.get("direction", "down"), distance=float(payload.get("distance", 0.6)))
    elif kind == "drag":
        p.wda.swipe_w3c(payload["x1"], payload["y1"], payload["x2"], payload["y2"])
    elif kind == "type":
        p.type_text(payload.get("text", ""), submit=bool(payload.get("submit")))
    elif kind == "open_app":
        p.open_app(payload.get("name", ""))
    elif kind == "open_url":
        p.open_url(payload.get("url", ""))
    else:
        return {"ok": False, "error": f"unknown gesture {kind!r}"}
    _guard.audit("manual_gesture", **payload)
    return {"ok": True}


class Run:
    """One agent run, so it can actually be stopped.

    Cancelling the asyncio task is not enough: the brain is a child process, and
    the MCP server is a grandchild. Both have to be signalled, or "stop" leaves a
    process still driving the phone.
    """

    def __init__(self):
        self.proc: asyncio.subprocess.Process | None = None
        self.stopped = False

    async def kill(self) -> None:
        self.stopped = True
        proc = self.proc
        if proc is None or proc.returncode is not None:
            return
        try:
            # Signal the whole process group: claude spawns the MCP server, which
            # holds the WebDriverAgent session.
            os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            proc.terminate()
        try:
            await asyncio.wait_for(proc.wait(), timeout=4)
        except asyncio.TimeoutError:
            try:
                os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                proc.kill()


async def run_agent(task: str, ws: WebSocket, model: str, run: "Run") -> None:
    """Spawn the brain and relay its stream to the browser."""
    claude = shutil.which("claude") or str(Path.home() / ".local/bin/claude")
    if not Path(claude).exists():
        await ws.send_json({"type": "error", "text": "the `claude` CLI is not installed on this Mac"})
        return
    cfg_path = write_mcp_config()
    cmd = [
        claude, "-p", task,
        "--output-format", "stream-json", "--verbose",
        "--mcp-config", str(cfg_path),
        "--allowedTools", *ALLOWED_TOOLS,
        "--append-system-prompt", SYSTEM_APPEND,
        "--model", model,
        "--max-turns", "60",
    ]
    proc = await asyncio.create_subprocess_exec(
        *cmd, cwd=str(ROOT),
        stdin=asyncio.subprocess.DEVNULL,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,   # own process group, so stop can signal all of it
    )
    run.proc = proc
    t0 = time.time()
    await ws.send_json({"type": "started", "task": task, "at": t0})
    assert proc.stdout
    pending: dict[str, float] = {}
    while True:
        line = await proc.stdout.readline()
        if not line:
            break
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        now = time.time()
        if kind == "assistant":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "text" and block.get("text", "").strip():
                    await ws.send_json({"type": "say", "text": block["text"],
                                        "at": now, "elapsed": now - t0})
                elif block.get("type") == "tool_use":
                    name = str(block.get("name", "")).replace("mcp__phoneshell__phone_", "")
                    pending[str(block.get("id") or name)] = now
                    await ws.send_json({
                        "type": "action", "name": name, "input": block.get("input", {}),
                        "id": str(block.get("id") or name), "at": now, "elapsed": now - t0,
                    })
        elif kind == "user":
            for block in event.get("message", {}).get("content", []):
                if block.get("type") == "tool_result":
                    started = pending.pop(str(block.get("tool_use_id") or ""), None)
                    content = block.get("content")
                    text = ""
                    if isinstance(content, list):
                        text = " ".join(c.get("text", "") for c in content if isinstance(c, dict))
                    elif isinstance(content, str):
                        text = content
                    await ws.send_json({
                        "type": "result", "text": text[:400],
                        "id": str(block.get("tool_use_id") or ""),
                        "took": (now - started) if started else None,
                        "at": now, "elapsed": now - t0,
                    })
        elif kind == "result":
            SPEND["tasks"] += 1
            SPEND["usd"] += float(event.get("total_cost_usd") or 0)
            SPEND["turns"] += int(event.get("num_turns") or 0)
            await ws.send_json({
                "type": "done",
                "text": event.get("result", ""),
                "at": now, "elapsed": now - t0,
                "cost": event.get("total_cost_usd"),
                "turns": event.get("num_turns"),
                "duration_ms": event.get("duration_ms"),
                "spend_usd": round(SPEND["usd"], 4),
                "spend_tasks": SPEND["tasks"],
            })
    err = (await proc.stderr.read()).decode()[-600:] if proc.stderr else ""
    await proc.wait()
    if run.stopped:
        await ws.send_json({"type": "stopped", "elapsed": time.time() - t0})
    elif proc.returncode != 0:
        await ws.send_json({"type": "error", "text": err or f"agent exited {proc.returncode}",
                            "elapsed": time.time() - t0})


@app.websocket("/ws")
async def websocket(ws: WebSocket) -> None:
    await ws.accept()
    task_handle: asyncio.Task | None = None
    run: Run | None = None
    try:
        while True:
            msg = await ws.receive_json()
            if msg.get("type") == "task":
                if task_handle and not task_handle.done():
                    await ws.send_json({"type": "error", "text": "a task is already running"})
                    continue
                model = msg.get("model") or Config.load().brain.model
                run = Run()
                task_handle = asyncio.create_task(run_agent(msg["text"], ws, model, run))
            elif msg.get("type") == "stop":
                if run:
                    await run.kill()
                if task_handle and not task_handle.done():
                    task_handle.cancel()
                await ws.send_json({"type": "stopped"})
    except WebSocketDisconnect:
        if run:
            await run.kill()
        if task_handle:
            task_handle.cancel()


# ---------------------------------------------------------------- collect page
#
# A second page, for one job: connect a phone from the browser and have it walk
# every screen it can reach, collecting a screenshot of each. The chat page above
# delegates a TASK to a model; this one runs a deterministic crawl with no model
# in it at all, which is why it can run for an hour for nothing.

BRIDGE = Bridge()
CRAWL: dict[str, object] = {"crawler": None, "thread": None, "run": None}


_scanner = Scanner()


@app.get("/scan")
def scan_page() -> FileResponse:
    return FileResponse(UI_DIR / "scan.html")


@app.post("/api/scan/start")
async def scan_start(payload: dict | None = None) -> dict:
    """Enqueue-style: the scan runs in a server thread and the page renders its
    state, so a reload or a closed tab never stops or loses it."""
    scan = _scanner.start((payload or {}).get("label", ""))
    return {"ok": True, "scan": scan.summary()}


@app.post("/api/scan/stop")
async def scan_stop() -> dict:
    scan = _scanner.stop()
    return {"ok": True, "scan": scan.summary() if scan else None}


@app.get("/api/scans")
def scan_history() -> dict:
    cur = _scanner.current
    return {"current": cur.summary() if cur else None, "scans": Scanner.history()}


def _scan_or_404(sid: str):
    scan = _scanner.get(sid)
    if scan is None:
        return None, JSONResponse({"error": "no such scan"}, status_code=404)
    return scan, None


@app.get("/api/scan/{sid}")
def scan_state(sid: str, since: int = 0):
    """Lines are returned whole (they are re-ordered as you scroll back up), frames
    only past `since` so the thumbnail strip polls cheaply."""
    scan, err = _scan_or_404(sid)
    if err:
        return err
    return {**scan.summary(), "frame_count": len(scan.frames), "events": scan.events, "extract": scan.extract,
            "lines": [{"text": l.text, "frame": l.frame, "gap": l.gap} for l in scan.content()],
            "frames": [f.__dict__ for f in scan.frames if f.n > since]}


@app.get("/api/scan/{sid}/frame/{n}")
def scan_frame(sid: str, n: int) -> Response:
    scan, err = _scan_or_404(sid)
    if err:
        return err
    path = scan.dir / "frames" / f"{n:04d}.jpg"
    return FileResponse(path) if path.exists() else JSONResponse({"error": "no such frame"}, status_code=404)


@app.get("/api/scan/{sid}/export")
def scan_export(sid: str, fmt: str = "txt") -> Response:
    import csv
    import io as _io
    scan, err = _scan_or_404(sid)
    if err:
        return err
    name = f"scan-{scan.id}"
    headers = lambda ext: {"Content-Disposition": f'attachment; filename="{name}.{ext}"'}
    if fmt == "json":
        body = json.dumps({"scan": scan.summary(), "lines": [l.text for l in scan.content()],
                           "extract": scan.extract}, ensure_ascii=False, indent=1)
        return Response(body, media_type="application/json", headers=headers("json"))
    if fmt == "csv":
        ex = scan.extract
        if ex.get("status") != "done":
            return JSONResponse({"error": "run Extract first; CSV is the structured table"}, status_code=409)
        buf = _io.StringIO()
        w = csv.DictWriter(buf, fieldnames=ex["columns"], extrasaction="ignore")
        w.writeheader()
        for row in ex["rows"]:
            w.writerow({c: row.get(c, "") for c in ex["columns"]})
        return Response(buf.getvalue(), media_type="text/csv", headers=headers("csv"))
    return Response(scan.text(), media_type="text/plain; charset=utf-8", headers=headers("txt"))


@app.post("/api/scan/{sid}/extract")
async def scan_extract(sid: str, payload: dict | None = None):
    """Queues the LLM pass on a worker thread and returns at once; the page polls
    scan.extract for queued -> running -> done | failed."""
    scan, err = _scan_or_404(sid)
    if err:
        return err
    if scan.extract.get("status") == "running":
        return JSONResponse({"error": "an extraction is already running for this scan"}, status_code=409)
    instruction = ((payload or {}).get("instruction") or "").strip()
    scan.extract = {"status": "running", "instruction": instruction, "started": time.time()}
    threading.Thread(target=run_extract, args=(scan, instruction), daemon=True).start()
    return {"ok": True}


@app.delete("/api/scan/{sid}")
def scan_delete(sid: str):
    if _scanner.current and _scanner.current.id == sid:
        if _scanner.current.status == "running":
            return JSONResponse({"error": "stop the scan first"}, status_code=409)
        _scanner.current = None
    return {"ok": Scanner.delete(sid)}


@app.get("/collect")
def collect_page() -> FileResponse:
    return FileResponse(UI_DIR / "collect.html")


@app.get("/api/devices")
def devices() -> dict:
    """Every phone this Mac can see, and what this process has a hold on."""
    return {"devices": [c.as_dict() for c in scan_devices()], "bridge": BRIDGE.state()}


# --------------------------------------------------------------------------
# The farm: every phone on this Mac, not just the configured one.
# --------------------------------------------------------------------------

FARM = Farm(BRIDGE)


@app.get("/farm")
def farm_page() -> FileResponse:
    return FileResponse(UI_DIR / "farm.html")


@app.get("/api/farm")
def farm_state(deep: bool = False) -> dict:
    """Every phone, its slot, its ports and how it is.

    `deep` costs an HTTP round trip to each phone we hold a bridge for, so the
    grid's timer asks shallowly and a card someone opened asks deeply.
    """
    members = FARM.members(deep=deep)
    return {
        "members": [m.as_dict() for m in members],
        "live": sum(1 for m in members if m.bridge.get("up")),
        "seen": len(members),
        "configured": Config.load().device.udid,
    }


@app.get("/api/farm/screenshot")
def farm_screenshot(udid: str) -> Response:
    """One still from one phone. The grid polls this; it never opens a stream."""
    png = FARM.screenshot(udid)
    if png is None:
        return Response(status_code=503, content=b"", media_type="image/png")
    return Response(content=png, media_type="image/png",
                    headers={"Cache-Control": "no-store"})


@app.post("/api/farm/connect")
async def farm_connect(payload: dict) -> dict:
    """Bring one phone up on its own ports, and report every rung of the ladder."""
    udid = str(payload.get("udid") or "").strip()
    if not udid:
        return {"ok": False, "error": "which phone? pass a udid"}
    relaunch = bool(payload.get("relaunch"))

    def work() -> list[dict]:
        return list(FARM.connect(udid, relaunch=relaunch))

    try:
        events = await asyncio.to_thread(work)
    except Exception as exc:
        return {"ok": False, "error": str(exc)[:300]}
    failed = [e for e in events if not e.get("ok")]
    return {
        "ok": not failed,
        "events": events,
        "member": next((m.as_dict() for m in FARM.members(deep=True) if m.udid == udid), None),
    }


@app.post("/api/farm/disconnect")
async def farm_disconnect(payload: dict) -> dict:
    udid = str(payload.get("udid") or "").strip()
    if not udid:
        return {"ok": False, "error": "which phone? pass a udid"}
    return await asyncio.to_thread(
        FARM.disconnect, udid, bool(payload.get("stop_runner"))
    )


@app.get("/api/checks")
def device_checks(udid: str = "") -> dict:
    return {"checks": checks_for(udid or None)}


@app.get("/api/health")
def health() -> dict:
    """Why the phone is not usable, in one sentence, with the remedy.

    The page turns a CHANGE in this into a message in the chat, so the person
    using AppScan finds out that the cable came out from AppScan, rather than
    from a scan that quietly stops producing screenshots.
    """
    return diagnose(BRIDGE)


@app.on_event("startup")
async def _heal_on_startup() -> None:
    """Reconnect a bridge the last helper took with it when it exited."""
    def work() -> None:
        try:
            for event in BRIDGE.heal():
                log.info("heal: %s", event.get("detail") or event.get("title"))
        except Exception as exc:                       # a failed heal is not fatal
            log.warning("heal failed: %s", exc)
    threading.Thread(target=work, name="bridge-heal", daemon=True).start()


@app.post("/api/disconnect")
async def disconnect(payload: dict | None = None) -> dict:
    stop_runner = bool((payload or {}).get("stop_runner"))
    return {"ok": True, "bridge": BRIDGE.disconnect(stop_runner=stop_runner)}


@app.get("/api/apps")
def apps_list() -> dict:
    """The app catalog, for the picker. Cheap enough to ask for on page load."""
    try:
        catalog = installed_apps(Config.load())
    except Exception as exc:
        return {"apps": [], "error": str(exc)[:200], "live": False}
    # With no phone attached, installed_apps falls back to a short list of common
    # bundle ids. That is a useful default and a terrible thing to present as
    # "apps on this phone", so the page is told which one it is holding.
    live = diagnose(BRIDGE).get("state") in {"ready", "locked", "runner_dead", "stale_tunnel", "no_tunnel"}
    deny = DENY_BUNDLES | set(Config.load().safety.denied_bundle_ids)
    apps = [
        {"name": name, "bundle": bundle, "denied": bundle in deny,
         "apple": bundle.startswith("com.apple.")}
        for name, bundle in sorted(catalog.items(), key=lambda kv: kv[0].lower())
        if is_user_app(name, bundle)
    ]
    return {"apps": apps, "total": len(catalog), "live": live}


@app.post("/api/plan")
async def plan_endpoint(payload: dict) -> dict:
    """Turn a plain-language prompt into a plan, and say what was understood.

    Separate from starting the crawl on purpose: the page shows you the plan it
    read out of your sentence before anything touches the phone.
    """
    prompt = str(payload.get("prompt") or "")
    try:
        catalog = installed_apps(Config.load())
    except Exception:
        catalog = None
    # The app picked in the page is the DEFAULT, not an override: an app named in
    # the sentence still wins, and a sentence that names none keeps the pick
    # instead of falling back to Settings.
    picked = (str(payload.get("label") or payload["app"]), str(payload["app"])) if payload.get("app") else None
    plan, notes = plan_from_prompt(prompt, Config.load(), catalog, default_app=picked,
                                   fresh=bool(payload.get("fresh")))
    for key in ("max_screens", "max_depth", "max_minutes", "variant_cap", "scroll_cap",
                "per_app_screens", "per_app_minutes"):
        if key in payload and payload[key] not in (None, ""):
            setattr(plan, key, type(getattr(plan, key))(payload[key]))
    if payload.get("apps"):
        plan.apps = [str(a) for a in payload["apps"]]
    earlier = earlier_scans(plan.app) if plan.scope == "app" else {"runs": 0, "screens": 0}
    return {"plan": plan.as_dict(), "notes": notes, "earlier": earlier}


@app.get("/api/crawls")
def crawls() -> dict:
    return {"runs": list_runs()}


@app.get("/api/crawl/{run}/manifest")
def crawl_manifest(run: str) -> dict:
    path = (CRAWLS / run / "manifest.json").resolve()
    if not str(path).startswith(str(CRAWLS.resolve())) or not path.is_file():
        return {"error": "no such run"}
    return json.loads(path.read_text())


def _run_dir(run: str) -> Path | None:
    out = (CRAWLS / run).resolve()
    return out if out.parent == CRAWLS.resolve() and out.is_dir() else None


def _run_label(out: Path) -> tuple[str, str]:
    """`20260911-123729-talika` -> ("talika", "20260911-123729")."""
    parts = out.name.split("-", 2)
    return (parts[2] if len(parts) == 3 else out.name), "-".join(parts[:2])


def _shot_file(out: Path, rel: str) -> Path | None:
    """The full PNG, or its thumbnail when the run was told not to keep PNGs."""
    for cand in (out / rel, out / rel.replace("shots/", "thumbs/", 1).replace(".png", ".jpg")):
        cand = cand.resolve()
        if cand.is_file() and cand.parent in (out / "shots", out / "thumbs"):
            return cand
    return None


@app.get("/api/crawl/{run}/shot")
def crawl_shot(run: str, shot: str):
    """One screenshot at full resolution, as a download named for its screen."""
    out = _run_dir(run)
    path = _shot_file(out, shot) if out else None
    if not out or not path:
        return JSONResponse({"error": "no such screenshot"}, status_code=404)
    name = Path(shot_filenames(out).get(shot, path.name)).stem
    return FileResponse(path, filename=f"{_run_label(out)[0]}-{name}{path.suffix}")


@app.get("/api/crawl/{run}/zip")
def crawl_zip(run: str):
    """Every screenshot in a run, as one zip a person can open without AppScan.

    Stored, not deflated: PNGs are already compressed, and deflating them again
    costs seconds per hundred screens for a file a few percent smaller.
    """
    out = _run_dir(run)
    if not out:
        return JSONResponse({"error": "no such scan"}, status_code=404)
    label, stamp = _run_label(out)
    folder = f"{label}-{stamp}" if stamp else label
    fd, tmp = tempfile.mkstemp(suffix=".zip")
    os.close(fd)
    count = 0
    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_STORED) as z:
        for rel, name in shot_filenames(out).items():
            path = _shot_file(out, rel)
            if path:
                z.write(path, f"{folder}/{Path(name).stem}{path.suffix}")
                count += 1
        if (out / "manifest.json").is_file():
            z.write(out / "manifest.json", f"{folder}/manifest.json")
    if not count:
        os.unlink(tmp)
        return JSONResponse({"error": "this scan has no screenshots yet"}, status_code=404)
    return FileResponse(tmp, media_type="application/zip", filename=f"{folder}.zip",
                        background=BackgroundTask(os.unlink, tmp))


if CRAWLS.exists() or CRAWLS.mkdir(parents=True, exist_ok=True) is None:
    # The collected screenshots, served to the gallery straight off disk.
    app.mount("/collected", StaticFiles(directory=str(CRAWLS)), name="collected")


@app.websocket("/ws/collect")
async def collect_ws(ws: WebSocket) -> None:
    """One socket for the whole page: bring the bridge up, then run a crawl.

    The crawler is synchronous and blocking by design (it is a loop of HTTP calls
    to the phone), so it runs in a worker thread and posts its events back onto
    the event loop. That keeps the socket answering `stop` while a tap is in
    flight, which a plain `await to_thread(...)` would not.
    """
    await ws.accept()
    loop = asyncio.get_running_loop()
    queue: asyncio.Queue = asyncio.Queue()
    pump: asyncio.Task | None = None

    async def drain() -> None:
        while True:
            event = await queue.get()
            try:
                await ws.send_json(event)
            except Exception:
                return

    pump = asyncio.create_task(drain())

    def push(event: dict) -> None:
        loop.call_soon_threadsafe(queue.put_nowait, event)

    try:
        while True:
            msg = await ws.receive_json()
            kind = msg.get("type")

            if kind == "connect":
                udid = msg.get("udid") or None
                relaunch = bool(msg.get("relaunch"))
                await ws.send_json({"type": "connecting", "udid": udid})

                def bring_up() -> None:
                    try:
                        for event in BRIDGE.connect(udid, relaunch=relaunch):
                            push(event)
                    except Exception as exc:           # noqa: BLE001 - reported, never raised at a socket
                        push({"type": "step", "step": "error", "ok": False, "detail": str(exc)[:300]})
                    push({"type": "connected", "bridge": BRIDGE.state()})

                await asyncio.to_thread(bring_up)

            elif kind == "disconnect":
                # Disconnect means the automation goes off, banner and all. Dropping
                # only the tunnel left the runner on the phone, still "running".
                state = await asyncio.to_thread(BRIDGE.park)
                await ws.send_json({"type": "connected", "bridge": state})
                await ws.send_json({"type": "parked", "why": "asked"})

            elif kind == "crawl":
                thread = CRAWL.get("thread")
                if isinstance(thread, threading.Thread) and thread.is_alive():
                    await ws.send_json({"type": "error", "text": "a crawl is already running"})
                    continue
                plan = CrawlPlan(**{k: v for k, v in (msg.get("plan") or {}).items()
                                    if k in CrawlPlan.__dataclass_fields__})
                out = new_run_dir(plan)
                CRAWL["run"] = out.name
                _guard.audit("crawl_start", run=out.name, plan=plan.as_dict())
                await ws.send_json({"type": "run", "run": out.name, "plan": plan.as_dict()})

                def work() -> None:
                    ran = False
                    try:
                        with device_lock(f"collect {out.name}"):
                            crawler = Crawler(phone(), plan, out, emit=push)
                            CRAWL["crawler"] = crawler
                            ran = True
                            crawler.run()
                    except DeviceBusy as exc:
                        push({"type": "error", "text": str(exc)})
                    except Exception as exc:           # noqa: BLE001
                        log.exception("crawl failed")
                        push({"type": "error", "text": f"{type(exc).__name__}: {exc}"[:400]})
                    finally:
                        CRAWL["crawler"] = None
                        # Hand the phone back when the scan ends, however it ends. Left
                        # up, the runner sits idle under iOS's "Automation Running"
                        # banner with nothing moving. (Not after DeviceBusy: then the
                        # phone is someone else's, and so is the automation.)
                        if ran:
                            try:
                                BRIDGE.park()
                                push({"type": "parked", "why": "scan"})
                            except Exception as exc:   # noqa: BLE001 - never fail a finished scan
                                log.warning("could not turn the automation off: %s", exc)

                worker = threading.Thread(target=work, name=f"crawl-{out.name}", daemon=True)
                CRAWL["thread"] = worker
                worker.start()

            elif kind == "stop":
                crawler = CRAWL.get("crawler")
                if isinstance(crawler, Crawler):
                    crawler.stop("stopped from the page")
                    await ws.send_json({"type": "stopping"})
                else:
                    await ws.send_json({"type": "error", "text": "nothing is running"})

            elif kind == "reply":
                # The other half of the chat: the scan asked a question and is
                # blocked on a threading.Event waiting for this.
                crawler = CRAWL.get("crawler")
                text = str(msg.get("text") or "")
                if isinstance(crawler, Crawler):
                    crawler.answer(text)
                else:
                    await ws.send_json({
                        "type": "say",
                        "text": "No scan is running, so there is nothing waiting on an answer. "
                                "Pick an app on the left and press Start.",
                    })

    except WebSocketDisconnect:
        # A closed tab does NOT stop a crawl: it is a long job, and the page can
        # be reopened and reattached to the run that is still writing to disk.
        pass
    finally:
        if pump:
            pump.cancel()


if UI_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(UI_DIR)), name="static")


def main() -> None:
    import uvicorn
    cfg = Config.load()
    write_mcp_config()
    uvicorn.run(app, host=cfg.server_host, port=cfg.server_port, log_level="warning")


if __name__ == "__main__":
    main()
