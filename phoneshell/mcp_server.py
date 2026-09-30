"""phoneshell as an MCP server: any MCP client becomes a phone operator.

This is the path that needs no API key. Point Claude Code (or Claude Desktop, or
anything else that speaks MCP) at this server and the model on the other end can
see and drive the iPhone directly.

The contract with the model is deliberately narrow:
  * it never invents coordinates, it names an id from the last observation
  * ids are resolved here, against the snapshot the model actually saw
  * anything that looks irreversible comes back as a refusal until the call is
    repeated with confirmed=true, so the decision is visible in the transcript
"""
from __future__ import annotations

import logging
import time
from typing import Any, Literal

from mcp.types import ContentBlock, ImageContent, TextContent
from mcp.server.mcpserver import MCPServer

from .actions import Phone
from .agent import macros as macro_lib
from .agent import overlays
from .agent import playbook
from .config import Config
from .perception.tree import find_by_text
from .safety import Guard
from .wda.client import WDAError, WDAUnreachable

log = logging.getLogger("phoneshell.mcp")

INSTRUCTIONS = """\
You are operating a real iPhone that belongs to a real person, over a live connection.

How to work:
1. Call phone_observe first. It returns the screen as a table of elements plus a
   picture. Every row has an id.
2. Act by id (phone_tap, phone_type). Never guess coordinates; if what you want
   is not in the table, scroll or re-observe.
3. After every action you get the new screen back. Read it before deciding again.
4. If the same screen comes back twice after the same action, stop repeating it
   and try a different route.
5. Prefer phone_open_app and phone_open_url over tapping your way through the
   home screen. A deep link is one deterministic step instead of five fragile ones.
6. Anything that spends money, sends a message, or deletes something will be
   refused once and will tell you so. Show the user what you are about to do,
   and only then repeat the call with confirmed=true.

This is someone's actual phone, with their real accounts, money and contacts on
it. Slow down and read the screen before you tap.
"""

mcp = MCPServer(
    name="phoneshell",
    instructions=INSTRUCTIONS,
    version="0.1.0",
)


class Bridge:
    """Lazily-connected phone with the last observation kept for id resolution."""

    def __init__(self):
        self.cfg = Config.load()
        self._phone: Phone | None = None
        self.guard = Guard(self.cfg)
        self.last_obs = None
        self.step = 0
        self.recorder = macro_lib.Recorder()
        self.pending_transition: tuple | None = None
        self.last_signature: str | None = None
        self.seen_screens: set[str] = set()   # fingerprints already pictured this session

    @property
    def phone(self) -> Phone:
        if self._phone is None:
            self._phone = Phone(self.cfg)
        return self._phone

    def observe(self, force_som: bool = False, include_image: bool = True):
        self.step += 1
        obs = self.phone.observe(step=self.step, force_som=force_som, include_image=include_image)
        # Close the loop on the previous action: this screen is where it led.
        if self.pending_transition is not None:
            bundle, before, element = self.pending_transition
            self.pending_transition = None
            try:
                self.phone.memory.record_action(bundle, before, element, obs.bundle_id, obs.elements)
            except Exception as exc:
                log.debug("could not record the route: %s", exc)
        self.last_obs = obs
        return obs

    def element(self, element_id: int):
        if self.last_obs is None:
            raise ValueError("call phone_observe first, there is no current screen")
        for e in self.last_obs.elements:
            if e.idx == element_id:
                return e
        raise ValueError(
            f"there is no element [{element_id}] on the screen that was last observed "
            f"(it had {len(self.last_obs.elements)} elements). Observe again."
        )


BRIDGE = Bridge()


def _turn_or_message() -> list | None:
    """In shared mode the owner has right of way."""
    phone = BRIDGE.phone
    if phone.cfg.session.mode != "shared":
        return None
    if not phone.coexist.human_has_the_phone():
        phone.coexist.claim()
        return None
    ok, note = phone.coexist.wait_for_turn(timeout=phone.cfg.session.yield_wait_timeout)
    if ok:
        phone.coexist.claim()
        return None
    return _text(
        f"PAUSED: the phone is being used by its owner right now ({phone.coexist.presence.last_reason}). "
        "Shared mode gives them right of way. Tell the user the task is waiting for the phone, "
        "and stop; do not keep tapping."
    )


def _unlocked_or_message() -> list | None:
    """Every action goes through here. A locked phone silently swallows taps, so
    it is better to unlock it (when a passcode is stored) or say so plainly."""
    try:
        if not BRIDGE.phone.wda.is_locked():
            return None
    except WDAError:
        return None
    result = BRIDGE.phone.ensure_unlocked()
    BRIDGE.guard.audit("auto_unlock", ok=result.ok, detail=result.detail or result.error)
    if result.ok:
        return None
    return _text(
        f"THE PHONE IS LOCKED and could not be unlocked. {result.error}\n"
        "Stop here and tell the user: nothing you do will have any effect until it is unlocked."
    )


def _screen_payload(obs, header: str = "", allow_image: bool = True) -> list[ContentBlock]:
    text = (header + "\n" if header else "") + obs.as_text()
    out: list[ContentBlock] = [TextContent(type="text", text=text)]
    # One picture per distinct screen per session. Seeing the same layout again
    # teaches the model nothing and costs ~700 tokens every time it is re-read.
    if obs.image_b64 and allow_image:
        if obs.signature in BRIDGE.seen_screens:
            out[0] = TextContent(
                type="text",
                text=text + "\n(no picture: you have already seen this screen this session)",
            )
        else:
            BRIDGE.seen_screens.add(obs.signature)
            out.append(ImageContent(type="image", data=obs.image_b64,
                                    mimeType=obs.image_media_type))
    return out


def _text(message: str) -> list[ContentBlock]:
    return [TextContent(type="text", text=message)]


def _anchor_text() -> str:
    """A distinctive piece of the current screen, used to verify a replay is on course."""
    if BRIDGE.last_obs is None:
        return ""
    for e in BRIDGE.last_obs.elements:
        if e.text and len(e.text) > 3 and e.type in {"StaticText", "Button", "NavigationBar"}:
            return e.text[:60]
    return ""


def _act_and_report(header: str) -> list[ContentBlock]:
    """Report the result of an action as cheaply as the situation allows.

    Returning the whole screen after every action is what makes a long task
    expensive: each dump stays in the conversation and is re-read on every
    subsequent turn, so a 20-step task pays for the same tree twenty times. Most
    actions do not need it. If the screen did not change there is nothing to say,
    and if it did the model gets the new screen once.
    """
    previous = BRIDGE.last_signature
    obs = BRIDGE.observe()
    try:
        BRIDGE.phone.coexist.release()
    except Exception:
        pass
    if BRIDGE.phone.cfg.session.mode == "shared":
        header += "  [shared mode: yields if the owner picks the phone up]"

    unchanged = previous is not None and obs.signature == previous
    BRIDGE.last_signature = obs.signature
    if unchanged and not obs.alert:
        return _text(
            f"{header}\nTHE SCREEN DID NOT CHANGE. Same {obs.app} screen, same element ids as "
            "before, so you can act on them again. If you expected a change, that action did "
            "not do what you thought: try something different rather than repeating it."
        )
    return _screen_payload(obs, header)


@mcp.tool(
    structured_output=False,
    description="Check the connection to the phone: is the bridge up, which device, which app is in front."
)
def phone_status() -> str:
    cfg = BRIDGE.cfg
    try:
        status = BRIDGE.phone.wda.status()
    except WDAUnreachable as exc:
        return (
            f"NOT CONNECTED. {exc}\n"
            f"Expected WebDriverAgent at {cfg.wda_base_url}.\n"
            "Run `phoneshell up` on the Mac to start the bridge, and check the phone is "
            "plugged in or on the same wifi."
        )
    info = {}
    try:
        info = BRIDGE.phone.wda.active_app_info()
    except WDAError:
        pass
    ios = status.get("os", {})
    return (
        f"connected to {status.get('device')} running iOS {ios.get('version')}\n"
        f"WebDriverAgent {status.get('build', {}).get('version')} at {cfg.wda_base_url}\n"
        f"foreground app: {info.get('bundleId', 'unknown')}\n"
        f"transport: {cfg.wda.transport}"
    )


@mcp.tool(
    structured_output=False,
    description=(
        "Look at the phone screen. Returns a table of the elements currently on screen "
        "(id, type, text, centre point, size, flags) plus a picture of the screen. "
        "USE THIS SPARINGLY. Every action tool already hands back the screen as it is "
        "afterwards, with fresh element ids, so calling observe after a tap or a swipe buys "
        "nothing and costs a whole round trip. Call it to look before your first action, or "
        "when you are genuinely lost. "
        "Set force_marks=true to get numbered boxes drawn on the picture when you need to "
        "see exactly where an element is."
    )
)
def phone_observe(force_marks: bool = False) -> list[ContentBlock]:
    obs = BRIDGE.observe(force_som=force_marks)
    return _screen_payload(obs)


@mcp.tool(
    structured_output=False,
    description=(
        "Tap an element by its id from the last observation. "
        "Returns the screen as it is afterwards, with fresh ids, so do not call phone_observe next. "
        "Anything that looks irreversible (pay, order, send, delete) is refused the first "
        "time: tell the user what you are about to do, then call again with confirmed=true."
    )
)
def phone_tap(id: int, confirmed: bool = False) -> list[ContentBlock]:
    blocked = _unlocked_or_message() or _turn_or_message()
    if blocked:
        return blocked
    element = BRIDGE.element(id)
    verdict = BRIDGE.guard.classify_tap(element)
    if verdict.needs_confirmation and not confirmed:
        BRIDGE.guard.audit("tap_blocked", element=element.describe(), reason=verdict.reason)
        return _text(
            f"NOT DONE, this needs the user to agree first.\n{verdict.reason}\n"
            f"Element: {element.describe()}\n"
            "Tell the user exactly what this will do, and if they say yes call "
            f"phone_tap(id={id}, confirmed=true)."
        )
    from .agent.verify import consistency_gate
    snap_png = getattr(BRIDGE.phone, "_last_png", None)
    gate = consistency_gate(snap_png, element, BRIDGE.phone.wda.geometry().scale)
    if not gate.ok:
        return _text(f"NOT DONE. {gate.reason}. Observe again and pick the element off the fresh screen.")
    BRIDGE.guard.audit("tap", element=element.describe(), confirmed=confirmed)
    anchor = _anchor_text()
    if BRIDGE.last_obs is not None:
        BRIDGE.pending_transition = (BRIDGE.last_obs.bundle_id, BRIDGE.last_obs.elements, element)
    BRIDGE.phone.tap_element(element)
    BRIDGE.recorder.record("tap", {}, anchor=anchor, target_text=element.text)
    return _act_and_report(f"tapped [{id}] {element.type} {element.text!r}")


@mcp.tool(structured_output=False, description="Press and hold an element by id, for context menus and drag handles. Returns the screen as it is afterwards, with fresh ids, so do not call phone_observe next. ")
def phone_long_press(id: int, seconds: float = 1.0) -> list[ContentBlock]:
    blocked = _unlocked_or_message() or _turn_or_message()
    if blocked:
        return blocked
    element = BRIDGE.element(id)
    BRIDGE.guard.audit("long_press", element=element.describe())
    BRIDGE.phone.long_press(element, duration=seconds)
    return _act_and_report(f"held [{id}] {element.text!r} for {seconds}s")


@mcp.tool(
    structured_output=False,
    description=(
        "Type text. Give the id of the field to type into (it will be focused first). "
        "Returns the screen as it is afterwards, with fresh ids, so do not call phone_observe next. "
        "Omit the id only to append to a field that already has the keyboard. "
        "submit=true presses return afterwards."
    )
)
def phone_type(text: str, id: int | None = None, submit: bool = False, clear_first: bool = False) -> list[ContentBlock]:
    blocked = _unlocked_or_message() or _turn_or_message()
    if blocked:
        return blocked
    element = BRIDGE.element(id) if id is not None else None
    BRIDGE.guard.audit("type", text=text[:200], into=element.describe() if element else None)
    anchor = _anchor_text()
    typed = BRIDGE.phone.type_text(text, into=element, submit=submit, clear_first=clear_first)
    BRIDGE.recorder.record("type", {"text": text, "submit": submit}, anchor=anchor,
                           target_text=element.text if element else "")
    header = f"typed {text!r}{' and pressed return' if submit else ''}"
    if not typed.ok:
        # The verb layer checked whether the text actually reached a field. If it
        # did not, saying so here is the whole point: an agent that believes it
        # typed will move on and report success for work it never did.
        header = typed.detail
    return _act_and_report(header)


@mcp.tool(
    structured_output=False,
    description=(
        "Swipe the screen. direction is the direction the CONTENT moves: 'down' shows you "
        "what is below the fold. distance is a fraction of the screen, 0.1 to 0.9."
    )
)
def phone_swipe(
    direction: Literal["up", "down", "left", "right"],
    distance: float = 0.6,
    speed: Literal["slow", "normal", "fast"] = "normal",
) -> list[ContentBlock]:
    blocked = _unlocked_or_message() or _turn_or_message()
    if blocked:
        return blocked
    BRIDGE.phone.swipe(direction, distance=distance, speed=speed)
    BRIDGE.recorder.record("swipe", {"direction": direction, "distance": distance})
    return _act_and_report(f"swiped {direction} {int(distance * 100)}%")


@mcp.tool(
    structured_output=False,
    description=(
        "Swipe repeatedly until some text appears on screen. Stops early if the screen "
        "stops changing, so it will not spin forever at the end of a list."
    )
)
def phone_scroll_to(text: str, max_swipes: int = 8, direction: Literal["up", "down"] = "down") -> list[ContentBlock]:
    result = BRIDGE.phone.scroll_to_text(text, max_swipes=max_swipes, direction=direction)
    header = result.detail if result.ok else f"NOT FOUND: {result.error}"
    return _act_and_report(header)


@mcp.tool(
    structured_output=False,
    description=(
        "Put a picture or video from this Mac into the phone's camera roll, so an app on the "
        "phone can pick it. Takes a path on the Mac, e.g. '/Users/me/shot.jpg'. Accepts jpg, "
        "jpeg, png, heic, gif, mov, mp4, m4v, up to 100 MB. Use this before any task that has "
        "to upload, post or attach a photo: nothing else here can put a file on the phone. "
        "It adds to the library and never deletes or reads anything already there."
    )
)
def phone_push_media(path: str) -> list[ContentBlock]:
    blocked = _unlocked_or_message() or _turn_or_message()
    if blocked:
        return blocked
    result = BRIDGE.phone.push_media(path)
    if not result.ok:
        return _text(f"could not push {path!r}: {result.error}")
    BRIDGE.guard.audit("push_media", path=str(path))
    BRIDGE.recorder.record("push_media", {"path": str(path)})
    return _act_and_report(result.detail)


@mcp.tool(
    structured_output=False,
    description=(
        "Open an app by name or bundle id, e.g. 'WhatsApp', 'Grab', 'com.apple.mobilesafari'. "
        "Always prefer this over hunting for an icon on the home screen."
    )
)
def phone_open_app(name: str) -> list[ContentBlock]:
    blocked = _unlocked_or_message() or _turn_or_message()
    if blocked:
        return blocked
    verdict = BRIDGE.guard.classify_app(name)
    if not verdict.allowed:
        return _text(f"REFUSED: {verdict.reason}")
    result = BRIDGE.phone.open_app(name)
    if not result.ok:
        if result.data.get("blocked"):
            # Do not let the model burn twenty turns inventing new ways to open
            # an app the device is refusing to foreground.
            return _text(
                f"STOP. {result.error}\n"
                "Do not try deep links, Spotlight, or tapping the icon: they fail the same way. "
                "Tell the user the phone is refusing to bring apps to the foreground and that the "
                "bridge needs restarting (`phoneshell up --relaunch`), or the phone rebooting."
            )
        installed = ", ".join(sorted(BRIDGE.phone.load_catalog().keys())[:40])
        return _text(f"could not open {name!r}: {result.error}\ninstalled apps include: {installed}")
    BRIDGE.guard.audit("open_app", app=name, bundle=result.data.get("bundleId"))
    BRIDGE.recorder.record("open_app", {"name": name})
    header = f"opened {name} ({result.data.get('bundleId')})"
    # Apps love to greet you with a promo sheet. Clearing it here costs one cheap
    # reflex; leaving it to the model costs a dozen turns of narration. Only
    # clearly promotional overlays are auto-closed, never a system dialog.
    try:
        snap = BRIDGE.phone.snapshot(with_screenshot=False)
        found = overlays.detect(snap.elements, snap.geometry.point_w,
                                snap.geometry.point_h, snap.alert)
        if found and found.promotional and found.kind != "system_alert":
            outcome = overlays.dismiss(BRIDGE.phone)
            if outcome.dismissed:
                header += f", and closed a promo overlay by {outcome.how}"
                BRIDGE.guard.audit("auto_dismiss", how=outcome.how)
    except Exception as exc:  # never let the reflex break the open
        log.debug("overlay reflex skipped: %s", exc)
    return _act_and_report(header)


@mcp.tool(
    structured_output=False,
    description=(
        "Open a URL or deep link on the phone, e.g. https://..., whatsapp://send?phone=60123456789, "
        "grab://open?screenType=GRABFOOD. This is the fastest way to reach a known screen."
    )
)
def phone_open_url(url: str) -> list[ContentBlock]:
    blocked = _unlocked_or_message() or _turn_or_message()
    if blocked:
        return blocked
    BRIDGE.guard.audit("open_url", url=url)
    result = BRIDGE.phone.open_url(url)
    if not result.ok:
        return _text(f"STOP. {result.error}")
    BRIDGE.recorder.record("open_url", {"url": url})
    return _act_and_report(f"opened {url}")


@mcp.tool(
    structured_output=False,
    description=(
        "Set a picker wheel (the spinning column iOS uses for times, dates, durations "
        "and units) to a value. Use this instead of swiping at a wheel: a wheel moves "
        "one row per tap beside the selected row, so swiping overshoots and never "
        "settles. wheel=0 is the leftmost column, 1 the next, and so on -- a timer is "
        "hours, minutes, seconds; a date picker is month, day, year. Call phone_observe "
        "first if you are unsure which column is which."
    )
)
def phone_set_picker(value: str, wheel: int = 0) -> list[ContentBlock]:
    blocked = _unlocked_or_message() or _turn_or_message()
    if blocked:
        return blocked
    BRIDGE.guard.audit("set_picker", value=value, wheel=wheel)
    anchor = _anchor_text()
    result = BRIDGE.phone.set_picker(value, wheel=wheel)
    BRIDGE.recorder.record("set_picker", {"value": value, "wheel": wheel}, anchor=anchor)
    return _act_and_report(result.detail or f"set wheel {wheel} to {value!r}")


@mcp.tool(structured_output=False, description="Press a hardware or system button: home, or back (which is a nav-bar tap or an edge swipe). Returns the screen as it is afterwards, with fresh ids, so do not call phone_observe next. ")
def phone_press(button: Literal["home", "back"]) -> list[ContentBlock]:
    if button == "home":
        BRIDGE.phone.home()
    else:
        BRIDGE.phone.back()
    return _act_and_report(f"pressed {button}")


@mcp.tool(
    structured_output=False,
    description=(
        "Answer a system alert (permissions, confirmations). Use the exact button label when "
        "you know it. Alerts are modal: nothing else works until one is dealt with."
    )
)
def phone_alert(action: Literal["accept", "dismiss"], button: str | None = None) -> list[ContentBlock]:
    text = BRIDGE.phone.wda.alert_text()
    if not text:
        return _text("there is no alert on screen right now")
    BRIDGE.guard.audit("alert", action=action, button=button, text=text)
    if action == "accept":
        BRIDGE.phone.accept_alert(button)
    else:
        BRIDGE.phone.dismiss_alert(button)
    return _act_and_report(f"{action}ed the alert {text!r}")


@mcp.tool(description="List the apps installed on the phone, as name -> bundle id.")
def phone_list_apps(filter: str = "") -> str:
    catalog = BRIDGE.phone.load_catalog()
    rows = sorted(
        (n, b) for n, b in catalog.items()
        if not filter or filter.lower() in n.lower() or filter.lower() in b.lower()
    )
    return "\n".join(f"{n}\t{b}" for n, b in rows) or "no matching apps"


@mcp.tool(structured_output=False, description="Wait for some text to appear on screen, for slow loads. Returns as soon as it shows up.")
def phone_wait_for(text: str, timeout: float = 15.0) -> list[ContentBlock]:
    result = BRIDGE.phone.wait_for_text(text, timeout=timeout)
    return _act_and_report(result.detail if result.ok else f"TIMEOUT: {result.error}")


GESTURES = {
    # name: (method, what it is for)
    "double_tap": "zoom in or out on a map or photo, like a post",
    "triple_tap": "select a whole paragraph of text",
    "long_press": "context menu, preview, start a drag, select a word",
    "force_touch": "deep press for a preview or a hidden menu",
    "two_finger_tap": "zoom to fit on a map",
    "three_finger_tap": "accessibility action",
    "pinch_in": "zoom out",
    "pinch_out": "zoom in",
    "rotate": "rotate a map or a photo",
    "two_finger_scroll": "scroll where one finger would draw or pan",
    "three_finger_swipe_left": "undo (iOS text editing)",
    "three_finger_swipe_right": "redo",
    "copy_selection": "three-finger pinch in, copies the selection",
    "paste_selection": "three-finger pinch out, pastes",
    "pull_to_refresh": "reload a feed",
    "flick": "fast scroll with momentum, throws the content",
    "page_left": "next page of a carousel, home screen or story",
    "page_right": "previous page",
    "control_centre": "open Control Centre (swipe from the top-right corner)",
    "notification_centre": "open Notification Centre (swipe from the top-left corner)",
    "app_switcher": "open the app switcher (swipe up from the bottom and hold)",
    "spotlight": "open search",
    "swipe_row_left": "reveal a row's actions: delete, archive, reply",
    "swipe_row_right": "the other row action: mark as read, complete",
    "drag_to_reorder": "pick a row up and move it",
    "cursor_drag": "use the keyboard as a trackpad to move the text cursor",
    "screenshot": "take a screenshot on the phone itself",
    "volume_up": "hardware volume up",
    "volume_down": "hardware volume down",
    "lock": "lock the screen",
    "wake": "wake the screen",
}


@mcp.tool(
    structured_output=False,
    description=(
        "Do several steps in one call, without stopping to look between them. This is the fastest "
        "way to work and you should prefer it whenever you already know the route. "
        "steps is a list like: "
        '[{\"open_app\": \"Settings\"}, {\"tap\": \"General\"}, {\"tap\": \"About\"}]. '
        "Verbs: open_app, open_url, tap (matches visible text, scrolling to find it), "
        "type (optionally {\"type\": \"hello\", \"into\": \"Search\", \"submit\": true}), "
        "set_picker ({\"set_picker\": \"5\", \"wheel\": 1} turns a spinning wheel, wheel 0 is "
        "leftmost), gesture (any name from phone_gesture), "
        "swipe (up/down/left/right), scroll_to, press (home/back), wait (seconds). "
        "It stops at the first step that fails and hands you the screen at that point, so a wrong "
        "guess costs one call rather than derailing you. One call replaces three or four "
        "observe-then-tap round trips."
    ),
)
def phone_do(steps: list, stop_on_failure: bool = True) -> list[ContentBlock]:
    blocked = _unlocked_or_message() or _turn_or_message()
    if blocked:
        return blocked
    from .perception.tree import find_by_text
    phone = BRIDGE.phone
    log_lines: list[str] = []

    def one(step: dict) -> tuple[bool, str]:
        if "open_app" in step:
            r = phone.open_app(str(step["open_app"]))
            return r.ok, f"open_app {step['open_app']!r}" + ("" if r.ok else f" FAILED: {r.error[:110]}")
        if "open_url" in step:
            r = phone.open_url(str(step["open_url"]))
            return r.ok, f"open_url {step['open_url']!r}" + ("" if r.ok else f" FAILED: {r.error[:110]}")
        if "tap" in step:
            target = str(step["tap"])
            snap = phone.snapshot(with_screenshot=False, stable=False)
            hits = find_by_text(snap.elements, target)
            if not hits:
                found = phone.scroll_to_text(target, max_swipes=6)
                if not found.ok:
                    return False, f"tap {target!r} FAILED: not on this screen"
                snap = phone.snapshot(with_screenshot=False, stable=False)
                hits = find_by_text(snap.elements, target)
                if not hits:
                    return False, f"tap {target!r} FAILED: found then lost it"
            element = hits[0]
            verdict = BRIDGE.guard.classify_tap(element)
            if verdict.needs_confirmation:
                return False, (f"tap {target!r} STOPPED: {verdict.reason}. Use phone_tap with "
                               "confirmed=true after telling the user what it will do.")
            BRIDGE.guard.audit("do_tap", element=element.describe())
            phone.tap_element(element)
            return True, f"tapped {target!r}"
        if "type" in step:
            into = step.get("into")
            element = None
            if into:
                snap = phone.snapshot(with_screenshot=False, stable=False)
                hits = find_by_text(snap.elements, str(into), clickable_only=False)
                element = hits[0] if hits else None
                if element is None:
                    return False, f"type FAILED: no field matching {into!r}"
            BRIDGE.guard.audit("do_type", text=str(step["type"])[:120])
            typed = phone.type_text(str(step["type"]), into=element, submit=bool(step.get("submit")))
            if not typed.ok:
                # type_text reads the screen back. A step that typed into nothing
                # must stop the sequence, or every step after it acts on a field
                # that is still empty.
                return False, f"type FAILED: {typed.detail}"
            return True, f"typed {str(step['type'])[:40]!r}" + (f" into {into!r}" if into else "")
        if "set_picker" in step:
            r = phone.set_picker(str(step["set_picker"]), wheel=int(step.get("wheel", 0)))
            return r.ok, ("set_picker " + r.detail if r.ok
                          else f"set_picker FAILED: {r.detail}")
        if "gesture" in step:
            name = str(step["gesture"])
            if name not in GESTURES:
                return False, f"gesture {name!r} FAILED: not a known gesture"
            # phone_gesture owns the dispatch for all 49 of them; reuse it rather
            # than keeping a second copy of that table in step form.
            phone_gesture(name, id=step.get("id"), amount=step.get("amount"))
            return True, f"gesture {name!r}"
        if "swipe" in step:
            phone.swipe(str(step["swipe"]))
            return True, f"swiped {step['swipe']}"
        if "scroll_to" in step:
            r = phone.scroll_to_text(str(step["scroll_to"]))
            return r.ok, f"scroll_to {step['scroll_to']!r}" + ("" if r.ok else " FAILED: not found")
        if "press" in step:
            phone.home() if step["press"] == "home" else phone.back()
            return True, f"pressed {step['press']}"
        if "wait" in step:
            time.sleep(min(float(step["wait"]), 10))
            return True, f"waited {step['wait']}s"
        return False, f"unknown step {step!r}"

    for i, step in enumerate(steps, 1):
        if not isinstance(step, dict):
            log_lines.append(f"{i}. skipped, not an object: {step!r}")
            continue
        try:
            ok, line = one(step)
        except Exception as exc:
            ok, line = False, f"{step!r} raised {type(exc).__name__}: {exc}"
        log_lines.append(f"{i}. {line}")
        if not ok and stop_on_failure:
            log_lines.append(f"stopped at step {i}; the screen below is where it stopped")
            break
    return _act_and_report("\n".join(log_lines))


@mcp.tool(
    structured_output=False,
    description=(
        "Perform a gesture beyond tap/type/swipe: " + ", ".join(GESTURES)
        + ". Pass id for element-targeted gestures (long_press, force_touch, swipe_row_*, "
        "drag_to_reorder), amount to tune it (zoom factor, radians, seconds, target y)."
    ),
)
def phone_gesture(name: str, id: int | None = None, amount: float | None = None) -> list[ContentBlock]:
    blocked = _unlocked_or_message() or _turn_or_message()
    if blocked:
        return blocked
    if name not in GESTURES:
        return _text(f"unknown gesture {name!r}. Known: {', '.join(sorted(GESTURES))}")
    g = BRIDGE.phone.gestures
    el = BRIDGE.element(id) if id is not None else None
    x, y = (el.cx, el.cy) if el else (
        BRIDGE.phone.wda.geometry().point_w / 2, BRIDGE.phone.wda.geometry().point_h / 2)
    BRIDGE.guard.audit("gesture", name=name, element=el.describe() if el else None, amount=amount)
    try:
        if name == "double_tap": g.double_tap(x, y)
        elif name == "triple_tap": g.triple_tap(x, y)
        elif name == "long_press": g.long_press(x, y, amount or 1.0)
        elif name == "force_touch": g.force_touch(x, y, amount or 1.0)
        elif name == "two_finger_tap": g.two_finger_tap(x, y)
        elif name == "three_finger_tap": g.three_finger_tap(x, y)
        elif name == "pinch_in": g.zoom_out(amount or 0.5)
        elif name == "pinch_out": g.zoom_in(amount or 2.0)
        elif name == "rotate": g.rotate(amount or 1.57)
        elif name == "two_finger_scroll": g.two_finger_scroll("down", amount or 0.5)
        elif name == "three_finger_swipe_left": g.undo()
        elif name == "three_finger_swipe_right": g.redo()
        elif name == "copy_selection": g.copy_selection(x, y)
        elif name == "paste_selection": g.paste_selection(x, y)
        elif name == "pull_to_refresh": g.pull_to_refresh()
        elif name == "flick": g.flick("down", amount or 0.7)
        elif name == "page_left": g.page("left")
        elif name == "page_right": g.page("right")
        elif name == "control_centre": g.control_centre() if hasattr(g, "control_centre") else g.control_center()
        elif name == "notification_centre": g.notification_center()
        elif name == "app_switcher": g.app_switcher()
        elif name == "spotlight": g.scroll("up", 0.35)
        elif name == "swipe_row_left":
            if el is None: return _text("swipe_row_left needs an id: which row?")
            g.swipe_row(el, "left")
        elif name == "swipe_row_right":
            if el is None: return _text("swipe_row_right needs an id: which row?")
            g.swipe_row(el, "right")
        elif name == "drag_to_reorder":
            if el is None: return _text("drag_to_reorder needs an id and an amount (target y)")
            g.drag_to_reorder(el, amount or el.cy)
        elif name == "cursor_drag": g.cursor_drag(x, y, x + (amount or 80), y)
        elif name == "screenshot": g.screenshot_chord()
        elif name == "volume_up": g.press_hardware("volume_up")
        elif name == "volume_down": g.press_hardware("volume_down")
        elif name == "lock": g.lock_screen()
        elif name == "wake": g.wake()
    except Exception as exc:
        return _text(f"{name} failed: {exc}")
    return _act_and_report(f"{name}{f' on [{id}]' if id else ''}")


@mcp.tool(
    structured_output=False,
    description=(
        "The full popup cheatsheet: every kind of thing that blocks a phone screen, how to "
        "recognise it, and the ordered moves that close it. Read this when phone_dismiss_popup "
        "could not clear something and you have to do it yourself."
    ),
)
def phone_popup_help() -> list[ContentBlock]:
    return _text(playbook.as_text())


@mcp.tool(
    structured_output=False,
    description=(
        "What the agent remembers about apps it has driven before: how many screens of each app "
        "it knows, how often it has been there, and which controls led where. "
        "action='name' with a name labels the CURRENT screen so future visits say what it is. "
        "action='forget' with an app clears that app's memory (or everything, with no app)."
    ),
)
def phone_memory(action: str = "list", app: str = "", name: str = "") -> list[ContentBlock]:
    memory = BRIDGE.phone.memory
    if action == "forget":
        n = memory.forget(app or None)
        return _text(f"forgot {n} app memor{'y' if n == 1 else 'ies'}")
    if action == "name":
        if not name:
            return _text("give a name for this screen")
        obs = BRIDGE.last_obs or BRIDGE.observe()
        memory.name_screen(obs.bundle_id, obs.elements, name)
        return _text(f"this screen is now remembered as {name!r}")
    rows = memory.stats()
    if not rows:
        return _text("nothing remembered yet")
    lines = ["app\tscreens\tvisits\tcontrols\troutes"]
    lines += [f"{r['app']}\t{r['screens']}\t{r['visits']}\t{r['controls']}\t{r['routes']}" for r in rows]
    return _text("\n".join(lines))


@mcp.tool(
    structured_output=False,
    description=(
        "Close whatever popup, promo sheet, interstitial or onboarding card is covering the "
        "screen. It identifies the shape of the thing and follows the known way out, verifying "
        "after each attempt. Call it the moment something unrelated to your task is in the way. "
        "If it fails it tells you what works for that shape. It never answers a system permission "
        "dialog: those are the user's decision. phone_popup_help has the full cheatsheet."
    ),
)
def phone_dismiss_popup() -> list[ContentBlock]:
    result = overlays.dismiss(BRIDGE.phone)
    BRIDGE.guard.audit("dismiss_popup", dismissed=result.dismissed, how=result.how or result.note)
    if result.dismissed:
        return _act_and_report(f"closed the overlay by {result.how}")
    if result.overlay and result.overlay.kind == "system_alert":
        return _act_and_report(
            f"there is a SYSTEM DIALOG on screen: {result.overlay.text[:150]!r}. "
            "Use phone_alert to accept or dismiss it, and if it is a permission or a payment, "
            "ask the user first."
        )
    return _act_and_report(result.note or "nothing to dismiss")


@mcp.tool(
    structured_output=False,
    description=(
        "List the saved macros: routes through an app that have already been driven "
        "successfully once. Check this before working a task out from scratch."
    ),
)
def phone_macro_list() -> list[ContentBlock]:
    items = macro_lib.list_macros()
    if not items:
        return _text("no macros saved yet")
    rows = [
        f"{m.name}\t{m.description}\t{len(m.steps)} steps\t{m.runs} runs, {m.failures} failures"
        f"\t{m.inputs or ''}"
        for m in items
    ]
    return _text("name\tdescription\tsteps\thistory\tinputs\n" + "\n".join(rows))


@mcp.tool(
    structured_output=False,
    description=(
        "Replay a saved macro. Fast: it finds each button by its text and taps it. It stops the moment "
        "the screen stops matching and hands you back the current screen, so you can carry on from there. "
        "It ALSO stops before any step that spends money, sends or deletes something: show the person "
        "the screen, and only after they say yes call again with confirmed=true and from_step=<that step>. "
        "inputs overrides a macro's variable values, e.g. {\"dish\": \"Green curry\"}."
    ),
)
def phone_macro_run(name: str, confirmed: bool = False, from_step: int = 1,
                    inputs: dict[str, str] | None = None) -> list[ContentBlock]:
    macro = macro_lib.get(name)
    if macro is None:
        return _text(f"no macro called {name!r}. Use phone_macro_list to see what exists.")
    BRIDGE.guard.audit("macro_run", name=name, steps=len(macro.steps), from_step=from_step,
                       confirmed=confirmed, inputs=inputs or {})
    result = macro_lib.replay(macro, BRIDGE.phone, find_by_text, confirmed=confirmed,
                              start_at=from_step, inputs=inputs, guard=BRIDGE.guard)
    macro.runs += 1
    if not result.ok and not result.needs_confirmation:
        macro.failures += 1
    macro.save()
    header = (f"macro {name!r}: {result.message} "
              f"({result.completed}/{result.total} steps done)")
    if result.log:
        header += "\n" + "\n".join(result.log)
    return _act_and_report(header)


def _server(path: str, method: str = "GET", body: dict | None = None, timeout: float = 20.0) -> dict:
    """The watcher lives in the phoneshell web server (it must outlive this
    process); start that server if it is not running."""
    import subprocess
    import httpx
    base = f"http://{BRIDGE.cfg.server_host}:{BRIDGE.cfg.server_port}"
    try:
        httpx.get(f"{base}/api/watches", timeout=2)
    except httpx.HTTPError:
        from .config import ROOT
        subprocess.Popen([str(ROOT / ".venv" / "bin" / "python"), "-m", "phoneshell.server"], cwd=str(ROOT),
                         stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                         start_new_session=True)
        for _ in range(40):
            time.sleep(0.5)
            try:
                httpx.get(f"{base}/api/watches", timeout=2)
                break
            except httpx.HTTPError:
                pass
    r = httpx.request(method, f"{base}{path}", json=body, timeout=timeout)
    return r.json()


@mcp.tool(
    structured_output=False,
    description=(
        "Watch and learn, step 1: start watching the phone while the PERSON does a task by hand "
        "(e.g. order their usual food). Then tell them: do the task on the phone normally, and say when "
        "you're done. Do not touch the phone yourself while watching."
    ),
)
def phone_watch_start(label: str) -> list[ContentBlock]:
    res = _server("/api/watch/start", "POST", {"label": label})
    d = res.get("demo") or {}
    return _text(f"watching (demo {d.get('id')}). Ask the person to do '{label}' on the phone now and tell "
                 f"you when they're done. They can see it at http://127.0.0.1:{BRIDGE.cfg.server_port}/learn")


@mcp.tool(
    structured_output=False,
    description=(
        "Watch and learn, step 2: the person finished. Stops watching, learns the steps (30-90 s), "
        "checks each against the recorded screens, and saves them as a macro under `name` unless "
        "save=false. Show the person the step list; then phone_macro_run(name) repeats the task fast."
    ),
)
def phone_watch_finish(name: str = "", description: str = "", save: bool = True) -> list[ContentBlock]:
    res = _server("/api/watch/stop", "POST", {"learn": True})
    d = res.get("demo")
    if not d:
        return _text("nothing was being watched")
    did = d["id"]
    for _ in range(300):
        time.sleep(2)
        st = _server(f"/api/watch/{did}")
        if st["status"] not in ("learning", "watching"):
            break
    if st["status"] != "ready":
        return _text(f"learning did not finish: {st['status']} {st.get('reason', '')}")
    draft = st["draft"]
    lines = []
    for i, s in enumerate(draft["steps"], 1):
        flag = "" if s.get("verified") else "  [UNVERIFIED: " + "; ".join(s.get("checks", [])) + "]"
        conf = "  [asks first]" if s.get("confirm") else ""
        lines.append(f"  {i}. {s['action']} {s.get('target_text') or s.get('params')}{conf}{flag}")
    out = (f"learned {len(draft['steps'])} steps from {st['frames']} screens"
           f" (inputs: {draft.get('inputs') or 'none'}):\n" + "\n".join(lines))
    if draft.get("notes"):
        out += f"\nnotes: {draft['notes']}"
    if save:
        saved = _server(f"/api/watch/{did}/save", "POST",
                        {"name": name or draft.get("name"), "description": description or draft.get("description", "")})
        out += (f"\nsaved as macro {saved.get('name')!r}. Run it with phone_macro_run." if saved.get("ok")
                else f"\nnot saved: {saved.get('error')}")
    return _text(out)


@mcp.tool(
    structured_output=False,
    description=(
        "Save what you just did as a reusable macro, once the task has actually worked. "
        "last_n limits it to the last N actions, so you can drop the exploring and keep "
        "only the route that worked."
    ),
)
def phone_macro_save(name: str, description: str, last_n: int | None = None) -> list[ContentBlock]:
    if not BRIDGE.recorder.trace:
        return _text("nothing to save: no actions have been recorded in this session")
    bundle = BRIDGE.last_obs.bundle_id if BRIDGE.last_obs else ""
    macro = BRIDGE.recorder.to_macro(name, description, bundle, last_n=last_n, created=time.time())
    path = macro.save()
    BRIDGE.guard.audit("macro_save", name=name, steps=len(macro.steps))
    listing = "\n".join(
        f"  {i}. {s.action} {s.target_text or s.params}" for i, s in enumerate(macro.steps, 1)
    )
    return _text(f"saved {len(macro.steps)} steps to {path}\n{listing}")


def main() -> None:
    # stdout belongs to the protocol; keep every library quiet on it.
    logging.basicConfig(level=logging.WARNING)
    for noisy in ("httpx", "phoneshell.wda", "phoneshell.actions", "pymobiledevice3"):
        logging.getLogger(noisy).setLevel(logging.ERROR)
    mcp.run()


if __name__ == "__main__":
    main()
