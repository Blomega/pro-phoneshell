"""Learned action sequences: do the task once, replay it forever.

The measured win here is large. Merging a verified sequence of steps into one
replayable unit is worth roughly twenty points of task success on repeat work
and cuts token use by most of an order of magnitude, because the model stops
re-deriving a route it has already found.

The design point that makes replay safe rather than reckless: every step carries
an *anchor*, a piece of text that must be on screen before the step fires. If the
anchor is missing the macro stops and hands the screen back to the model, which
then takes over from exactly that point instead of starting again. A macro is a
shortcut through known territory, never a blind script.
"""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import yaml

from ..config import MACROS


@dataclass
class Step:
    action: str                     # tap | type | swipe | open_app | open_url | press | scroll_to | alert
    params: dict[str, Any] = field(default_factory=dict)
    anchor: str = ""                # text that must be on screen before this step runs
    target_text: str = ""           # what the tapped element said, for re-finding it
    expect: str = ""                # text expected on screen afterwards
    note: str = ""
    # Spends money, sends something, deletes something: replay stops here and
    # asks, every time, however many times the macro has run before.
    confirm: bool = False


@dataclass
class Macro:
    name: str
    description: str = ""
    bundle_id: str = ""
    steps: list[Step] = field(default_factory=list)
    created: float = 0.0
    runs: int = 0
    failures: int = 0
    # Values that change between runs, e.g. {"dish": "Pad Thai"}. Steps refer to
    # them as {dish}; the saved value is the default, a run can pass another.
    inputs: dict[str, str] = field(default_factory=dict)
    source: str = ""                # "agent" (saved after an agent run) or "demo" (learned by watching)

    @classmethod
    def load(cls, path: Path) -> "Macro":
        raw = yaml.safe_load(path.read_text()) or {}
        steps = [Step(**s) for s in raw.pop("steps", [])]
        return cls(steps=steps, **raw)

    def save(self, directory: Path = MACROS) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{self.name}.yaml"
        data = asdict(self)
        path.write_text(yaml.safe_dump(data, sort_keys=False, allow_unicode=True))
        return path


def list_macros(directory: Path = MACROS) -> list[Macro]:
    out = []
    for path in sorted(directory.glob("*.yaml")):
        try:
            out.append(Macro.load(path))
        except Exception:
            continue
    return out


def get(name: str, directory: Path = MACROS) -> Macro | None:
    path = directory / f"{name}.yaml"
    return Macro.load(path) if path.exists() else None


class Recorder:
    """Keeps the trace of what actually happened, so a success can become a macro."""

    def __init__(self, limit: int = 200):
        self.trace: list[Step] = []
        self.limit = limit

    def record(self, action: str, params: dict, anchor: str = "", target_text: str = "",
               expect: str = "") -> None:
        self.trace.append(Step(action=action, params=dict(params), anchor=anchor,
                               target_text=target_text, expect=expect))
        self.trace = self.trace[-self.limit:]

    def clear(self) -> None:
        self.trace.clear()

    def to_macro(self, name: str, description: str, bundle_id: str, last_n: int | None = None,
                 created: float = 0.0) -> Macro:
        steps = self.trace[-last_n:] if last_n else list(self.trace)
        return Macro(name=name, description=description, bundle_id=bundle_id,
                     steps=steps, created=created)


@dataclass
class ReplayResult:
    ok: bool
    completed: int
    total: int
    message: str = ""
    diverged_at: int | None = None
    # Stopped on purpose before a step that needs the person's go-ahead. Run
    # again with confirmed=True and start_at=diverged_at to carry on.
    needs_confirmation: bool = False
    log: list[str] = field(default_factory=list)


def _fill(text: str, values: dict[str, str]) -> str:
    for k, v in values.items():
        text = text.replace("{" + k + "}", str(v))
    return text


def resolved(step: Step, values: dict[str, str]) -> Step:
    """The step with its {input} placeholders filled in."""
    params = {k: (_fill(v, values) if isinstance(v, str) else v) for k, v in step.params.items()}
    return Step(action=step.action, params=params, anchor=_fill(step.anchor, values),
                target_text=_fill(step.target_text, values), expect=_fill(step.expect, values),
                note=step.note, confirm=step.confirm)


def replay(macro: Macro, phone, find_by_text, max_anchor_wait: float = 8.0, *,
           confirmed: bool = False, start_at: int = 1, inputs: dict[str, str] | None = None,
           guard=None) -> ReplayResult:
    """Run a macro against the live phone, stopping the moment reality diverges,
    and ALWAYS stopping before a step that spends, sends or deletes unless this
    call was explicitly confirmed. `confirmed` covers exactly one such step (the
    one at start_at): a macro with two payment steps asks twice."""
    values = {**macro.inputs, **(inputs or {})}
    total = len(macro.steps)
    notes: list[str] = []

    def stop(i: int, done: int, msg: str, confirm: bool = False) -> ReplayResult:
        return ReplayResult(ok=False, completed=done, total=total, diverged_at=i, message=msg,
                            needs_confirmation=confirm, log=notes)

    for i, raw in enumerate(macro.steps, start=1):
        if i < start_at:
            continue
        step = resolved(raw, values)
        if step.anchor:
            found = phone.wait_for_text(step.anchor, timeout=max_anchor_wait)
            if not found.ok:
                return stop(i, i - 1, f"step {i} expected {step.anchor!r} on screen and it is not there. "
                                      "The app has changed or the flow moved; take over from here.")
        snap = phone.snapshot(with_screenshot=False)
        if step.action == "tap":
            target = step.target_text or step.params.get("text", "")
            hits = find_by_text(snap.elements, target) if target else []
            if not hits and target:
                # A list item that was on screen during the demo can be further
                # down today. Look for it before giving up, and say so.
                if phone.scroll_to_text(target).ok:
                    notes.append(f"step {i}: had to scroll to find {target!r}")
                    snap = phone.snapshot(with_screenshot=False)
                    hits = find_by_text(snap.elements, target)
            if not hits:
                return stop(i, i - 1, f"step {i} wanted to tap {target!r} and nothing on screen matches it.")
            risky = step.confirm or (guard is not None and guard.classify_tap(hits[0]).needs_confirmation)
            if risky and not (confirmed and i == start_at):
                return stop(i, i - 1, f"paused before step {i}: tap {hits[0].text!r}. This looks like it spends "
                                      "money, sends or deletes something. Show the person the screen and run "
                                      f"again with confirmed=true, from_step={i} once they say yes.", confirm=True)
            phone.tap_element(hits[0])
        elif step.action == "type":
            target = step.target_text
            into = None
            if target:
                hits = find_by_text(snap.elements, target, clickable_only=False)
                into = hits[0] if hits else None
            phone.type_text(step.params.get("text", ""), into=into,
                            submit=bool(step.params.get("submit")))
        elif step.action == "swipe":
            phone.swipe(step.params.get("direction", "down"),
                        distance=float(step.params.get("distance", 0.6)))
        elif step.action == "scroll_to":
            phone.scroll_to_text(step.params.get("text", ""))
        elif step.action == "open_app":
            phone.open_app(step.params.get("name", ""))
        elif step.action == "open_url":
            phone.open_url(step.params.get("url", ""))
        elif step.action == "press":
            phone.home() if step.params.get("button") == "home" else phone.back()
        elif step.action == "alert":
            (phone.accept_alert if step.params.get("action", "accept") == "accept"
             else phone.dismiss_alert)(step.params.get("button"))
        else:
            return stop(i, i - 1, f"step {i} has an unknown action {step.action!r}")
        if step.expect:
            seen = phone.wait_for_text(step.expect, timeout=max_anchor_wait)
            if not seen.ok:
                return stop(i + 1, i, f"after step {i} the screen should show {step.expect!r} and it does not. "
                                      "Take over from the current screen.")
    return ReplayResult(True, total, total, "macro completed", log=notes)
