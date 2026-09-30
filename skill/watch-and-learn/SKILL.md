---
name: watch-and-learn
description: Teach Claude a phone task by doing it once on Sam's iPhone, then have Claude repeat it fast. Use when Sam says "watch me", "learn how I do this", "remember how I order X", "do my usual order", "repeat what I did", or asks Claude to do a phone task it has a saved route for. Invoke with /watch-and-learn.
---

# Watch and learn (phoneshell)

Code: `~/Desktop/projects/pro-phoneshell` (`phoneshell/watch.py`, `phoneshell/agent/macros.py`).
Page: http://127.0.0.1:8765/learn. MCP server `phoneshell` (user scope).

## Teach (Sam does it once)
1. Phone on the cable and unlocked. If `phone_status` says not connected: `scrollscan up` (or `bin/phoneshell up`).
2. `phone_watch_start(label="order Pad Thai on Grab")`, then tell Sam: do it on the phone normally, say when done.
   Do NOT touch the phone while watching.
3. When Sam says done: `phone_watch_finish(name="grab-pad-thai")`. It learns the steps (~40 s), checks each against
   the recorded screens, and saves the macro. Show him the step list, including any UNVERIFIED step and the inputs.
   He can also review, untick and rename on the /learn page.

## Repeat (Claude does it)
1. `phone_macro_list` to find it (inputs show what can change, e.g. dish, restaurant).
2. `phone_macro_run(name, inputs={...})`. Takes seconds.
3. It ALWAYS pauses before a step that pays, orders, sends or deletes. Show Sam the screen (`phone_observe`) and what
   is about to happen (total, item, recipient). Only after he says yes: `phone_macro_run(name, confirmed=true,
   from_step=<the step it paused at>)`. Never confirm on his behalf.
4. If it stops early ("expected X on screen"), the app changed: take over from the current screen with the normal
   phone tools, finish the task, and tell Sam the macro needs re-teaching.

## Rules
- The phone is Sam's real phone with real money. Nothing irreversible without his explicit yes in this conversation.
- Recordings (screens) stay local in `runtime/watch/`; they can contain private data.
