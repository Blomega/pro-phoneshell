"""Watch and learn: learned steps are checked against what was seen, and replay
never spends, sends or deletes without an explicit yes."""
from types import SimpleNamespace

from phoneshell.agent.macros import Macro, Step, replay
from phoneshell.perception.tree import Element, find_by_text
from phoneshell.watch import Demo, Keyframe, check_steps


def el(i, text, typ="Button", y=100):
    return Element(idx=i, type=typ, label=text, x=10, y=y, w=200, h=40)


class FakePhone:
    """Screens are lists of labels; tapping a label moves to the screen it leads to."""

    def __init__(self, screens, links, start, hidden=None):
        self.screens, self.links, self.at = screens, links, start
        self.hidden = hidden or {}        # label -> revealed only after scroll_to_text
        self.taps, self.typed, self.scrolled = [], [], []

    def _els(self):
        return [el(i, t, y=100 + 50 * i) for i, t in enumerate(self.screens[self.at])]

    def snapshot(self, with_screenshot=False):
        return SimpleNamespace(elements=self._els())

    def wait_for_text(self, needle, timeout=0):
        return SimpleNamespace(ok=bool(find_by_text(self._els(), needle, clickable_only=False)))

    def tap_element(self, e):
        self.taps.append(e.text)
        self.at = self.links.get((self.at, e.text), self.at)

    def type_text(self, text, into=None, submit=False):
        self.typed.append(text)
        self.at = self.links.get((self.at, "typed"), self.at)

    def scroll_to_text(self, text):
        self.scrolled.append(text)
        if text in self.hidden.get(self.at, []):
            self.screens[self.at] = self.screens[self.at] + [text]
            return SimpleNamespace(ok=True)
        return SimpleNamespace(ok=False)


def food_phone(hidden=False):
    screens = {"home": ["Search"], "results": ["Pad Thai", "Green Curry"] if not hidden else ["Green Curry"],
               "dish": ["Add to basket"], "basket": ["Place order"], "placed": ["Order placed"]}
    links = {("home", "typed"): "results", ("results", "Pad Thai"): "dish", ("results", "Green Curry"): "dish",
             ("dish", "Add to basket"): "basket", ("basket", "Place order"): "placed"}
    return FakePhone(screens, links, "home", hidden={"results": ["Pad Thai"]} if hidden else None)


MACRO = Macro(name="order", inputs={"dish": "Pad Thai"}, steps=[
    Step("type", {"text": "{dish}"}, target_text="Search"),
    Step("tap", target_text="{dish}"),
    Step("tap", target_text="Add to basket"),
    Step("tap", target_text="Place order", confirm=True, expect="Order placed"),
])


def test_replay_stops_before_paying_and_resumes_only_when_confirmed():
    phone = food_phone()
    res = replay(MACRO, phone, find_by_text)
    assert not res.ok and res.needs_confirmation and res.diverged_at == 4
    assert "Place order" not in phone.taps
    res = replay(MACRO, phone, find_by_text, confirmed=True, start_at=res.diverged_at)
    assert res.ok and phone.taps[-1] == "Place order"


def test_a_confirmation_covers_one_step_only():
    phone = food_phone()
    res = replay(MACRO, phone, find_by_text, confirmed=True)   # confirmed, but for step 1, not 4
    assert res.needs_confirmation and "Place order" not in phone.taps


def test_inputs_change_what_is_ordered():
    phone = food_phone()
    replay(MACRO, phone, find_by_text, inputs={"dish": "Green Curry"})
    assert phone.typed == ["Green Curry"] and "Green Curry" in phone.taps


def test_guard_pauses_a_risky_tap_even_when_the_step_was_not_marked():
    macro = Macro(name="m", steps=[Step("tap", target_text="Search")])
    guard = SimpleNamespace(classify_tap=lambda e: SimpleNamespace(needs_confirmation=True))
    res = replay(macro, food_phone(), find_by_text, guard=guard)
    assert res.needs_confirmation


def test_a_target_below_the_fold_is_scrolled_to_and_reported():
    phone = food_phone(hidden=True)
    res = replay(MACRO, phone, find_by_text)
    assert res.needs_confirmation            # got all the way to the payment step
    assert phone.scrolled == ["Pad Thai"] and any("scroll" in n for n in res.log)


def frame(n, labels):
    return Keyframe(n=n, t=0, app="Grab", bundle="com.grab", w=430, h=932,
                    elements=[{"id": i, "type": "Button", "text": t, "label": t, "x": 10, "y": 100 + 50 * i,
                               "w": 200, "h": 40} for i, t in enumerate(labels)])


def test_learned_steps_are_checked_against_the_screens_they_came_from():
    demo = Demo(id="d", label="order", started=0,
                frames=[frame(1, ["Pad Thai", "Menu"]), frame(2, ["Add to basket", "Pad Thai details"])])
    steps = check_steps(demo, [
        {"action": "tap", "target_text": "{dish}", "anchor": "Menu", "expect": "Add to basket",
         "from_frame": 1, "to_frame": 2},
        {"action": "tap", "target_text": "Checkout", "from_frame": 1, "to_frame": 2},
        {"action": "tap", "target_text": "Pad Thai", "anchor": "Today only 20% off", "from_frame": 1, "to_frame": 2},
    ], {"dish": "Pad Thai"})
    assert steps[0]["verified"] and not steps[0]["checks"]
    assert not steps[1]["verified"]                         # not on the screen it claims to come from
    assert steps[2]["verified"] and steps[2]["anchor"] == ""  # bad anchor dropped, step kept


def test_landmarks_that_change_with_time_are_dropped():
    demo = Demo(id="d", label="x", started=0, frames=[frame(1, ["Good afternoon, Sam", "Food"]), frame(2, ["Total 145"])])
    [s] = check_steps(demo, [{"action": "tap", "target_text": "Food", "anchor": "Good afternoon",
                              "expect": "Total 145", "from_frame": 1, "to_frame": 2}], {})
    assert s["anchor"] == "" and s["expect"] == "" and s["verified"]


def test_landmarks_holding_a_demo_value_follow_the_input():
    from phoneshell.watch import to_macro
    demo = Demo(id="d", label="x", started=0, frames=[frame(1, ["a"])])
    m = to_macro(demo, "n", "", [{"action": "type", "params": {"text": "{q}"}, "target_text": "Search",
                                  "expect": "Pad Thai Ban Khun Mae"}], {"restaurant": "Ban Khun Mae", "q": "pad thai"})
    assert m.steps[0].expect == "{restaurant}" and m.steps[0].target_text == "Search"
