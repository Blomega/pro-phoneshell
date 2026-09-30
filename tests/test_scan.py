"""Scroll-scan stitching: screens read one at a time become one document."""
from phoneshell.scan import Scan, merge_frame, same_line


def frame(lines, edges=True):
    last = len(lines) - 1
    return [(t, 1.0, 0.1 + i * 0.05, edges and i in (0, last)) for i, t in enumerate(lines)]


def texts(scan):
    return [l.text for l in scan.content()]


def fresh():
    return Scan(id="t", label="t", started=0)


FEED = [x for i in range(12) for x in (f"@user_{i}", f"Loaf {i} costs {i * 3 + 5} dollars", "Reply")]


def test_scrolling_down_keeps_every_repeated_line_in_order():
    scan = fresh()
    for start in range(0, len(FEED) - 8, 5):          # 9-line screens, 4 lines of overlap
        merge_frame(scan, frame(FEED[start:start + 9]), start)
    merge_frame(scan, frame(FEED[-9:]), 99)
    assert texts(scan) == FEED


def test_scrolling_back_up_inserts_above():
    scan = fresh()
    merge_frame(scan, frame(FEED[15:24]), 1)
    merge_frame(scan, frame(FEED[9:18]), 2)
    merge_frame(scan, frame(FEED[3:12]), 3)
    assert texts(scan) == FEED[3:24]


def test_templated_lines_that_differ_only_in_numbers_are_different():
    assert not same_line("loaf 12 costs 41 dollars", "loaf 3 costs 14 dollars")
    assert same_line("sourdough loaf 12 costs 41 dollars", "sourdough loaf 12 costs 41 dollrs")


def test_a_screen_with_no_overlap_is_flagged_as_a_gap():
    scan = fresh()
    merge_frame(scan, frame(FEED[0:9]), 1)
    merge_frame(scan, frame(FEED[20:29]), 2)
    assert texts(scan) == FEED[0:9] + FEED[20:29]
    assert [l.gap for l in scan.content()].count(True) == 1


def test_a_fragment_under_the_header_is_not_kept():
    scan = fresh()
    merge_frame(scan, frame(FEED[0:9]), 1)
    # Next screen: its top line is a half-hidden "Reply" read as junk, landing
    # between two lines the first screen read in full.
    merge_frame(scan, frame(["rrupiy"] + FEED[3:12]), 2)
    assert "rrupiy" not in texts(scan)
    assert texts(scan) == FEED[0:12]
