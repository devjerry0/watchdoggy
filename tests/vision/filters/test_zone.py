import pytest

from doggy.core.config import TunableSettings
from doggy.vision.analysis import FrameAnalysis
from doggy.vision.detection import Detection
from doggy.vision.filters.zone import ZoneInclusionFilter

# A triangle covering the top-left area of a 100x100 frame.
TRI = [(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)]
SHAPE = (100, 100, 3)

# The left half of a 100x100 frame -> mask columns 0..49 are inside. A box's
# overlap fraction is then just how much of its width falls left of x=50.
RECT_ZONE = [(0.0, 0.0), (0.5, 0.0), (0.5, 1.0), (0.0, 1.0)]

def test_box_inside_zone_is_kept():
    z = ZoneInclusionFilter()
    assert z.in_zone((5, 5, 15, 15), TRI, SHAPE) is True

def test_box_outside_zone_is_dropped():
    z = ZoneInclusionFilter()
    assert z.in_zone((80, 80, 95, 95), TRI, SHAPE) is False

def test_box_straddling_boundary_overlaps():
    z = ZoneInclusionFilter()
    # box spans the diagonal edge -> partial overlap -> True
    assert z.in_zone((25, 25, 45, 45), TRI, SHAPE) is True

def test_filter_keeps_only_in_zone():
    z = ZoneInclusionFilter()
    inside = Detection("dog", 0.9, (5, 5, 15, 15))
    outside = Detection("dog", 0.9, (80, 80, 95, 95))
    assert z.filter([inside, outside], TRI, SHAPE) == [inside]

def test_fewer_than_three_points_passes_through():
    z = ZoneInclusionFilter()
    d = Detection("dog", 0.9, (80, 80, 95, 95))
    assert z.filter([d], [(0.1, 0.1)], SHAPE) == [d]

def test_apply_narrows_inventory_to_zone():
    # The zone defines "the counter": out-of-zone inventory must not be reported.
    z = ZoneInclusionFilter()
    inside = Detection("cup", 0.5, (5, 5, 15, 15))
    outside = Detection("cup", 0.5, (80, 80, 95, 95))
    analysis = FrameAnalysis(SHAPE, [], [], [], inventory=[inside, outside])
    z.apply(analysis, TunableSettings(zone_enabled=True, zone_points=TRI))
    assert analysis.inventory == [inside]

def test_mask_rebuilds_on_shape_change():
    z = ZoneInclusionFilter()
    assert z.in_zone((5, 5, 15, 15), TRI, (100, 100, 3)) is True
    # different shape must not reuse the old mask (would index out of range)
    assert z.in_zone((5, 5, 15, 15), TRI, (50, 50, 3)) is True


def test_overlap_fraction_half_in():
    # A box straddling x=50 so exactly half its area lies inside the left-half
    # zone. (20,20,80,80) is centred on the boundary -> fraction ~= 0.5.
    z = ZoneInclusionFilter()
    assert z.overlap_fraction((20, 20, 80, 80), RECT_ZONE, SHAPE) == pytest.approx(
        0.5, abs=0.02)


def test_overlap_fraction_degenerate_box_is_zero():
    # A zero-area box cannot overlap anything -> 0.0, not a divide-by-zero.
    z = ZoneInclusionFilter()
    assert z.overlap_fraction((50, 50, 50, 60), RECT_ZONE, SHAPE) == 0.0


def test_candidate_scraping_edge_excluded_at_default_threshold():
    # Box (35,20,85,80): only its leftmost 15 of 50 columns are inside -> 30%.
    # At the default 0.4 threshold a box that merely scrapes the edge is dropped.
    z = ZoneInclusionFilter()
    scraper = Detection("dog", 0.9, (35, 20, 85, 80))
    analysis = FrameAnalysis(SHAPE, [], [scraper], [scraper])
    z.apply(analysis, TunableSettings(
        zone_enabled=True, zone_points=RECT_ZONE, zone_overlap=0.4))
    assert analysis.candidates == []


def test_candidate_scraping_edge_kept_at_low_threshold():
    # The same 30%-inside box passes once the threshold drops below 0.3.
    z = ZoneInclusionFilter()
    scraper = Detection("dog", 0.9, (35, 20, 85, 80))
    analysis = FrameAnalysis(SHAPE, [], [scraper], [scraper])
    z.apply(analysis, TunableSettings(
        zone_enabled=True, zone_points=RECT_ZONE, zone_overlap=0.2))
    assert analysis.candidates == [scraper]


def test_inventory_keeps_any_overlap_under_high_threshold():
    # Inventory never alerts, so an item half-off the counter should still be
    # counted: it ignores zone_overlap while candidates honour it.
    z = ZoneInclusionFilter()
    scraper = Detection("cup", 0.5, (35, 20, 85, 80))  # 30% inside
    analysis = FrameAnalysis(
        SHAPE, [], [scraper], [scraper], inventory=[scraper])
    z.apply(analysis, TunableSettings(
        zone_enabled=True, zone_points=RECT_ZONE, zone_overlap=0.9))
    assert analysis.candidates == []          # dropped by the high threshold
    assert analysis.inventory == [scraper]    # kept by any-overlap


def test_zone_overlap_zero_preserves_any_pixel_semantics():
    # A box whose overlap is a sliver (~2%) still passes when the threshold is 0,
    # matching the original any-pixel behaviour; a real threshold would drop it.
    z = ZoneInclusionFilter()
    sliver = (49, 20, 99, 80)  # only column 49 is inside -> ~0.02
    assert z.in_zone(sliver, RECT_ZONE, SHAPE, 0.0) is True
    assert z.in_zone(sliver, RECT_ZONE, SHAPE, 0.4) is False
