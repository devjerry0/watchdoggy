from doggy.vision.detection import Detection
from doggy.vision.filters.zone import ZoneInclusionFilter

# A triangle covering the top-left area of a 100x100 frame.
TRI = [(0.0, 0.0), (0.6, 0.0), (0.0, 0.6)]
SHAPE = (100, 100, 3)

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

def test_mask_rebuilds_on_shape_change():
    z = ZoneInclusionFilter()
    assert z.in_zone((5, 5, 15, 15), TRI, (100, 100, 3)) is True
    # different shape must not reuse the old mask (would index out of range)
    assert z.in_zone((5, 5, 15, 15), TRI, (50, 50, 3)) is True
