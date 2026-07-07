from doggy.vision.detection import Detection
from doggy.vision.filters.person import iou, suppress_dogs_overlapping_people


def dog(box, c=0.9):
    return Detection("dog", c, box)


def person(box, c=0.9):
    return Detection("person", c, box)


def test_iou_identical_is_one():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0


def test_iou_disjoint_is_zero():
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_partial_overlap():
    # a & b each area 100, intersection (5,0,10,10)=50, union=150
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - (50 / 150)) < 1e-9


def test_coincident_dog_and_person_is_suppressed():
    # "dog" box ~= person box (one human, double-labeled) -> removed
    dogs = [dog((0, 0, 100, 200))]
    people = [person((2, 2, 98, 198))]
    assert suppress_dogs_overlapping_people(dogs, people, 0.85) == []


def test_real_dog_behind_person_is_kept():
    # dog has its own small distinct box that only clips the person -> low IoU -> kept
    dogs = [dog((150, 150, 190, 190))]
    people = [person((0, 0, 100, 200))]
    assert suppress_dogs_overlapping_people(dogs, people, 0.85) == dogs


def test_no_people_keeps_all_dogs():
    dogs = [dog((0, 0, 10, 10))]
    assert suppress_dogs_overlapping_people(dogs, [], 0.85) == dogs


def test_only_the_coincident_dog_is_removed():
    coincident = dog((0, 0, 100, 200))
    real = dog((300, 300, 340, 340))
    people = [person((0, 0, 100, 200))]
    assert suppress_dogs_overlapping_people([coincident, real], people, 0.85) == [real]
