from doggy.decision.gate import FireGate
from doggy.reaction.clips import ClipBuffer, ClipService
from doggy.reaction.hub import ReactionHub, SafeReaction
from doggy.reaction.outcome import OutcomeWatcher
from doggy.reaction.sound import FakeAlerter, SoundReaction
from doggy.vision.analysis import DetectionAnalyzer
from doggy.vision.filters.base import FilterChain
from doggy.vision.filters.person import PersonSuppressionFilter
from doggy.vision.filters.zone import ZoneInclusionFilter


def _analyzer(detector):
    return DetectionAnalyzer(
        detector, FilterChain([PersonSuppressionFilter(), ZoneInclusionFilter()]))


def _clips(store, settings, runtime):
    return ClipService(store, store.dir, ClipBuffer(settings.clip_window_seconds), runtime)


def _outcome(store, runtime):
    return OutcomeWatcher(store, FireGate(runtime), FakeAlerter(), runtime)


def _hub(alerter, clip_service, store, outcome):
    return ReactionHub(
        [SafeReaction(SoundReaction(alerter, store)), SafeReaction(clip_service),
         SafeReaction(outcome)])
