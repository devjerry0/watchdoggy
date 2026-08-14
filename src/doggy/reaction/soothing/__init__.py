"""Soothing playback, split by concern: `player` (loop + hub Reaction),
`session` (playing one track), `state` (the cross-thread monitor), and
`audio` (platform backend + library listing)."""
from doggy.reaction.soothing.player import SoothingPlayer

__all__ = ["SoothingPlayer"]
