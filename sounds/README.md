Drop deterrent clips here (.wav/.flac/.ogg/.mp3). One is chosen at random per fire
(anti-habituation). Point elsewhere with DOGGY_CLIPS_DIR.

The shipped default is `alert-chirp.wav` — two rising 1.4->3.7 kHz sine sweeps,
regenerate/tweak with `python3 scripts/gen-beeps.py` (see `--variants` for
alternatives: softer/square chirps, triple beep, alarm warble).

On Linux the CommandAlerter plays via `pw-play`/`paplay`, which need WAV/FLAC —
not mp3. To use a different sound (e.g. a recording of your e-collar's tone),
drop the WAV here; to make it the ONLY alert, remove the others from this dir.
