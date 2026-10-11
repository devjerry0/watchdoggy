# watchdoggy — agent guide

Raspberry Pi appliance ("Counter Watch") that watches a kitchen counter with a
camera, detects the dog via YOLO/NCNN on-CPU, and plays a deterrent sound
through a Bluetooth/USB speaker. Web dashboard for live view, watch-area,
catch log, and settings. Runs fully offline (nftables egress firewall); only a
sandboxed `trainer` user may reach the internet, for Modal cloud fine-tuning.

This file is the shared brain for all coding agents (Claude Code reads it via
the CLAUDE.md symlink; Codex reads it natively). Capture durable, non-obvious
learnings here (e.g. via /ce-compound). Keep entries terse; cite code paths.

## Ground rules

- **Public repo.** Never commit camera frames, snapshots, or anything from
  `events/`, `dataset-pull/`, or `training-runs/`. Never `git add -A` or
  `git add .` — stage files explicitly by name.
- Python is uv-managed: run things with `uv run doggy`, `uv run pytest tests/`.
  On the Pi, never plain `uv sync` — it removes out-of-band packages and the
  Pi's firewall blocks re-fetching (see docs/ and skills for the deploy flow).
- Package name is `doggy` (src/doggy); entry point `doggy = doggy.main:main`.
- Config split: `.env` is STRUCTURAL only (camera, ports, paths, TLS; the 19
  `Settings`-only fields). The 51 live tunables (`TunableSettings`) live in
  `settings.json` via `core/settings_store.py` (atomic write, first-boot
  migration from .env, `settings-changes.jsonl` audit log with client IP).
  Never add a tunable that reads `.env` at runtime; out-of-process readers
  (trainer daemon) read `settings.json` directly. Both files are appliance
  state: gitignored, snapshotted by `doggy-install-code`, excluded by
  `deploy-to-pi.sh`.
- Docs voice (user rule): plain sentences that say what a thing does. No em
  dashes, no "not X but Y" / "X, not Y" contrast framing, no "actually",
  no punchy slogan bullets.
- Architecture overview: ARCHITECTURE.md. Systemd units: systemd/.
  Training pipeline: scripts/train_kitchen_model.py.

## Learnings

<!-- Append dated entries below. Terse: what was non-obvious, why, code path. -->

- **2026-09-06 — Modal Volumes have a hard 500,000-inode (file count) cap;
  ENOSPC on mkdir means files, not bytes.** The Sep 1-6 outage: timed-out
  training runs each left a built dataset (~100k files) in their run dir,
  prune only ran on success, the cap hit, and every run died at
  `run_dir.mkdir` while the daemon re-queued one per 30-min tick (~$13
  burned, silent). Diagnose with a cloud-side `du -s --inodes` / `df -i`
  probe, not `du -h`. Guards now in `scripts/kitchen_training/hygiene.py`
  (start-of-run janitor + dataset strip + `.completed` markers + a
  ceiling-fit epoch cap — epochs are NEVER scaled down just because data
  grew, user-decided; the full schedule always runs, see 2026-10-11) and
  `scripts/trainer_daemon/queue.py` (per-kind failure
  backoff; one >4h failure arms it alone). Cleanup must run at the START
  of a run — a full volume prevents the success that end-of-run cleanup
  depends on.
- **2026-09-06 — Any auto-retried cloud job needs backoff + a visible
  failure streak from day one.** 260 identical failures cost real money
  before anyone noticed; the /training budget card now banners at 3
  consecutive failures (`cloud_failure_streak` in
  `src/doggy/web/routers/training.py`, threshold mirrored from
  `trainer_daemon.queue.BACKOFF_AFTER`). Credit refusals carry status
  `refused` so "no credits" never reads as "cloud broken".
- **2026-09-06 — The detector's NCNN threads sync per layer, so ~1 core of
  co-load costs ~25%+ FPS (2.2 -> 1.0-1.3 during trainer passes).**
  `doggy-trainer.service` runs at Nice=10 CPUWeight=20 (drop-in on the Pi +
  `scripts/setup-pi-trainer.sh`). If FPS "mysteriously" sags, check what
  else is on the cores before touching the model.
- **2026-09-09 — FPS on the Pi is inference-bound (~2.4 fps at imgsz 640);
  everything else is about not stealing from NCNN and not stalling the
  loop.** Fixed: catch-clip WebP encode ran inline on the detect thread
  (72 frames = 10-15s freeze, dashboard read "0.1 FPS" during Cooling
  down) -> worker thread (`reaction/clips.py`); MJPEG stream re-encoded the
  same frame 10x/s per viewer -> encode on `FrameBuffer.version` change;
  webcam decoded 30 fps for a 2-fps consumer -> `CAP_PROP_FPS` 10
  (`vision/camera.py`); dashboard shows `fps_avg` (EMA) not instantaneous
  1/dt. Profile with `py-spy record --rate 25 --format raw` on the main
  process (root via trainer venv), but NOTE py-spy counts threads parked in
  `epoll`/`Popen.wait` as active -- read real CPU from `top`. "Cooling
  down" on the dashboard = post-catch COOLDOWN state, not thermal (thermal
  shows "easing off" by the temperature). Next lever if more FPS is wanted:
  export imgsz 640 -> 480 (~1.7x), gate-judged.
- **2026-09-09 — At ~47k frames (~95k dir entries) any O(dataset) directory
  pass on a hot path is an FPS bug.** `SidecarIndex.snapshot()` re-stat'ed
  every entry per request (1-3s Pi CPU) and the training page polls it ->
  ~1 FPS whenever that page was open; `DatasetCapture` ran the cap prune
  (stat of every sample_*) on EVERY saved frame -> a dip every few minutes
  at idle. Now the index is EVENT-DRIVEN: writers call
  `index.upsert(stem)` after an in-place sidecar edit (one stat+parse); a
  directory-mtime change (capture/prune) triggers a names-only readdir
  that stats/parses only new names; the full re-stat runs once a DAY (or
  `invalidate()`). Prune runs once a day (`PRUNE_INTERVAL_SECONDS`).
  Rule: anything in the web process that edits a sidecar in place MUST
  `upsert(stem)` -- an in-place write does not move the dir mtime, and the
  daily net is the only other catch. The cold build + daily re-stat run on
  `SidecarIndex.start()`'s daemon thread at nice 15 (never on a request);
  heavy aggregate loops call `core.priority.lower_thread_priority(10)`
  first (Linux per-thread nice, sticks for the worker's life -- fine, the
  dashboard must never outrun the detector). Snapshot rows are bisect-
  inserted, not re-sorted. Tests build apps with
  `create_app(..., index_refresh=False)` (synchronous index); the index
  binds `threading.Thread` at import because test_api_serve fakes it to run
  serve()'s uvicorn threads inline. `api_frames` still walks all sidecars
  per call (~10-15% while the label page fetches, now at nice 10).
  Live view: `MjpegFanout` encodes each frame once for all viewers and caps
  concurrent streams at 4 (503 + dashboard hint beyond that).
- **2026-09-12 — "The model isn't catching" was the THRESHOLD, not the
  model.** `DOGGY_CONFIDENCE` had been slid to 0.95 (persisted in .env);
  real dogs score 0.6-0.9 through the NCNN export, so 1,745 sightings in 3
  days produced zero fires while the dashboard looked armed. Diagnose with
  the exam curve (`training-runs/<run>/local-gate.json`, or the report):
  deployed model @0.8 82/127, @0.9 30/127, @0.95 4/127. Guard:
  `MAX_ALARM_CONFIDENCE = 0.85` in `core/tunables.py` (422 on PATCH,
  slider max mirrors it). The user's operating point is **0.6**. When a
  catch-rate complaint arrives, check `.env` + `events.jsonl` confidences
  BEFORE blaming the model.
- **2026-09-12 — A cloud run that dies after export can be finished on the
  Mac for free.** `kitchen_ncnn_model/` + `train/` survive on the volume
  (`modal volume get`); rebuild the exam split locally from
  `build._split` (val stems as empty marker files under
  `run_dir/dataset/images/val`), pull only the human-labeled images
  (`rsync --files-from` of the `.labeled-index.json` stems, ~1.3k files
  not 57k), run `deploy_gate` + `robustness` (inference only, minutes on
  CPU/MPS), then stage via `doggy` sudo into `/home/trainer/staging_ncnn_model`
  and `doggy-install-model`. Script: the local_gate.py pattern in this
  session; the labeled index lives in `jobs/`, not `dataset/`, on the Pi.
  Write the job's `.result.json` overlay by hand (status done + summary) so
  the training card and the failure-streak/backoff see the success.
  **NEVER stage a bundle with `tar` from a Mac**: it ships AppleDouble
  `._model.ncnn.param` twins, NCNN's loader globs `*.param`, picks the
  junk first ("parse magic failed" -> `IndexError: list index out of
  range` in `inference_worker.init`), and every inference child dies at
  load -- the detector ran blind for 25 min on 2026-09-12 at "0.1 FPS".
  Use `rsync -a` / `scp -r` (they don't create `._` files) and, when the
  detector shows 0.1 FPS after a restart with IndexError in the journal,
  check `ls -a models/kitchen_ncnn_model` first. `doggy-install-model` now
  strips `._*` and refuses a bundle without exactly one .param + one .bin.
- **2026-09-12 — Contracts the Codex cross-review pinned (keep them):**
  (a) `load_settings()` must never let a LEGACY .env tunable stop boot --
  a model-level pydantic error has an empty `loc`, so classify by "does it
  validate with all tunables defaulted?", not by field name. (b) Anything
  in `app.py` that reads a tunable at construction (retention, clip
  window, log level) must read the merged settings.json value:
  `settings.model_copy(update=tunables.model_dump())`. (c) `SidecarIndex`
  hands out its snapshot list and readers iterate it outside the lock, so
  incremental updates are COPY-ON-WRITE; the refresher thread must catch
  everything (its first exception handler had a NameError -- the test
  `test_refresher_survives_a_scan_exception` guards it). (d) Nightly jobs
  run TWO cloud phases; Pi-side wait wrappers = 2 x CLOUD_JOB_CEILING.
  Declined findings: kickoff-dir pruning vs in-flight jobs (daemon
  serializes; current run exempt), backoff edge cases (bounded by the 24h
  cap). Codex auth gotcha: `codex login status` can say "Logged in" with
  a revoked refresh token; a real `codex login` (browser callback) is the
  only fix -- verify with a cheap `codex exec` before relying on it.
- **2026-10-11 — Cloud training is GPU-step-bound, not dataloader-bound;
  "auto" GPU is L40S.** Synthetic-frame Modal benchmark at the pipeline's
  settings (yolo26n freeze=10, batch 32, imgsz 640, RAM cache, cpu=8): L4
  180 img/s, L40S 416-435, A100-40GB 184 (no faster than L4), H100
  445-478; 16 cores/workers changed nothing; GPU util 30-60% everywhere =
  per-step overhead, so throughput scales with batch. L40S is ~2.3x L4
  AND cheaper per trained image ($1.60 vs $2.02 per million incl.
  cpu+memory). The ceiling-fit cap uses per-GPU rates
  (`hygiene.TRAIN_IMAGES_PER_SECOND`, scaled down by batch/32, unmeasured
  GPUs keep 60); the old flat 60 trimmed the Sep 7 run to 32 of 80 epochs
  while it was still improving. The GPU reaches the container in
  `recipe["gpu"]`: `DOGGY_TRAIN_GPU` exists only on the launching machine.
  Benchmark a GPU before adding its rate. Worst-case hung run on L40S is
  ~$25 (10h ceiling). Modal now requires a payment method on file for ANY
  GPU, T4 included ("Please add a payment method to use <GPU> GPU
  functions", policy change between Sep 7 and Oct 11). A Pi without
  internet can't refresh `jobs/billing.json`, so a stale "credits spent"
  file silently keeps the credit gate closed (Sep 13 - Oct 11: zero jobs).
- **2026-10-11 — Schedule + labeling GPU (user-decided).** Label weekly
  (`label_interval_hours` 168, at `nightly_prelabel_hour`); train every
  `train_interval_hours` 336 but ONLY on the pass right after a labeling
  pass (`FRESH_LABELS_SECONDS`), so every run trains on fresh auto-labels
  (~twice a month). Escapes so it can't deadlock: nothing awaits labels, or
  the labeler is in failure backoff (`trainer_daemon/queue.py`
  synthesize_job, labeling checked first). Labeling GPU is chosen
  separately (`AUTO_LABEL_GPU` -> `DOGGY_LABEL_GPU` -> `run_prelabels`):
  yolo26x one-frame-per-call on real mirror frames, cpu=4: T4 21 frames/s,
  L4 38, L40S 81.5 -> L40S same $/frame as L4 at 2x speed, T4 priciest per
  frame. Batched ultralytics predict over a path list OOMs (17.6 GiB ask at
  batch 8 or 32) -- not worth chasing at ~10 min per 46k frames.
- **2026-10-11 — Training cost is the recipe, not the GPU.** The Sep 7
  run's epoch was 13,929 real frames + 41,787 pre-written blur/motion/dark
  copies (75% copies) and every run re-learned from stock yolo26n. Now:
  `augment` = auto|all|one|off (`build.augment_mode`; auto = all 3 copies
  below `AUTO_ALL_VARIANTS_BELOW` verdicted frames -- the new-install case --
  then ONE copy per frame, picked by a stable stem hash); auto batch 128
  (`BATCH_TIERS`; L40S 32->422, 64->468, 128->532, 256->540 img/s, so 128
  is the ceiling; ultralytics accumulates to an effective batch of 64, so 128
  also halves the optimizer steps); every run starts from the deployed
  model's weights `pipeline/models/champion.pt` (`training.start_weights`,
  fresh from yolo26n when absent) and a gate-deployed run replaces it
  (`pipeline.promote_champion`, copy-then-rename). Seeded from
  training-runs/20260907-011158/best.pt (NCNN md5 matched the Pi's model;
  backup in ~/watchdoggy-backups/). **A manual/Mac-side deploy must also
  copy its best.pt to pipeline/models/champion.pt**, or the next run warm-
  starts from a stale model. Robustness (stress row) is an indicator, not a
  gate: if "one" copy weakens it, set augment to "all".
- **2026-10-11 — Augment copies are for small datasets only.** `augment`
  auto = all 3 copies below `AUTO_ALL_VARIANTS_BELOW` (3,000 verdicted
  frames), OFF above (user-decided: 30k+ real camera frames cover night and
  blur; ultralytics' hsv_v covers brightness but nothing synthesizes blur).
  Watch the report's stress row (new vs deployed under blur/dark/shift):
  if blur catches slip, set augment to "one".
- **2026-10-11 — Warm start trains on ALL the data; gate on hand + auto
  labels (user-decided).** Training only on frames newer than the champion
  (plus 1:1 replay) was built and dropped the same day: after one copy per
  frame, batch 128 and the warm start, it saved ~$1-3 per run for a
  forgetting risk and cutoff bookkeeping. Every run fine-tunes
  `champion.pt` on the full dataset for the full schedule. The auto-train gate `min_new_labels` counts
  hand + auto labels (`queue.labels_since`), default 1000: hand labels
  alone run ~100 a month.
- **2026-10-11 — NCNN bundles export at the camera shape (480, 640); never
  pass imgsz when scoring a bundle.** The square export spent 25% of every
  inference on letterbox padding (160 of 640 rows grey); `EXPORT_SHAPE`
  is 1.30x faster per frame. Scored correctly it matches the square bundle
  on the exam (135/4 vs 137/3; 5-vs-3 discordant dog frames = noise) and
  PyTorch at that shape (136/4). Passing `imgsz=640` to a 640x480 bundle
  makes ultralytics square the frame for a graph built for 640x480 and the
  score collapses (109/7) -- that first looked like a pnnx bug; pnnx
  20260526 and 20260704 export identically. `evaluation._max_dog_conf` and
  `consensus._nano_scores` predict without imgsz (the jury runs the
  DEPLOYED bundle, so this fix had to reach the Pi before any 640x480
  model did); the Pi detector never passed one.
- **2026-10-11 — Full 120-epoch schedule, no early stopping
  (research-backed, user-decided).** `patience=epochs` in
  `training._train_local`: ultralytics' own default is epochs 100 /
  patience 100 (never fires); the gains come from the final phase
  (lr decays to lrf=1%, close_mosaic=10, the YOLOX recipe) and can appear
  after a plateau (epoch-wise double descent); our old patience 20 on a
  379-frame val quit before it. Epochs 80 -> 120: ultralytics accumulates
  to nbs=64 but never scales lr with batch, so batch 128 halves updates per
  epoch; 120 x ~29k / 128 ~= 27k updates, like the Sep 7 run that was still
  improving at its end. Auto optimizer is MuSGD (lr0 0.01) above 10k
  iterations, AdamW below. One-off experiment: train job param
  `compare_init` (job-only, never a setting) trains fresh-from-yolo26n
  beside the warm start on the same build, gates both, keeps the better
  (`pipeline.pick_start`, Ash & Adams 2020 warm-start gap); both scores
  land in recipe.init_compare.
- **2026-10-11 — The Modal Volume (v1) is the training bottleneck: too many
  files.** watchdoggy-train-2 is a v1 volume; Modal's guidance is under
  50k files (per-file latency scales with the count; hard cap 500k
  inodes). pipeline/mirror alone holds ~192k (jpg + json per frame, flat).
  Measured with a cold container: 4.9 frames/s serial, 14/s with 32
  threads, while a fresh small volume read the same 90 KB files at 431/s
  serial and 1,975/s parallel. Labeling 46k frames ran at ~6/s with the
  L40S 95% idle (~2h), and the container itself missed heartbeats
  ("Modal Client -> Modal Worker ... Deadline exceeded", UTC stamps, i.e.
  container-side) during the read storms. `dataset.read_frames` now reads
  ahead in parallel (prelabel, jury, build) and prelabel logs progress and
  checkpoints the cache every 2,000 frames. FIXED same day: pipeline now
  uses watchdoggy-train-3, a v2 volume (`modal volume create --version=2`;
  `Volume.from_name(..., version=2)`, never create_if_missing which makes
  v1), mirror files in day folders (`mirror_sync.shard_of` = first 5
  timestamp digits, ~1.16 days; v2 caps a folder at 262,144 files; readers
  go through `config.mirror_path`/`mirror_glob`, flat on a Mac pull).
  v2 read 8k sharded files at 1,120/s serial. Migration: Modal has no
  v1->v2 tool; a 48-container parallel copy moved 191,991 files (9.2 GB)
  in ~80 s (the v1 limit is per container), vs ~4.5 h re-uploading from
  the Pi. The Pi's upload manifest stayed valid. v2 is beta: the Pi holds
  the originals, so the worst case is a re-sync. Old volume kept briefly.
- **2026-10-11 — Cloud runs survive a dropped Pi; the jury batches across
  containers.** Kickoffs `spawn()` their cloud call, write its id to
  `jobs/<job>.call` (`--call-file`), then wait; the trainer runs
  `modal run --detach`, and on a non-zero exit or a hung-client timeout it
  runs `modal_pipeline.py::collect --call-id` (Modal keeps results 7
  days). Labeling is ONE cloud call (`run_label_pass`: GPU labeling, jury
  plan, `run_consensus_chunk.map` over 4,000-frame chunks, merge) so a
  disconnect can't strand the jury step. **Never batch an NCNN model in
  ultralytics:** its NCNN backend reads only `im[0]` of a batch and
  silently drops the rest; the jury and every gate/eval score bundles one
  frame per call and parallelize across containers instead. Not covered
  yet: a Pi reboot mid-run (the tmp out dir is gone; the next pass redoes
  the work, mostly cached).
- **2026-10-11 — Applying a big labeling pass on the Pi must be chunked
  and niced.** The month-backlog jury returned 38,927 auto-labels; the
  trainer posted them in ONE `apply-cloud-results` request (180 s
  timeout). The server needed ~17 min, the client timed out, and the
  per-frame fallback re-posted every frame over HTTPS at ~4/s (hours)
  while the server was still applying the batch; detector FPS fell to
  0.7-0.9 for the duration. Now `trainer_daemon/apply.py` sends
  `APPLY_CHUNK` (1,000) frames per request and falls back per-frame only
  for a failed chunk, and the handler lowers its thread to
  `WEB_WORKER_NICE`. A FastAPI sync handler keeps running after its
  client times out: check sidecar mtimes before assuming a timed-out
  write did nothing.
- **2026-10-11 — Machine boxes reach the Pi by "what the Pi lacks", not
  "what this pass computed".** The labeling pass that computed 46k boxes
  died on the empty jpg after its cache checkpoints; the rerun found them
  cached, so `fresh_prelabels` never sent them and 46,188 sidecars had no
  boxes on the label page (the jury and training read the cloud cache and
  were fine). The shipped index now carries a 5th field, `has_boxes`
  (`mirror_sync._write_index`); `pipeline.fresh_prelabels` sends new boxes
  plus every cached frame the index marks as lacking them. Index readers
  take `entry[:4]`. Unplugging the Pi mid-write leaves 0-byte jpg + json
  pairs (two on 2026-10-11 13:23): every reader must skip an empty file,
  and `apply-cloud-results` counts one as missing instead of failing the
  batch.
- **2026-10-11 — Mirror uploads are sized by bytes.** Every label change
  edits a sidecar, so the sync after a big labeling pass re-uploads tens of
  thousands of 1-3 KB json files. At a flat 100 files per request (plus a
  10 MB manifest rewrite per request on the SD card) 39k sidecars took
  ~20 min of a 26 min pass. `mirror_sync.upload_chunks` caps a request at
  8 MB or 2,000 files, which keeps jpg requests the same size.
