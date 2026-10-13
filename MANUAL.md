# The owner's manual

Everything about living with a Counter Watch appliance: the dashboard, the
self-training loop, the training console, and the machinery that keeps the
device current without you. Setup and deployment are in the
[README](README.md); the code layout is in [ARCHITECTURE.md](ARCHITECTURE.md).

## Using the dashboard

Open `http://<pi-host>:8000` from any device on your network.

- The status pill shows Watching, Dog spotted, or Cooling down.
- Draw the watch area by tapping corners around the counter on the live view,
  then Save area.
- Simple settings cover how sure it must be, how long the dog must linger,
  the wait between reactions, the hourly cap, and Ignore people.
- Advanced holds the detection-window and person-matching knobs. System shows
  temperature, power, and speed.
- The catch log shows each alarm with its clip; a one-tap "Not a dog" button
  turns any false alarm straight into training data and a permanent exam
  question.
- **Label** is the annotation tool; **Training** is the training console,
  including the software version, the auto-update toggle, and an Update now
  button.

Soothing sounds loops your own calm audio, music or white noise, through the
speaker while it watches. Upload tracks from the Soothing sounds card, up to
1 GB in total, then turn it on. The deterrent always takes priority, so an
alarm interrupts the music the moment a dog is confirmed and the loop resumes
on its own a little later.

![Weekly activity report, per-sound deterrence rates, and the soothing schedule](docs/watch-panels.png)

## HTTPS (for push-to-talk and notifications)

Browsers only allow the microphone and notifications on secure pages, so
those features need the dashboard served over https. One script sets it up:

```sh
./scripts/setup-https.sh <user@host>
```

It creates a "watchdoggy home CA" on the Pi, issues the dashboard a
certificate signed by it, and restarts the service. The CA never leaves your
Pi and nothing talks to the internet.

Then open the same address you always use, `http://<pi-host>:8000`, on each
device. The page checks whether that device already trusts the home
certificate. If it does, it sends you straight to the secure dashboard. If it
does not, it hands you the certificate and the one-time steps to trust it (it
detects your platform and offers the right file):

- iPhone/iPad: open the profile, install it in Settings, then Settings >
  General > About > Certificate Trust Settings > enable it
- Mac: open the file in Keychain Access, set Trust to Always
- Android: Settings > Security > Install a certificate > CA certificate

The page rechecks on its own, so as soon as the device trusts the certificate
it moves you along. After that the padlock is normal on every visit and your
old bookmark keeps working. The certificate lasts about two years; re-run the
script to renew it, and your devices keep working without any new steps. The
secure dashboard also has a direct address, `https://<pi-host>:8443`, once a
device is trusted.

## Configuration

Structural config (camera, ports, paths, TLS) is set with `DOGGY_*` environment
variables (see `.env.example`). Everything the dashboard changes live lives in
`settings.json` next to it, written by the app; every change is appended to
`settings-changes.jsonl` (when, which key, old, new, and the client that made
it) and shown by `GET /api/settings/history`. The first start after upgrading
migrates the tunables out of `.env` automatically.
Live-tunable params are also editable from the dashboard and persist the
moment you change them, so the appliance's own self-update restarts never
revert a toggle. Structural params (camera, model, audio backend) need a
restart. Training recipe, schedule, and the auto-update toggle live on the
Training page.

## The self-training loop

The stock COCO-trained model has never seen *your* kitchen from *your* camera
angle. Ours kept mistaking a person loading the dishwasher for a dog. The fix
is a closed loop that lives on the appliance:

> **Privacy:** this loop is optional and only runs after cloud training is
> enabled ([Pi guide, step 6](docs/pi/README.md#6-cloud-training-and-self-updates-optional)).
> Live detection stays local either way. With it on, the frames saved in step 1,
> including frames with people in them, are uploaded to the Modal volume in
> your own account for labeling and training.

1. **Capture.** The detector saves interesting frames as they happen: every
   alarm (plus the raw seconds before it), borderline detections, suppressed
   boxes, person activity, periodic background shots, and "flicker" moments
   where the model keeps changing its mind about the same scene, which is a
   stronger uncertainty signal than any single score. A brightness failsafe
   skips lights-off frames, and a perceptual-hash check drops near-duplicates
   of dogless scenes, so neither darkness nor a long cooking session can
   flood the queue. Frames containing a dog are never thinned: on a fixed
   camera the unchanging background dominates the hash, so distinct dog
   moments look alike to it. Storage is
   capped, and when the cap is reached the oldest unlabeled frames go first;
   labeled frames are never deleted, the appliance warns instead.
2. **Machine labeling, weekly.** A cloud pass runs once a week (at 2am by
   default): a large model (yolo26x) draws `·x` boxes for every new frame,
   and a **two-model
   jury** (the deployed fine-tune plus the big model) auto-labels the frames
   it can vouch for. A frame is auto-labeled dog only when both models see
   the dog, and it trains on the big model's box. When the deployed model is
   sure of a dog (0.85 or higher) and the big model sees nothing, the frame
   is flagged Disputed for you instead. The jury's thresholds are backtested against the full human-labeled corpus with the current champion
   as juror, and re-audited when the champion changes. In the latest backtest
   the shipped rules mislabel about 0.1% of dog frames and fabricate none,
   while clearing about a third of a fresh queue per pass. The jury also
   remembers: frames it declined stay declined until the model or the rules
   change, so no pass pays to re-judge the same undecidable frames. It also
   **audits existing labels**: when both models strongly contradict a human
   verdict, the frame is flagged Disputed for re-review. Anything the jury is
   unsure about goes to the human.
3. **Human labeling, only the disagreements.** The `/label` page is a fixed
   stage with a filmstrip and filter chips: the Queue (what the jury couldn't
   settle), Disputed (your own labels a model doubts come first), Auto (spot-check the machine's work; one tap overrules
   forever), and Needs-boxes finders. Verdicts are one keystroke; a full box
   editor handles frames needing hand-drawn truth, which then outranks every
   model. In training, your labels count double the jury's.

   ![Label page: filter chips, the big model's boxes, filmstrip navigation, one-tap verdicts](docs/label.png)
4. **Train.** Every two weeks (configurable), right after a labeling pass
   and only once at least 1,000 frames have been labeled since the last run
   (yours and the jury's together), the trainer sends one job to a Modal
   cloud GPU: label fusion, near-duplicate pruning of dogless frames,
   training, frame-level evaluation scored the way the alarm fires,
   NCNN export, and a robustness stress test (blur, darkness, overexposure,
   compression, camera shift). Auto-labeled frames train but never enter
   the exam.

   Each run starts from the model that is currently deployed and fine-tunes
   it on the whole dataset, so it never re-learns the kitchen from scratch.
   Every run goes the full 120 epochs and keeps its best epoch. A
   brand-new install, with no model of its own yet, starts from
   the stock weights. Blur, motion and darkness copies of every frame are
   added only while the dataset is small (under 3,000 labeled frames); after
   that the camera's own footage covers those conditions and the copies
   would mostly cost time.
5. **Gate.** Challenger and incumbent sit the **identical, 100%
   human-verified held-out exam**, judged at the appliance's runtime
   threshold, and **the best model wins**: fewest total errors (missed dogs
   plus false fires), with false fires as tie-breaker and always-visible
   indicator. Every false alarm you ever flagged from the catch log is a
   permanent exam member, so a mistake that reached you keeps being tested
   forever. Reports break the exam into failure-mode slices (dark scenes,
   person present, small or edge-of-frame dogs, fire-origin frames) for both
   models side by side, and include confidence-calibration diagnostics, so an
   improving average can't hide a regressing corner. The previous model is
   kept on disk for instant rollback.
6. **Repeat.** Every mistake it makes becomes training data against it, and
   the exam grows harder as the dataset grows.

Your only job is a few minutes of arbitration on the Label page now and then.
Everything else (weekly prelabels, consensus auto-labeling, label audits,
scheduled training, evaluation, gated deployment) happens on its own.

### Small datasets get extra copies

While a setup has fewer than 3,000 labeled frames, every training frame goes
in three more times: once with motion blur (a dog trotting past at the
camera's roughly 2 frames a second), once out of focus, and once darkened to
dusk. These change only brightness and sharpness, so every box still fits.
Only training frames get copies; the exam always uses the original frames.

On the first dataset of 168 frames, the copies decided the stress test. The
model trained with them caught 22 or 23 of 23 held-out dogs under every
condition (dark, bright, blur, JPEG compression, shifted camera) with no
false alarms. The same training without copies caught 19 or 20 and raised
one false alarm.

The copies make every epoch four times bigger, which is cheap on a small
dataset. Past 3,000 labeled frames the "auto" setting stops adding them: the
camera's own footage covers night and motion blur by then, and the copies
would make up three quarters of every epoch.

### What a run costs, and why

Cloud cost is GPU time, and GPU time is images processed divided by speed.
The defaults come from benchmarks at this project's exact training settings:

- **GPU: L40S, for training and labeling.** It trained about 420 images a
  second against 180 on an L4, and labeled about 81 frames a second against
  38, at the same or lower cost per image. A faster H100 was only slightly
  quicker and about 1.7 times the price per image; an A100 was no faster
  than an L4. Adding CPU cores changed nothing.
- **Batch 128.** The GPUs spend much of each step on fixed overhead, so a
  bigger batch gets more done per step. 128 was about a quarter faster than
  32; going to 256 gained nothing more. Tiny first datasets keep a small
  batch.
- **Fine-tuning the deployed model, without copies on a big dataset.**
  Together these cut the cost of each epoch by about four times.
- **120 epochs, always run to the end.** Ultralytics lowers the learning
  rate over the planned epochs and switches mosaic off for the last 10;
  that last stretch is where late gains come from (it is the YOLOX
  recipe), and models can keep improving after an apparent plateau
  ("epoch-wise double descent", Nakkiran et al. 2020). Stopping early on
  the small exam quit before it. Batch 128 also halves the weight updates
  per epoch, so 120 epochs gives about as many updates as the September
  run that was still improving when it ended. A run costs about $6.

Starting each run from the deployed model can generalize slightly worse
than training fresh on everything (Ash and Adams 2020). The first run after
this change trains both from the same data and keeps whichever scores
better on the exam, to measure it here.

The training page can override the GPU, batch and copies settings, for
example "all 3 copies" if the robustness table ever shows blur handling
slipping.

### Training the labeler

The big model that draws the `·x` boxes and sits on the jury is stock
yolo26x. Once the small model has trained on your kitchen, it can beat the
big one: on this kitchen's exam the deployed small model caught 173 of 190
dogs and stock yolo26x caught 96. The jury needs both models to agree, so a
weak big model sends good frames to your queue and leaves dogs without boxes.

The **Train labeler** button on the training page fine-tunes yolo26x on your
kitchen. It never runs on a schedule.

- **Data.** Every hand-labeled frame outside the exam, counted three times,
  plus 8,000 auto-labels both models agreed on, picked the same way each run.
  Frames taken within 30 seconds of an exam frame stay out of training, so
  the exam stays a fair test. Labels keep COCO's classes, so the model keeps
  everything stock yolo26x knows.
- **Recipe.** 100 epochs from the stock weights, the first 10 layers frozen,
  batch 16.
- **Gate.** Both labelers label every exam frame at the label bar (0.45).
  The new one is swapped in only when all three hold: it makes fewer
  mistakes, it wins the frames where the two disagree by more than chance
  would give (a one-sided sign test, p below 0.05), and replaying the jury's
  rules with its boxes clears no more real dogs and invents no more dogs
  than the current labeler. A smaller win reads "inconclusive" and the
  current labeler stays. The run's report shows the full table.
- **Exam labels it doubts.** Every exam frame you labeled without a dog
  where the new labeler sees one is flagged Disputed for you, whatever the
  decision. When those frames do show the dog, relabeling them can turn a
  held-back labeler into a swap on its next run.
- **After a swap.** The trainer queues a labeling pass. It relabels every
  frame with the new labeler, the jury forgets the frames it declined
  before, and it re-judges every auto-label once. Auto-labels it no longer
  backs go back to your queue. Hand labels are never touched.
- **Cost.** It trains about 89 images a second on an L40S, so a full run
  over this kitchen's 10,000-image epoch takes about 3.5 hours and costs
  about $9. The 3-epoch probe took 15 minutes in the cloud and cost $0.77.

To check the whole path cheaply first, enter 3 at the epochs prompt.

**Rollback.** The stock model and its boxes stay on the volume. Delete the
fine-tuned weights with
`modal volume rm watchdoggy-train-3 pipeline/models/labeler.pt`, then press
**Fetch ·x boxes now**. That pass reuses the stock boxes it already has
(only frames added since the swap get labeled) and the jury re-judges the
auto-labels with the stock model.

## The training console

The `/training` page is the console: the live model's scorecard (scored at
the threshold the detector runs with), dataset composition, real
cloud spend against your credit allowance, an improvement chart across runs,
the cloud run's stages and log streaming live while it works, recipe and
schedule knobs, and every run's full report (threshold curves, gate
comparison, robustness table, failure-mode slices).

The knobs, with their defaults:

| Setting | Default | What it does |
|---|---|---|
| label every (h) | 168 (weekly) | How often the labeling pass runs |
| labeling at (h) | 2 | Hour of day the labeling pass starts |
| train every (h, after labeling) | 336 (two weeks) | Minimum gap between training runs; a run only starts right after a labeling pass |
| min new labels (hand + auto) | 1000 | New labels needed since the last run before training starts |
| blur/dark copies | auto | auto = 3 copies per frame under 3,000 labeled frames, none after; or all 3, 1 per frame, off |
| batch | auto | auto = 128 (16 for a tiny first dataset) |
| training gpu | auto | auto = L40S; labeling also uses L40S |
| monthly credits ($) | 30 | Hard stop for cloud spend (see below) |

![Training console: live model score, dataset composition, cloud budget, and the improvement chart](docs/training.png)

Billing is pulled straight from Modal and every run records its true cost.
The monthly credits setting is a hard stop: once spend reaches it, or
anything has been billed beyond credits, the trainer refuses to start cloud
jobs and says so, both on the budget card and in the refused job's history
entry. Self-updates still run (they come from GitHub), and cloud
jobs resume by themselves when the billing cycle resets.

Modal requires a payment method on file before it runs any GPU job, even
inside the free credits. The trainer's stop works between jobs, so a run
that starts just under the limit can still finish a few dollars over it;
setting a workspace spending limit in Modal's own billing settings is a
good backstop.

## It maintains itself

The same sandboxed trainer that talks to the cloud GPU also keeps the
appliance itself current:

- **Self-updates from GitHub releases.** Once a day (a dashboard toggle, on
  by default) it checks the repo's latest release over plain unauthenticated
  HTTPS; there is no GitHub credential anywhere on the device. A newer tag
  becomes a job in the same queue as training runs: download, verify,
  snapshot the current code, install through a fixed-path root helper,
  restart, then health-poll the detector for two minutes. If the appliance
  doesn't come back healthy it rolls itself back. Releases that change
  dependencies are refused with a note on the dashboard, because the offline
  appliance cannot install packages; those need one deploy from a
  workstation.
- **Keeps its own clock.** The Pi has no battery clock and the firewall
  blocks public NTP, so each trainer pass syncs time from an HTTPS Date
  header (the htpdate trick) through a bounds-checked root helper. On its
  first day it caught the clock 52 seconds off.
- **Snapshots state before every change.** Each self-update first copies the
  dataset sidecars, job history, models, and config aside on the Pi (newest
  two snapshots kept). `scripts/pull-backup.sh` pulls the same state to your
  workstation in one command, and the cloud volume mirrors the dataset with
  every training run, so labels exist in three places.

## Privacy details

Live detection stays local. Optional cloud training sends saved camera frames
to your own cloud account.

- The detector service has no internet access, enforced by an nftables egress
  firewall (`scripts/harden-pi.sh`, LAN only). The live stream is only served
  on your LAN.
- Cloud training is opt-in and runs as a separate `trainer` user. A per-UID
  firewall exception lets only that user reach the internet (DNS + HTTPS).
  When enabled, every frame the Pi saves for training is uploaded to your
  Modal volume, including frames with people in them. The trainer also
  contacts `api.github.com` for release checks and the time.
- To stop uploads, disable `doggy-trainer.timer`. Already-uploaded frames stay
  in your Modal volume until you delete them.
- Skip `setup-pi-trainer.sh` and nothing leaves the Pi. You can still run the
  same training pipeline manually from a workstation
  (`scripts/train_kitchen_model.py`, including a 10-config hyperparameter
  sweep with `--sweep`).
