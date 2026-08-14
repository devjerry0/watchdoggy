"""The Pi's trainer daemon: one pass per invocation (systemd timer, ~30 min).

Runs as the dedicated `trainer` user -- the only UID on the appliance with
cloud egress (nftables exception; the detector stays provably offline). Each
pass: pick up a queued job from the web UI, or synthesize one from the auto
rules, then send it to Modal and apply the results:

  prelabel  fresh big-model boxes for new frames, merged into sidecars via
            the local web API (so the review page shows them).
            Auto rule: >= AUTO_PRELABEL_MIN frames lack prelabels.
  train     the full cloud pipeline; if the deploy gate passes, the returned
            NCNN bundle is installed and the detector restarted.
            Auto rule: >= MIN_NEW_LABELS frames labeled since the last done
            train job AND that job is older than TRAIN_INTERVAL seconds.

Stdlib only. Progress is written to job_<id>.result.json files, which the
web's /api/training/status overlays on the originals (different users can't
edit each other's files). Modules: `env` (paths/settings/local API),
`queue` (job files + auto rules), `cloud` (Modal + billing), `apply`
(pushing results back into the appliance), `runs` (the two job runners),
`daemon` (main).
"""
