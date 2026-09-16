# FollowCam — Claude Code project notes

## Trigger: "execute both"

When the user says **"execute both"** (or "execute the plans"), run the two
implementation plans below with the `superpowers:subagent-driven-development`
skill (fallback: `superpowers:executing-plans`). They are independent and may
run in parallel on separate branches; each ends with a PR against `main`.

1. `docs/superpowers/plans/2026-09-16-spark-training-pipeline.md` — Spark training pipeline (`ml/spark/`)
2. `docs/superpowers/plans/2026-09-16-multiview-sync.md` — multi-view synced recording (iOS `app/`, `ingest/`, `vision/multiview/`)

Design they implement: `docs/superpowers/specs/2026-09-16-spark-training-and-multiview-design.md`.
"execute plan 1" / "execute plan 2" runs one of them.

## Environment facts (verified 2026-09-16)

- Python on this Mac: `ml/.venv/bin/python` (3.14, ultralytics 8.4, pytest). The
  `.venv/` the older docs mention does not exist here. Never touch system Python.
- Tests: `ml/.venv/bin/python -m pytest ml/spark/tests ingest/tests vision/multiview/tests -q`.
- iOS: `cd app && xcodegen generate` after adding Swift files; tests with
  `xcodebuild test -scheme FollowCam -destination 'platform=iOS Simulator,name=iPhone 17 Pro'`.
- Training hardware: NVIDIA DGX Spark (GB10, 128 GB, ARM64). See `ml/spark/SETUP.md` once it exists.
- Hackathon weights and clips (`models/`, `data/`) are gitignored and not on this Mac.
- `main` is PR-only for everyone except Malek. Commit messages end with
  `Co-Authored-By: WOZCODE <contact@withwoz.com>`.
