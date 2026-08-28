# FollowCam analytics pipeline (PIPELINE role, docs/tasks/PIPELINE.md)
#   make demo CLIP=data/clips/dev60.mp4      whole pipeline on one clip (skips steps that are current)
#   make demo CLIP=... ARGS="--force"        rerun everything; ARGS is passed to vision/run_all.py
#   make smoke                               10 s cut of dev60 through the pipeline, contract check, waits for a free GPU
#   make monitor                             status board on http://127.0.0.1:8600
#   make plan CLIP=...                       dry run, shows the commands per step

PY   := .venv/bin/python
CLIP ?= data/clips/dev60.mp4
OUT  ?= out
ARGS ?=

.PHONY: demo smoke plan monitor help

help:
	@sed -n '2,7p' $(MAKEFILE_LIST)

demo:
	$(PY) -m vision.run_all --clip $(CLIP) --out-dir $(OUT) $(ARGS)

plan:
	$(PY) -m vision.run_all --clip $(CLIP) --out-dir $(OUT) --dry-run $(ARGS)

smoke:
	$(PY) -m vision.smoke_test $(ARGS)

monitor:
	$(PY) -m vision.monitor.serve
