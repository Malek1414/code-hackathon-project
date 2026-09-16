# Spark Training Pipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A repeatable pipeline under `ml/spark/` that trains a four-class YOLO11 detector (player, ball, hoop, referee) on the NVIDIA DGX Spark, reports ball recall and the longest ball-less gap, and exports Core ML weights on the Mac.

**Architecture:** Small, single-purpose Python scripts that share one class map (`ml/spark/classes.yaml`) and one dataset layout (`data/spark/dataset`, YOLO format). Pure functions (remap, split, metrics) live in importable modules and are unit-tested on the Mac; the GPU-heavy commands (`train`, `predict`) run on the Spark inside NVIDIA's PyTorch container and are driven by committed config files. Everything under `data/spark/`, `runs/spark/` and `models/` is gitignored and synced with `rsync`.

**Tech Stack:** Python 3.14 (`ml/.venv` on the Mac), ultralytics 8.4, PyYAML, numpy, Pillow, pytest; on the Spark `nvcr.io/nvidia/pytorch` container, ARM64, CUDA; Roboflow Python SDK for downloads; coremltools via `yolo export` on macOS.

Spec: `docs/superpowers/specs/2026-09-16-spark-training-and-multiview-design.md`, Part A.

## Global Constraints

- Class ids are fixed: `0 player`, `1 ball`, `2 hoop`, `3 referee`. Every source is remapped to these.
- Train/val split is by source video or source dataset, never by random frame.
- Any clip from our own camera is held out entirely and is never trained on while it is the eval clip.
- Nothing under `data/spark/`, `runs/spark/`, `models/` is committed. `.gitignore` already covers `/data/`, `runs/`, `*.pt`, `models/*.pt`; add `*.mlpackage`.
- Mac Python is `ml/.venv/bin/python` (3.14, ultralytics 8.4.131, pytest 9.1). Never `pip install` into system Python.
- Tests run with `ml/.venv/bin/python -m pytest ml/spark/tests -q` from the repo root.
- Every script has a `main()` guarded by `if __name__ == "__main__":` and an argparse `--help`.
- Round-one models: YOLO11s and YOLO11m, imgsz 1280, 100 epochs, patience 20, `copy_paste=0.3`, `scale=0.5`, `close_mosaic=10`.
- Acceptance for Part A: longest ball-less gap under 2 s on the newest held-out rig clip, then the live court test.
- Commit after every task on the working branch; commit messages end with
  `Co-Authored-By: WOZCODE <contact@withwoz.com>`.

---

## File map

| Path | Responsibility |
|---|---|
| `ml/spark/__init__.py` | package marker |
| `ml/spark/classes.yaml` | canonical class map |
| `ml/spark/classes.py` | load the class map, remap a YOLO label line, remap a label file |
| `ml/spark/sources.yaml` | list of sources: Roboflow ids, local dirs, per-source `names` map |
| `ml/spark/fetch_public.py` | download Roboflow sources, import any YOLO-format dir with remap into `data/spark/sources/<name>/` |
| `ml/spark/merge_dataset.py` | group images by source video, split, write `data/spark/dataset/` + `data.yaml` + `manifest.json` |
| `ml/spark/train.py` | run one config on the Spark, copy `best.pt` to `models/` |
| `ml/spark/configs/{smoke,s_1280,m_1280}.yaml` | run configs |
| `ml/spark/metrics.py` | pure metrics: IoU, ball recall, FP per frame, longest gap |
| `ml/spark/eval_ball.py` | ball report on a labeled dir or a clip, worst-gap contact sheet |
| `ml/spark/pseudolabel.py` | frames → predictions → YOLO labels + review sheet + `review.json` |
| `ml/spark/export_coreml.py` | Mac-only Core ML export at 640 and 960 |
| `ml/spark/SETUP.md` | Spark access, container, rsync, first run |
| `ml/spark/requirements.txt` | Mac-side extras (pyyaml, pillow, imageio-ffmpeg) |
| `ml/spark/tests/` | pytest suite |
| `Makefile` | `spark-data`, `spark-train`, `spark-eval`, `spark-export`, `spark-smoke`, `spark-test` |

---

### Task 1: Package scaffold, class map and label remapping

**Files:**
- Create: `ml/spark/__init__.py`, `ml/spark/classes.yaml`, `ml/spark/classes.py`, `ml/spark/requirements.txt`, `ml/spark/tests/__init__.py`, `ml/spark/tests/test_classes.py`
- Modify: `.gitignore` (add `*.mlpackage`, `data/spark/`)

**Interfaces:**
- Produces: `classes.load_classes(path=CLASSES_YAML) -> dict[int, str]`, `classes.remap_line(line: str, id_map: dict[int, int]) -> str | None`, `classes.remap_file(src: Path, dst: Path, id_map: dict[int, int]) -> int` (returns boxes written), `classes.build_id_map(source_names: dict[int, str], canonical: dict[int, str], aliases: dict[str, str]) -> dict[int, int]`.

- [ ] **Step 1: Install Mac-side extras into `ml/.venv`**

```bash
cat > ml/spark/requirements.txt <<'EOF'
pyyaml>=6
pillow>=10
imageio-ffmpeg>=0.5
numpy>=1.26
EOF
ml/.venv/bin/python -m pip install -r ml/spark/requirements.txt
```

- [ ] **Step 2: Write the class map**

```bash
mkdir -p ml/spark/tests && touch ml/spark/__init__.py ml/spark/tests/__init__.py
cat > ml/spark/classes.yaml <<'EOF'
# Canonical FollowCam classes. Every source is remapped to these ids.
names:
  0: player
  1: ball
  2: hoop
  3: referee
# Source class names that mean the same thing (lowercased, matched exactly).
aliases:
  person: player
  players: player
  basketball-player: player
  basketball: ball
  sports ball: ball
  balls: ball
  rim: hoop
  basket: hoop
  net: hoop
  ref: referee
  refs: referee
  referees: referee
EOF
```

- [ ] **Step 3: Write the failing tests**

```python
# ml/spark/tests/test_classes.py
from pathlib import Path

from ml.spark.classes import build_id_map, load_classes, remap_file, remap_line


def test_load_classes_reads_canonical_ids():
    names = load_classes()
    assert names == {0: "player", 1: "ball", 2: "hoop", 3: "referee"}


def test_build_id_map_uses_aliases_and_drops_unknown():
    canonical = {0: "player", 1: "ball", 2: "hoop", 3: "referee"}
    aliases = {"person": "player", "basketball": "ball", "rim": "hoop"}
    src = {0: "Person", 1: "basketball", 2: "Rim", 3: "scoreboard"}
    assert build_id_map(src, canonical, aliases) == {0: 0, 1: 1, 2: 2}


def test_remap_line_rewrites_id_and_drops_unmapped():
    id_map = {5: 1}
    assert remap_line("5 0.5 0.5 0.1 0.1", id_map) == "1 0.5 0.5 0.1 0.1"
    assert remap_line("7 0.5 0.5 0.1 0.1", id_map) is None
    assert remap_line("   ", id_map) is None


def test_remap_file_writes_only_mapped_boxes(tmp_path: Path):
    src = tmp_path / "a.txt"
    src.write_text("0 0.1 0.1 0.2 0.2\n3 0.5 0.5 0.1 0.1\n9 0.9 0.9 0.1 0.1\n")
    dst = tmp_path / "out" / "a.txt"
    n = remap_file(src, dst, {0: 0, 3: 3})
    assert n == 2
    assert dst.read_text().splitlines() == ["0 0.1 0.1 0.2 0.2", "3 0.5 0.5 0.1 0.1"]


def test_remap_file_creates_empty_file_when_nothing_maps(tmp_path: Path):
    src = tmp_path / "a.txt"
    src.write_text("9 0.9 0.9 0.1 0.1\n")
    dst = tmp_path / "a_out.txt"
    assert remap_file(src, dst, {0: 0}) == 0
    assert dst.exists() and dst.read_text() == ""
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_classes.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.spark.classes'`

- [ ] **Step 5: Implement `classes.py`**

```python
# ml/spark/classes.py
"""Canonical class map and YOLO label remapping."""
from __future__ import annotations

from pathlib import Path

import yaml

CLASSES_YAML = Path(__file__).with_name("classes.yaml")


def load_classes(path: Path = CLASSES_YAML) -> dict[int, str]:
    data = yaml.safe_load(path.read_text())
    return {int(k): str(v) for k, v in data["names"].items()}


def load_aliases(path: Path = CLASSES_YAML) -> dict[str, str]:
    data = yaml.safe_load(path.read_text())
    return {str(k).lower(): str(v) for k, v in (data.get("aliases") or {}).items()}


def build_id_map(source_names: dict[int, str], canonical: dict[int, str],
                 aliases: dict[str, str]) -> dict[int, int]:
    """Map a source's class ids onto canonical ids. Unknown names are dropped."""
    by_name = {v: k for k, v in canonical.items()}
    out: dict[int, int] = {}
    for sid, name in source_names.items():
        key = name.strip().lower()
        key = aliases.get(key, key)
        if key in by_name:
            out[int(sid)] = by_name[key]
    return out


def remap_line(line: str, id_map: dict[int, int]) -> str | None:
    parts = line.split()
    if len(parts) < 5:
        return None
    cid = int(parts[0])
    if cid not in id_map:
        return None
    return " ".join([str(id_map[cid]), *parts[1:5]])


def remap_file(src: Path, dst: Path, id_map: dict[int, int]) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    kept = [l for l in (remap_line(x, id_map) for x in src.read_text().splitlines()) if l]
    dst.write_text("\n".join(kept) + ("\n" if kept else ""))
    return len(kept)
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_classes.py -q`
Expected: `5 passed`

- [ ] **Step 7: Update `.gitignore` and commit**

```bash
printf '\n# Spark training pipeline\ndata/spark/\n*.mlpackage\n' >> .gitignore
git add .gitignore ml/spark
git commit -m "spark: class map and label remapping

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 2: Source import and public dataset fetch

**Files:**
- Create: `ml/spark/sources.yaml`, `ml/spark/fetch_public.py`, `ml/spark/tests/test_fetch_public.py`

**Interfaces:**
- Consumes: `classes.build_id_map`, `classes.remap_file`, `classes.load_classes`, `classes.load_aliases`.
- Produces: `fetch_public.import_yolo_dir(src_root: Path, dest: Path, source_name: str, names: dict[int, str]) -> dict` (returns `{"name", "images", "boxes": {cls: n}}`), `fetch_public.read_data_yaml_names(path: Path) -> dict[int, str]`, `fetch_public.video_key(filename: str) -> str`. Imported sources land in `data/spark/sources/<name>/{images,labels}/*.{jpg,txt}` flat, plus `source.json` with the counts.

A source is any directory holding YOLO images and labels (Roboflow exports have `train/images`, `train/labels`, `valid/...`, `test/...`; the bootstrap set has `images/train`, `labels/train`). The importer finds every `.txt` under the root, pairs it with the image of the same stem, and copies both into one flat folder per source. The merge step (Task 3) decides train/val, so a Roboflow "test" split is just more images here.

- [ ] **Step 1: Write `sources.yaml`**

```yaml
# ml/spark/sources.yaml
# Each source is imported into data/spark/sources/<name>/ with canonical class ids.
# kind: roboflow -> downloaded via the Roboflow SDK (ROBOFLOW_API_KEY in .env)
# kind: local    -> an existing YOLO-format directory on disk
# `names` overrides the class names found in the source's data.yaml when that
# file is missing or wrong (source id -> source class name).
sources:
  - name: bootstrap_youtube
    kind: local
    path: ml/data/dataset
    names: {0: person, 1: ball}
    # frames are named <youtube_id>_<frame>.jpg; the youtube id is the video key
  # Fill in two or three Roboflow Universe basketball sets that carry ball,
  # hoop and referee. workspace/project/version come from the dataset page URL.
  - name: rf_basketball_a
    kind: roboflow
    workspace: CHANGE_ME
    project: CHANGE_ME
    version: 1
  - name: rf_basketball_b
    kind: roboflow
    workspace: CHANGE_ME
    project: CHANGE_ME
    version: 1
```

The `CHANGE_ME` entries are filled by whoever runs `make spark-data` the first time; `fetch_public.py` refuses to download a source whose fields still say `CHANGE_ME` and prints the Roboflow Universe search URL to pick from (`https://universe.roboflow.com/search?q=basketball%20ball%20hoop%20referee`). Choose sets whose class list includes ball, hoop and referee (or an alias in `classes.yaml`), preferring gym and sideline angles over broadcast.

- [ ] **Step 2: Write the failing tests**

```python
# ml/spark/tests/test_fetch_public.py
import json
from pathlib import Path

from PIL import Image

from ml.spark.fetch_public import import_yolo_dir, read_data_yaml_names, video_key


def _img(p: Path):
    p.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8)).save(p)


def test_video_key_strips_frame_suffix():
    assert video_key("CakjTJXij7g_000330.jpg") == "CakjTJXij7g"
    assert video_key("game10_f_00012.jpg") == "game10_f"
    assert video_key("single.jpg") == "single"


def test_read_data_yaml_names_accepts_list_and_dict(tmp_path: Path):
    (tmp_path / "a.yaml").write_text("names: ['Ball', 'Rim']\n")
    (tmp_path / "b.yaml").write_text("names:\n  0: player\n  1: ball\n")
    assert read_data_yaml_names(tmp_path / "a.yaml") == {0: "Ball", 1: "Rim"}
    assert read_data_yaml_names(tmp_path / "b.yaml") == {0: "player", 1: "ball"}


def test_import_yolo_dir_flattens_splits_and_remaps(tmp_path: Path):
    src = tmp_path / "rf"
    _img(src / "train" / "images" / "x_000001.jpg")
    (src / "train" / "labels").mkdir(parents=True)
    (src / "train" / "labels" / "x_000001.txt").write_text("0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.1 0.1\n")
    _img(src / "valid" / "images" / "y_000001.jpg")
    (src / "valid" / "labels").mkdir(parents=True)
    (src / "valid" / "labels" / "y_000001.txt").write_text("2 0.5 0.5 0.1 0.1\n")
    dest = tmp_path / "out"
    summary = import_yolo_dir(src, dest, "rf_test", names={0: "Ball", 1: "Rim", 2: "scoreboard"})
    assert summary["images"] == 2
    assert summary["boxes"] == {"ball": 1, "hoop": 1}
    assert (dest / "images" / "x_000001.jpg").exists()
    assert (dest / "labels" / "x_000001.txt").read_text() == "1 0.5 0.5 0.1 0.1\n2 0.2 0.2 0.1 0.1\n"
    assert (dest / "labels" / "y_000001.txt").read_text() == ""
    assert json.loads((dest / "source.json").read_text())["name"] == "rf_test"
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_fetch_public.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.spark.fetch_public'`

- [ ] **Step 4: Implement `fetch_public.py`**

```python
# ml/spark/fetch_public.py
"""Import training sources into data/spark/sources/<name>/ with canonical ids.

  ml/.venv/bin/python -m ml.spark.fetch_public                 # all sources in sources.yaml
  ml/.venv/bin/python -m ml.spark.fetch_public --only bootstrap_youtube
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import Counter
from pathlib import Path

import yaml

from ml.spark.classes import build_id_map, load_aliases, load_classes, remap_file

REPO = Path(__file__).resolve().parents[2]
SOURCES_YAML = Path(__file__).with_name("sources.yaml")
SOURCES_DIR = REPO / "data" / "spark" / "sources"
IMG_EXT = (".jpg", ".jpeg", ".png")
_FRAME_SUFFIX = re.compile(r"_\d+$")


def video_key(filename: str) -> str:
    """Group frames of one video: strip the extension and a trailing _<digits>."""
    stem = Path(filename).stem
    return _FRAME_SUFFIX.sub("", stem)


def read_data_yaml_names(path: Path) -> dict[int, str]:
    names = yaml.safe_load(path.read_text())["names"]
    if isinstance(names, list):
        return {i: str(n) for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def _find_image(label: Path) -> Path | None:
    # Roboflow: <split>/labels/x.txt <-> <split>/images/x.jpg
    # bootstrap: labels/train/x.txt  <-> images/train/x.jpg
    for cand_dir in (label.parent.parent / "images", Path(str(label.parent).replace("labels", "images"))):
        for ext in IMG_EXT:
            p = cand_dir / (label.stem + ext)
            if p.exists():
                return p
    return None


def import_yolo_dir(src_root: Path, dest: Path, source_name: str,
                    names: dict[int, str] | None = None) -> dict:
    if names is None:
        yml = next(iter(src_root.rglob("data.yaml")), None)
        if yml is None:
            raise SystemExit(f"{src_root}: no data.yaml and no names given")
        names = read_data_yaml_names(yml)
    canonical = load_classes()
    id_map = build_id_map(names, canonical, load_aliases())
    if not id_map:
        raise SystemExit(f"{source_name}: none of {names} maps to {canonical}")
    (dest / "images").mkdir(parents=True, exist_ok=True)
    (dest / "labels").mkdir(parents=True, exist_ok=True)
    boxes: Counter[str] = Counter()
    images = 0
    for label in sorted(src_root.rglob("*.txt")):
        if label.name == "classes.txt" or "labels" not in label.parts:
            continue
        img = _find_image(label)
        if img is None:
            continue
        shutil.copy2(img, dest / "images" / img.name)
        out_label = dest / "labels" / (img.stem + ".txt")
        remap_file(label, out_label, id_map)
        for line in out_label.read_text().splitlines():
            boxes[canonical[int(line.split()[0])]] += 1
        images += 1
    summary = {"name": source_name, "images": images, "boxes": dict(boxes), "id_map": id_map}
    (dest / "source.json").write_text(json.dumps(summary, indent=1))
    return summary


def _download_roboflow(entry: dict, into: Path) -> Path:
    if "CHANGE_ME" in (entry.get("workspace", ""), entry.get("project", "")):
        raise SystemExit(f"{entry['name']}: fill workspace/project/version in sources.yaml. "
                         "Pick from https://universe.roboflow.com/search?q=basketball%20ball%20hoop%20referee")
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        raise SystemExit("ROBOFLOW_API_KEY not set (put it in .env and `set -a; . .env`)")
    from roboflow import Roboflow  # imported lazily: only needed for kind=roboflow

    project = Roboflow(api_key=key).workspace(entry["workspace"]).project(entry["project"])
    ds = project.version(int(entry["version"])).download("yolov11", location=str(into), overwrite=True)
    return Path(ds.location)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=SOURCES_YAML)
    ap.add_argument("--dest", type=Path, default=SOURCES_DIR)
    ap.add_argument("--only", nargs="*", default=None, help="source names to import")
    a = ap.parse_args()
    entries = yaml.safe_load(a.sources.read_text())["sources"]
    for e in entries:
        if a.only and e["name"] not in a.only:
            continue
        dest = a.dest / e["name"]
        if e["kind"] == "local":
            root = REPO / e["path"]
        elif e["kind"] == "roboflow":
            root = _download_roboflow(e, a.dest / "_downloads" / e["name"])
        else:
            raise SystemExit(f"{e['name']}: unknown kind {e['kind']}")
        names = {int(k): str(v) for k, v in e["names"].items()} if e.get("names") else None
        s = import_yolo_dir(root, dest, e["name"], names)
        print(f"{s['name']}: {s['images']} images, boxes {s['boxes']}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_fetch_public.py -q`
Expected: `3 passed`

- [ ] **Step 6: Import the bootstrap source for real and check the counts**

Run: `ml/.venv/bin/python -m ml.spark.fetch_public --only bootstrap_youtube`
Expected: a line like `bootstrap_youtube: 1115 images, boxes {'player': N, 'ball': M}` and `data/spark/sources/bootstrap_youtube/source.json` present. (Exact N and M are whatever the bootstrap labels contain; write them into the commit message.)

- [ ] **Step 7: Commit**

```bash
ml/.venv/bin/python -m pip install roboflow
git add ml/spark/sources.yaml ml/spark/fetch_public.py ml/spark/tests/test_fetch_public.py
git commit -m "spark: import YOLO sources with canonical ids, Roboflow fetch

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 3: Merge sources into one dataset with a source-level split

**Files:**
- Create: `ml/spark/merge_dataset.py`, `ml/spark/tests/test_merge_dataset.py`

**Interfaces:**
- Consumes: `fetch_public.video_key`, `classes.load_classes`, the per-source folders from Task 2.
- Produces: `merge_dataset.split_groups(groups: list[str], val_frac: float, seed: int, holdout: set[str]) -> dict[str, str]` (group → `"train"|"val"|"holdout"`), `merge_dataset.build(sources_dir: Path, dataset_dir: Path, val_frac: float = 0.1, seed: int = 0, holdout: set[str] = frozenset()) -> dict` (the manifest), and on disk `data/spark/dataset/{images,labels}/{train,val}/`, `data/spark/dataset/data.yaml`, `data/spark/dataset/manifest.json`.

Group key for an image is `<source_name>/<video_key(filename)>`. Roboflow sets that are not video frames end up with many one-image groups, which is fine: the split is still by image for those and by video for ours. `holdout` names groups that go to neither split (our own eval clips).

- [ ] **Step 1: Write the failing tests**

```python
# ml/spark/tests/test_merge_dataset.py
import json
from pathlib import Path

from PIL import Image

from ml.spark.merge_dataset import build, split_groups


def _source(root: Path, name: str, files: dict[str, str]):
    (root / name / "images").mkdir(parents=True)
    (root / name / "labels").mkdir(parents=True)
    for fname, label in files.items():
        Image.new("RGB", (8, 8)).save(root / name / "images" / fname)
        (root / name / "labels" / (Path(fname).stem + ".txt")).write_text(label)
    (root / name / "source.json").write_text(json.dumps({"name": name}))


def test_split_groups_is_deterministic_and_respects_holdout():
    groups = [f"g{i}" for i in range(20)]
    a = split_groups(groups, val_frac=0.2, seed=1, holdout={"g3"})
    b = split_groups(groups, val_frac=0.2, seed=1, holdout={"g3"})
    assert a == b
    assert a["g3"] == "holdout"
    assert sum(v == "val" for v in a.values()) == 4
    assert set(a.values()) == {"train", "val", "holdout"}


def test_build_never_puts_one_video_in_both_splits(tmp_path: Path):
    src = tmp_path / "sources"
    # two videos with 5 frames each, and one "video" of single images
    _source(src, "vid", {f"A_{i:06d}.jpg": "1 0.5 0.5 0.1 0.1\n" for i in range(5)}
            | {f"B_{i:06d}.jpg": "0 0.5 0.5 0.1 0.1\n" for i in range(5)})
    _source(src, "rf", {f"img{i}.jpg": "2 0.5 0.5 0.1 0.1\n" for i in range(10)})
    ds = tmp_path / "dataset"
    manifest = build(src, ds, val_frac=0.5, seed=0)
    train = {p.name for p in (ds / "images" / "train").iterdir()}
    val = {p.name for p in (ds / "images" / "val").iterdir()}
    assert not (train & val)
    for vid in ("A_", "B_"):
        frames = {n for n in train | val if n.startswith(vid)}
        assert frames <= train or frames <= val, f"{vid} split across train and val"
    assert (ds / "data.yaml").exists()
    assert manifest["classes"] == {"0": "player", "1": "ball", "2": "hoop", "3": "referee"}
    assert manifest["sources"]["vid"]["images"] == 10
    assert manifest["split"]["train"] + manifest["split"]["val"] == 20
    assert all((ds / "labels" / s / (Path(n).stem + ".txt")).exists()
               for s, names in (("train", train), ("val", val)) for n in names)


def test_build_holdout_is_excluded(tmp_path: Path):
    src = tmp_path / "sources"
    _source(src, "own", {f"rig01_{i:06d}.jpg": "1 0.5 0.5 0.1 0.1\n" for i in range(4)})
    ds = tmp_path / "dataset"
    manifest = build(src, ds, val_frac=0.5, seed=0, holdout={"own/rig01"})
    assert manifest["split"]["holdout"] == 4
    assert not list((ds / "images" / "train").iterdir()) and not list((ds / "images" / "val").iterdir())
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_merge_dataset.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.spark.merge_dataset'`

- [ ] **Step 3: Implement `merge_dataset.py`**

```python
# ml/spark/merge_dataset.py
"""Merge data/spark/sources/* into data/spark/dataset with a source-level split.

  ml/.venv/bin/python -m ml.spark.merge_dataset [--val-frac 0.1] [--seed 0] [--holdout own/rig01 ...]
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import yaml

from ml.spark.classes import load_classes
from ml.spark.fetch_public import IMG_EXT, REPO, SOURCES_DIR, video_key

DATASET_DIR = REPO / "data" / "spark" / "dataset"


def split_groups(groups: list[str], val_frac: float, seed: int,
                 holdout: set[str] = frozenset()) -> dict[str, str]:
    rest = sorted(g for g in groups if g not in holdout)
    rng = random.Random(seed)
    rng.shuffle(rest)
    n_val = int(round(len(rest) * val_frac))
    out = {g: "holdout" for g in groups if g in holdout}
    out.update({g: "val" for g in rest[:n_val]})
    out.update({g: "train" for g in rest[n_val:]})
    return out


def build(sources_dir: Path, dataset_dir: Path, val_frac: float = 0.1, seed: int = 0,
          holdout: set[str] = frozenset()) -> dict:
    classes = load_classes()
    if dataset_dir.exists():
        shutil.rmtree(dataset_dir)
    for split in ("train", "val"):
        (dataset_dir / "images" / split).mkdir(parents=True)
        (dataset_dir / "labels" / split).mkdir(parents=True)

    by_group: dict[str, list[tuple[str, Path]]] = defaultdict(list)
    for src in sorted(p for p in sources_dir.iterdir() if (p / "images").is_dir()):
        for img in sorted(src.glob("images/*")):
            if img.suffix.lower() in IMG_EXT:
                by_group[f"{src.name}/{video_key(img.name)}"].append((src.name, img))

    assignment = split_groups(list(by_group), val_frac, seed, holdout)
    split_counts: Counter[str] = Counter()
    per_source: dict[str, dict] = defaultdict(lambda: {"images": 0, "boxes": Counter(), "train": 0, "val": 0, "holdout": 0})
    for group, items in by_group.items():
        split = assignment[group]
        for source_name, img in items:
            label = img.parent.parent / "labels" / (img.stem + ".txt")
            s = per_source[source_name]
            s["images"] += 1
            s[split] += 1
            split_counts[split] += 1
            if label.exists():
                for line in label.read_text().splitlines():
                    if line.strip():
                        s["boxes"][classes[int(line.split()[0])]] += 1
            if split == "holdout":
                continue
            out_name = f"{source_name}__{img.name}"
            shutil.copy2(img, dataset_dir / "images" / split / out_name)
            dst_label = dataset_dir / "labels" / split / f"{source_name}__{img.stem}.txt"
            if label.exists():
                shutil.copy2(label, dst_label)
            else:
                dst_label.write_text("")

    (dataset_dir / "data.yaml").write_text(yaml.safe_dump({
        "path": str(dataset_dir), "train": "images/train", "val": "images/val",
        "names": {k: v for k, v in classes.items()}}, sort_keys=False))
    manifest = {
        "classes": {str(k): v for k, v in classes.items()},
        "val_frac": val_frac, "seed": seed, "holdout": sorted(holdout),
        "split": {k: split_counts.get(k, 0) for k in ("train", "val", "holdout")},
        "sources": {k: {**v, "boxes": dict(v["boxes"])} for k, v in per_source.items()},
    }
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest, indent=1))
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", type=Path, default=SOURCES_DIR)
    ap.add_argument("--dataset", type=Path, default=DATASET_DIR)
    ap.add_argument("--val-frac", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--holdout", nargs="*", default=[], help="group keys <source>/<video> kept out of both splits")
    a = ap.parse_args()
    m = build(a.sources, a.dataset, a.val_frac, a.seed, set(a.holdout))
    print(json.dumps({"split": m["split"], "sources": {k: v["images"] for k, v in m["sources"].items()}}, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_merge_dataset.py -q`
Expected: `3 passed`

- [ ] **Step 5: Build the dataset from the bootstrap source and check the split is by video**

Run: `ml/.venv/bin/python -m ml.spark.merge_dataset && ml/.venv/bin/python - <<'EOF'
import json; m=json.load(open("data/spark/dataset/manifest.json")); print(m["split"], list(m["sources"]))
EOF`
Expected: `split` shows train and val counts summing to the bootstrap image count, and since the bootstrap set has two videos, one of the two is entirely in val or both are in train (with 2 groups and val_frac 0.1, `round(0.2)=0`, so val is empty: expected for now; the Roboflow sources fill val in the real run). Note this in `SETUP.md` in Task 4.

- [ ] **Step 6: Commit**

```bash
git add ml/spark/merge_dataset.py ml/spark/tests/test_merge_dataset.py
git commit -m "spark: merge sources with a source-level split

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 4: Training runner, configs, Spark setup and Makefile

**Files:**
- Create: `ml/spark/train.py`, `ml/spark/configs/smoke.yaml`, `ml/spark/configs/s_1280.yaml`, `ml/spark/configs/m_1280.yaml`, `ml/spark/SETUP.md`, `ml/spark/tests/test_train.py`
- Modify: `Makefile`

**Interfaces:**
- Consumes: `data/spark/dataset/data.yaml` from Task 3.
- Produces: `train.load_config(path: Path) -> dict` (validated, with defaults filled), `train.train_kwargs(cfg: dict, data_yaml: Path, project: Path) -> dict` (the exact kwargs passed to `YOLO.train`), weights at `models/followcam_<config_name>.pt`, run dir `runs/spark/<config_name>/`.

- [ ] **Step 1: Write the configs**

```yaml
# ml/spark/configs/smoke.yaml  (proves the loop; one epoch, tiny)
model: yolo11n.pt
imgsz: 640
epochs: 1
batch: 8
fraction: 0.05      # 5 percent of the train images
patience: 20
```

```yaml
# ml/spark/configs/s_1280.yaml
model: yolo11s.pt
imgsz: 1280
epochs: 100
batch: -1           # ultralytics auto-batch on the first run; pin the number it picks
patience: 20
optimizer: AdamW
cos_lr: true
scale: 0.5
copy_paste: 0.3
close_mosaic: 10
```

```yaml
# ml/spark/configs/m_1280.yaml
model: yolo11m.pt
imgsz: 1280
epochs: 100
batch: -1
patience: 20
optimizer: AdamW
cos_lr: true
scale: 0.5
copy_paste: 0.3
close_mosaic: 10
```

- [ ] **Step 2: Write the failing tests**

```python
# ml/spark/tests/test_train.py
from pathlib import Path

import pytest

from ml.spark.train import load_config, train_kwargs

CONFIGS = Path(__file__).resolve().parents[1] / "configs"


def test_every_committed_config_loads():
    for p in sorted(CONFIGS.glob("*.yaml")):
        cfg = load_config(p)
        assert cfg["name"] == p.stem
        assert cfg["model"].endswith(".pt") and cfg["imgsz"] % 32 == 0


def test_defaults_are_filled(tmp_path: Path):
    p = tmp_path / "x.yaml"
    p.write_text("model: yolo11n.pt\nimgsz: 640\nepochs: 2\n")
    cfg = load_config(p)
    assert cfg["batch"] == -1 and cfg["patience"] == 20 and cfg["fraction"] == 1.0


def test_unknown_key_is_rejected(tmp_path: Path):
    p = tmp_path / "x.yaml"
    p.write_text("model: yolo11n.pt\nimgsz: 640\nepochs: 2\nlearning_rate: 1\n")
    with pytest.raises(SystemExit, match="learning_rate"):
        load_config(p)


def test_train_kwargs_pin_name_and_project(tmp_path: Path):
    cfg = load_config(CONFIGS / "s_1280.yaml")
    kw = train_kwargs(cfg, tmp_path / "data.yaml", tmp_path / "runs")
    assert kw["name"] == "s_1280" and kw["project"] == str(tmp_path / "runs")
    assert kw["exist_ok"] is True and kw["imgsz"] == 1280 and kw["copy_paste"] == 0.3
    assert "model" not in kw and "name" in kw
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_train.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.spark.train'`

- [ ] **Step 4: Implement `train.py`**

```python
# ml/spark/train.py
"""Train one config on the Spark (or anywhere ultralytics runs).

  python -m ml.spark.train --config ml/spark/configs/s_1280.yaml [--device 0] [--data data/spark/dataset/data.yaml]

Writes runs/spark/<config>/ and copies best.pt to models/followcam_<config>.pt.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[2]
DEFAULTS = {"batch": -1, "patience": 20, "fraction": 1.0, "optimizer": "auto", "cos_lr": False,
            "scale": 0.5, "copy_paste": 0.0, "close_mosaic": 10, "workers": 8}
ALLOWED = {"model", "imgsz", "epochs", *DEFAULTS}


def load_config(path: Path) -> dict:
    cfg = yaml.safe_load(path.read_text()) or {}
    unknown = set(cfg) - ALLOWED
    if unknown:
        raise SystemExit(f"{path}: unknown keys {sorted(unknown)}; allowed {sorted(ALLOWED)}")
    for k in ("model", "imgsz", "epochs"):
        if k not in cfg:
            raise SystemExit(f"{path}: missing {k}")
    return {**DEFAULTS, **cfg, "name": path.stem}


def train_kwargs(cfg: dict, data_yaml: Path, project: Path) -> dict:
    kw = {k: v for k, v in cfg.items() if k not in ("model", "name")}
    kw.update(data=str(data_yaml), project=str(project), name=cfg["name"], exist_ok=True)
    return kw


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, required=True)
    ap.add_argument("--data", type=Path, default=REPO / "data" / "spark" / "dataset" / "data.yaml")
    ap.add_argument("--project", type=Path, default=REPO / "runs" / "spark")
    ap.add_argument("--device", default="0", help="cuda index, 'cpu' or 'mps'")
    a = ap.parse_args()
    from ultralytics import YOLO  # lazy: keeps --help and tests light

    cfg = load_config(a.config)
    model = YOLO(cfg["model"])
    model.train(device=a.device, **train_kwargs(cfg, a.data, a.project))
    best = a.project / cfg["name"] / "weights" / "best.pt"
    out = REPO / "models" / f"followcam_{cfg['name']}.pt"
    out.parent.mkdir(exist_ok=True)
    shutil.copy2(best, out)
    metrics = model.val(data=str(a.data), imgsz=cfg["imgsz"], device=a.device, plots=False)
    names = metrics.names
    print(f"\nbest weights: {out}")
    print("class            P      R   mAP50  mAP50-95")
    for i, cls_idx in enumerate(metrics.box.ap_class_index):
        p, r, m50, m = metrics.box.class_result(i)
        print(f"{names[int(cls_idx)]:<12} {p:6.3f} {r:6.3f} {m50:7.3f} {m:9.3f}")
    print(f"{'all':<12} {metrics.box.mp:6.3f} {metrics.box.mr:6.3f} {metrics.box.map50:7.3f} {metrics.box.map:9.3f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Run tests to verify they pass**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_train.py -q`
Expected: `4 passed`

- [ ] **Step 6: Add Makefile targets**

Append to `Makefile` (keep the existing targets):

```make

# ---- Spark training pipeline (ml/spark, docs/superpowers/plans/2026-09-16-spark-training-pipeline.md)
#   make spark-test                        unit tests on the Mac
#   make spark-data                        import sources + merge dataset (Mac or Spark)
#   make spark-smoke                       one-epoch smoke run (any device)
#   make spark-train CONFIG=s_1280         train one config (on the Spark, inside the container)
#   make spark-eval WEIGHTS=... CLIP=...    ball report on a clip
#   make spark-export WEIGHTS=...          Core ML export (Mac only)
MLPY   ?= ml/.venv/bin/python
CONFIG ?= smoke
DEVICE ?= 0
WEIGHTS ?= models/followcam_s_1280.pt
HOLDOUT ?=

.PHONY: spark-test spark-data spark-smoke spark-train spark-eval spark-export

spark-test:
	$(MLPY) -m pytest ml/spark/tests -q

spark-data:
	$(MLPY) -m ml.spark.fetch_public
	$(MLPY) -m ml.spark.merge_dataset --holdout $(HOLDOUT)

spark-smoke:
	$(MLPY) -m ml.spark.train --config ml/spark/configs/smoke.yaml --device $(DEVICE)

spark-train:
	$(MLPY) -m ml.spark.train --config ml/spark/configs/$(CONFIG).yaml --device $(DEVICE)

spark-eval:
	$(MLPY) -m ml.spark.eval_ball --weights $(WEIGHTS) --clip $(CLIP) --device $(DEVICE)

spark-export:
	$(MLPY) -m ml.spark.export_coreml --weights $(WEIGHTS)
```

- [ ] **Step 7: Run the smoke config on the Mac to prove the loop before touching the Spark**

Run: `make spark-smoke DEVICE=mps` (falls back to `DEVICE=cpu` if MPS errors)
Expected: ultralytics trains 1 epoch on 5 percent of the bootstrap frames, prints the per-class table, and `models/followcam_smoke.pt` exists. Takes a few minutes on the M2.

- [ ] **Step 8: Write `SETUP.md`**

```markdown
# Spark setup (DGX Spark, GB10, 128 GB unified, ARM64 DGX OS)

## Access
- Host: `spark.local` (fill in the real hostname or Tailscale name). User: `followcam`.
- `ssh followcam@spark.local`. Keys only; nothing in git.
- Project folder on the Spark: `~/followcam` (a clone of this repo).

## Sync
From the Mac, data and weights (gitignored) go over rsync:
    rsync -av --progress data/spark/ followcam@spark.local:~/followcam/data/spark/
    rsync -av followcam@spark.local:~/followcam/models/ models/          # weights back
    rsync -av followcam@spark.local:~/followcam/runs/spark/ runs/spark/  # curves, val plots

## Container
    docker run --gpus all -it --rm --ipc=host \
      -v ~/followcam:/workspace/followcam -w /workspace/followcam \
      nvcr.io/nvidia/pytorch:<TAG> bash
    pip install ultralytics==8.4.* pyyaml pillow roboflow imageio-ffmpeg
    python -c "import torch;print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
Pin `<TAG>` in this file the first time the line above prints `True GB10`.
Inside the container the Makefile targets run with `MLPY=python`:
    make spark-smoke MLPY=python DEVICE=0
    make spark-train MLPY=python DEVICE=0 CONFIG=s_1280

## First run checklist
1. `make spark-test` on the Mac is green.
2. `make spark-data` on the Mac (needs ROBOFLOW_API_KEY in .env and the
   sources.yaml entries filled). With only the bootstrap source the val split is
   empty because it has two videos; the Roboflow sources are what fill val.
3. rsync `data/spark/` to the Spark.
4. In the container: `make spark-smoke MLPY=python` (one epoch, minutes).
5. `make spark-train MLPY=python CONFIG=s_1280`, then `CONFIG=m_1280`. The
   first run with `batch: -1` prints the batch it picked; write that number
   into the config and commit it.
6. rsync `models/` and `runs/spark/` back. Record the val table in
   docs/RESULTS.md under "Spark round one".
```

- [ ] **Step 9: Commit**

```bash
git add ml/spark/train.py ml/spark/configs ml/spark/SETUP.md ml/spark/tests/test_train.py Makefile
git commit -m "spark: training runner, configs, setup notes, make targets

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 5: Ball metrics and the ball report

**Files:**
- Create: `ml/spark/metrics.py`, `ml/spark/eval_ball.py`, `ml/spark/tests/test_metrics.py`

**Interfaces:**
- Produces: `metrics.iou(a, b) -> float` on `(x1, y1, x2, y2)`; `metrics.ball_recall(preds: list[list[tuple[box, conf]]], gts: list[list[box]], conf: float, iou_thr: float = 0.3) -> dict` with keys `recall`, `tp`, `fn`, `fp`, `fp_per_frame`; `metrics.longest_gap(has_ball: list[bool], fps: float) -> dict` with keys `longest_gap_s`, `longest_gap_frames`, `gap_start_frame`, `coverage`. Files: `out/spark/eval_<weights-stem>_<clip-stem>.json` and `out/spark/eval_<...>_gaps.jpg`.

- [ ] **Step 1: Write the failing tests**

```python
# ml/spark/tests/test_metrics.py
from ml.spark.metrics import ball_recall, iou, longest_gap


def test_iou_basic():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (10, 10, 20, 20)) == 0.0
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_ball_recall_counts_tp_fn_fp_at_threshold():
    gts = [[(0, 0, 10, 10)], [(0, 0, 10, 10)], []]
    preds = [[((1, 1, 11, 11), 0.9)],            # tp
             [((50, 50, 60, 60), 0.9)],          # fp, and the gt is fn
             [((0, 0, 10, 10), 0.2)]]            # below conf: ignored
    r = ball_recall(preds, gts, conf=0.35)
    assert r == {"recall": 0.5, "tp": 1, "fn": 1, "fp": 1, "fp_per_frame": 1 / 3}


def test_ball_recall_handles_no_ground_truth():
    assert ball_recall([[]], [[]], conf=0.35)["recall"] is None


def test_longest_gap_measures_the_longest_false_run():
    has = [True, False, False, False, True, False, True, True]
    g = longest_gap(has, fps=2.0)
    assert g == {"longest_gap_frames": 3, "longest_gap_s": 1.5, "gap_start_frame": 1, "coverage": 0.5}


def test_longest_gap_all_seen_and_all_missing():
    assert longest_gap([True, True], fps=30)["longest_gap_frames"] == 0
    assert longest_gap([False] * 60, fps=30)["longest_gap_s"] == 2.0
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_metrics.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.spark.metrics'`

- [ ] **Step 3: Implement `metrics.py`**

```python
# ml/spark/metrics.py
"""Pure ball metrics. Boxes are (x1, y1, x2, y2) in pixels."""
from __future__ import annotations

Box = tuple[float, float, float, float]


def iou(a: Box, b: Box) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / area


def ball_recall(preds: list[list[tuple[Box, float]]], gts: list[list[Box]],
                conf: float, iou_thr: float = 0.3) -> dict:
    tp = fn = fp = 0
    for frame_preds, frame_gts in zip(preds, gts):
        kept = [b for b, c in frame_preds if c >= conf]
        matched = set()
        for g in frame_gts:
            hit = next((i for i, p in enumerate(kept) if i not in matched and iou(p, g) >= iou_thr), None)
            if hit is None:
                fn += 1
            else:
                tp += 1
                matched.add(hit)
        fp += len(kept) - len(matched)
    n_gt = tp + fn
    return {"recall": (tp / n_gt) if n_gt else None, "tp": tp, "fn": fn, "fp": fp,
            "fp_per_frame": fp / len(preds) if preds else 0.0}


def longest_gap(has_ball: list[bool], fps: float) -> dict:
    best = run = 0
    best_start = start = 0
    for i, seen in enumerate(has_ball):
        if seen:
            run = 0
            continue
        if run == 0:
            start = i
        run += 1
        if run > best:
            best, best_start = run, start
    return {"longest_gap_frames": best, "longest_gap_s": best / fps,
            "gap_start_frame": best_start if best else 0,
            "coverage": (sum(has_ball) / len(has_ball)) if has_ball else 0.0}
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_metrics.py -q`
Expected: `5 passed`

- [ ] **Step 5: Implement `eval_ball.py`**

```python
# ml/spark/eval_ball.py
"""Ball report for a model, on a labeled YOLO dir or on a clip.

  python -m ml.spark.eval_ball --weights models/followcam_s_1280.pt --labeled data/spark/dataset --split val
  python -m ml.spark.eval_ball --weights models/followcam_s_1280.pt --clip data/clips/rig01.mp4 --conf 0.35

Labeled mode: recall at IoU 0.3 and FP per frame for conf in 0.25/0.35/0.45.
Clip mode: fraction of frames with a ball, longest gap in seconds, and a
contact sheet of the 8 frames around the longest gap (out/spark/*_gaps.jpg).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw

from ml.spark.classes import load_classes
from ml.spark.metrics import ball_recall, longest_gap

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "out" / "spark"
BALL = 1


def _yolo_to_xyxy(line: str, w: int, h: int):
    c, cx, cy, bw, bh = line.split()[:5]
    cx, cy, bw, bh = float(cx) * w, float(cy) * h, float(bw) * w, float(bh) * h
    return int(c), (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)


def eval_labeled(model, root: Path, split: str, imgsz: int, device: str, confs: list[float]) -> dict:
    images = sorted((root / "images" / split).glob("*"))
    gts, preds = [], []
    for img in images:
        w, h = Image.open(img).size
        label = root / "labels" / split / (img.stem + ".txt")
        g = []
        if label.exists():
            for line in label.read_text().splitlines():
                if line.strip():
                    c, box = _yolo_to_xyxy(line, w, h)
                    if c == BALL:
                        g.append(box)
        gts.append(g)
        r = model.predict(str(img), imgsz=imgsz, conf=min(confs), device=device, verbose=False)[0]
        preds.append([(tuple(b.xyxy[0].tolist()), float(b.conf[0])) for b in r.boxes if int(b.cls[0]) == BALL])
    return {"mode": "labeled", "split": split, "images": len(images),
            "by_conf": {str(c): ball_recall(preds, gts, conf=c) for c in confs}}


def eval_clip(model, clip: Path, imgsz: int, device: str, conf: float, sheet: Path) -> dict:
    import cv2

    cap = cv2.VideoCapture(str(clip))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    has: list[bool] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        r = model.predict(frame, imgsz=imgsz, conf=conf, device=device, verbose=False)[0]
        has.append(any(int(b.cls[0]) == BALL for b in r.boxes))
    cap.release()
    g = longest_gap(has, fps)
    _gap_sheet(clip, has, g["gap_start_frame"], sheet)
    return {"mode": "clip", "clip": str(clip), "fps": fps, "frames": len(has), "conf": conf, **g,
            "gap_sheet": str(sheet)}


def _gap_sheet(clip: Path, has: list[bool], start: int, out: Path) -> None:
    """Second pass: seek to the longest gap and tile 8 frames around it. Keeps memory flat on long clips."""
    import cv2

    cap = cv2.VideoCapture(str(clip))
    first = max(0, start - 2)
    cap.set(cv2.CAP_PROP_POS_FRAMES, first)
    tiles = []
    for i in range(first, min(first + 8, len(has))):
        ok, f = cap.read()
        if not ok:
            break
        f = cv2.resize(f, (480, 270))
        cv2.putText(f, f"f{i} {'ball' if has[i] else 'no ball'}", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 255, 0) if has[i] else (0, 0, 255), 2)
        tiles.append(f)
    cap.release()
    if not tiles:
        return
    while len(tiles) % 4:
        tiles.append(tiles[-1] * 0)
    rows = [cv2.hconcat(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), cv2.vconcat(rows) if len(rows) > 1 else rows[0])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--labeled", type=Path, help="YOLO dataset root")
    ap.add_argument("--split", default="val")
    ap.add_argument("--clip", type=Path)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--conf", type=float, default=0.35, help="clip mode threshold")
    ap.add_argument("--device", default="0")
    a = ap.parse_args()
    if bool(a.labeled) == bool(a.clip):
        ap.error("give exactly one of --labeled or --clip")
    from ultralytics import YOLO

    model = YOLO(str(a.weights))
    assert load_classes()[BALL] == "ball"
    if a.labeled:
        rep = eval_labeled(model, a.labeled, a.split, a.imgsz, a.device, [0.25, 0.35, 0.45])
        name = f"eval_{a.weights.stem}_{a.labeled.name}_{a.split}"
    else:
        name = f"eval_{a.weights.stem}_{a.clip.stem}"
        rep = eval_clip(model, a.clip, a.imgsz, a.device, a.conf, OUT / f"{name}_gaps.jpg")
    rep["weights"] = str(a.weights)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(rep, indent=1))
    print(json.dumps(rep, indent=1))


if __name__ == "__main__":
    main()
```

- [ ] **Step 6: Run the labeled report against the smoke weights to prove the CLI**

Run: `ml/.venv/bin/python -m ml.spark.eval_ball --weights models/followcam_smoke.pt --labeled data/spark/dataset --split train --imgsz 640 --device cpu`
Expected: JSON printed with `by_conf` for 0.25, 0.35, 0.45 and the file `out/spark/eval_followcam_smoke_dataset_train.json`. Recall numbers are meaningless for the smoke model; the point is that the script runs end to end. (Use `--split train` because val is empty until the Roboflow sources are in.)

- [ ] **Step 7: Commit**

```bash
git add ml/spark/metrics.py ml/spark/eval_ball.py ml/spark/tests/test_metrics.py
git commit -m "spark: ball recall, longest-gap metric and ball report CLI

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 6: Pseudo-label loop with review sheet

**Files:**
- Create: `ml/spark/pseudolabel.py`, `ml/spark/tests/test_pseudolabel.py`

**Interfaces:**
- Consumes: a trained `.pt`; optionally `out/<session>/fused_ball.json` from the multi-view plan, whose shape is `{"slot_ms": 100, "views": {"<view_id>": {"offset_ms": 0}}, "slots": [{"t_ms": 0, "covered": true, "best_view": "<view_id>", "views": {"<view_id>": {"conf": 0.81}}}]}`.
- Produces: `pseudolabel.flag_frames(preds: dict[int, list[tuple[Box, float]]], low: float = 0.35, high: float = 0.6) -> list[dict]` (each `{"frame": i, "reason": "low_conf"|"dropout"}`), `pseudolabel.priority_from_fused(fused: dict, view_id: str, fps: float, min_other_conf: float = 0.7) -> set[int]` (frame indices of this view where another view saw the ball confidently and this one did not), on disk `data/spark/sources/<clip-stem>/{images,labels}/` (same layout Task 2 produces, so Task 3 can merge it) and `out/spark/review_<clip-stem>/{review.json, sheet_000.jpg, ...}`.

- [ ] **Step 1: Write the failing tests**

```python
# ml/spark/tests/test_pseudolabel.py
from ml.spark.pseudolabel import flag_frames, priority_from_fused


def test_flag_frames_low_conf_and_dropout():
    preds = {0: [((0, 0, 5, 5), 0.9)], 1: [], 2: [((0, 0, 5, 5), 0.4)], 3: [((0, 0, 5, 5), 0.9)]}
    flags = flag_frames(preds, low=0.35, high=0.6)
    assert {(f["frame"], f["reason"]) for f in flags} == {(1, "dropout"), (2, "low_conf")}


def test_flag_frames_ignores_isolated_empty_frames_without_neighbours():
    preds = {0: [], 1: []}
    assert flag_frames(preds) == []


def test_priority_from_fused_picks_frames_other_view_saw():
    fused = {"slot_ms": 100, "views": {"A": {"offset_ms": 0}, "B": {"offset_ms": 0}},
             "slots": [{"t_ms": 0, "covered": True, "best_view": "A", "views": {"A": {"conf": 0.9}, "B": {"conf": 0.0}}},
                       {"t_ms": 100, "covered": True, "best_view": "A", "views": {"A": {"conf": 0.5}, "B": {"conf": 0.0}}},
                       {"t_ms": 200, "covered": True, "best_view": "B", "views": {"A": {"conf": 0.0}, "B": {"conf": 0.8}}}]}
    assert priority_from_fused(fused, "B", fps=10.0) == {0}
    assert priority_from_fused(fused, "A", fps=10.0) == {2}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_pseudolabel.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'ml.spark.pseudolabel'`

- [ ] **Step 3: Implement `pseudolabel.py`**

```python
# ml/spark/pseudolabel.py
"""Pseudo-label a clip with a trained model and write a review sheet.

  python -m ml.spark.pseudolabel --clip data/clips/rig01.mp4 --weights models/followcam_s_1280.pt [--fps 2] [--fused out/<session>/fused_ball.json --view <view_id>]

Frames go to data/spark/sources/<clip-stem>/ (YOLO layout, canonical ids) so
`merge_dataset` can pick them up; review sheets to out/spark/review_<clip-stem>/.
A human opens review.json, fixes or approves flagged frames (any YOLO label
editor works on the source folder), then reruns `make spark-data`.
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

import imageio_ffmpeg
from PIL import Image, ImageDraw

from ml.spark.classes import load_classes

REPO = Path(__file__).resolve().parents[2]
BALL = 1
CONF = {0: 0.5, 1: 0.35, 2: 0.5, 3: 0.5}
Box = tuple[float, float, float, float]


def flag_frames(preds: dict[int, list[tuple[Box, float]]], low: float = 0.35, high: float = 0.6) -> list[dict]:
    flags = []
    for i in sorted(preds):
        balls = [c for _, c in preds[i]]
        if balls and max(balls) < high and max(balls) >= low:
            flags.append({"frame": i, "reason": "low_conf"})
        elif not balls and any(c >= high for _, c in preds.get(i - 1, []) + preds.get(i + 1, [])):
            flags.append({"frame": i, "reason": "dropout"})
    return flags


def priority_from_fused(fused: dict, view_id: str, fps: float, min_other_conf: float = 0.7) -> set[int]:
    offset = fused["views"][view_id]["offset_ms"]
    out = set()
    for slot in fused["slots"]:
        mine = slot["views"].get(view_id, {}).get("conf", 0.0)
        others = [v["conf"] for k, v in slot["views"].items() if k != view_id]
        if mine == 0.0 and others and max(others) >= min_other_conf:
            out.add(int(round((slot["t_ms"] - offset) / 1000.0 * fps)))
    return out


def extract_frames(clip: Path, fps: float, dest: Path) -> list[Path]:
    dest.mkdir(parents=True, exist_ok=True)
    subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(), "-y", "-loglevel", "error", "-i", str(clip),
                    "-vf", f"fps={fps}", "-q:v", "2", str(dest / f"{clip.stem}_%06d.jpg")], check=True)
    return sorted(dest.glob(f"{clip.stem}_*.jpg"))


def _sheet(paths: list[Path], labels: dict[Path, list[tuple[int, Box, float]]], out: Path) -> None:
    tiles = []
    for p in paths:
        im = Image.open(p).convert("RGB").resize((480, 270))
        w, h = Image.open(p).size
        d = ImageDraw.Draw(im)
        for c, (x1, y1, x2, y2), conf in labels.get(p, []):
            d.rectangle([x1 / w * 480, y1 / h * 270, x2 / w * 480, y2 / h * 270],
                        outline=(255, 128, 0) if c == BALL else (0, 200, 0), width=2)
        d.text((4, 4), p.name, fill=(255, 255, 0))
        tiles.append(im)
    cols, rows = 4, (len(tiles) + 3) // 4
    sheet = Image.new("RGB", (480 * cols, 270 * rows))
    for i, t in enumerate(tiles):
        sheet.paste(t, ((i % cols) * 480, (i // cols) * 270))
    sheet.save(out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--clip", type=Path, required=True)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--imgsz", type=int, default=1280)
    ap.add_argument("--device", default="0")
    ap.add_argument("--fused", type=Path, help="out/<session>/fused_ball.json from the multi-view pipeline")
    ap.add_argument("--view", help="this clip's view id inside --fused")
    a = ap.parse_args()
    from ultralytics import YOLO

    classes = load_classes()
    src = REPO / "data" / "spark" / "sources" / a.clip.stem
    frames = extract_frames(a.clip, a.fps, src / "images")
    (src / "labels").mkdir(exist_ok=True)
    model = YOLO(str(a.weights))
    preds: dict[int, list[tuple[Box, float]]] = {}
    drawn: dict[Path, list] = {}
    for i, p in enumerate(frames):
        r = model.predict(str(p), imgsz=a.imgsz, conf=min(CONF.values()), device=a.device, verbose=False)[0]
        w, h = r.orig_shape[1], r.orig_shape[0]
        lines, balls, boxes = [], [], []
        for b in r.boxes:
            c, conf = int(b.cls[0]), float(b.conf[0])
            if conf < CONF[c]:
                continue
            x1, y1, x2, y2 = b.xyxy[0].tolist()
            lines.append(f"{c} {(x1 + x2) / 2 / w:.6f} {(y1 + y2) / 2 / h:.6f} {(x2 - x1) / w:.6f} {(y2 - y1) / h:.6f}")
            boxes.append((c, (x1, y1, x2, y2), conf))
            if c == BALL:
                balls.append(((x1, y1, x2, y2), conf))
        (src / "labels" / (p.stem + ".txt")).write_text("\n".join(lines) + ("\n" if lines else ""))
        preds[i] = balls
        drawn[p] = boxes
    flags = flag_frames(preds)
    priority = set()
    if a.fused and a.view:
        priority = priority_from_fused(json.loads(a.fused.read_text()), a.view, a.fps)
        flags += [{"frame": i, "reason": "other_view_saw_ball"} for i in sorted(priority) if i < len(frames)]
    flags.sort(key=lambda f: (f["reason"] != "other_view_saw_ball", f["frame"]))
    review = REPO / "out" / "spark" / f"review_{a.clip.stem}"
    review.mkdir(parents=True, exist_ok=True)
    for n, i in enumerate(range(0, len(frames), 16)):
        _sheet(frames[i:i + 16], drawn, review / f"sheet_{n:03d}.jpg")
    (src / "source.json").write_text(json.dumps({"name": a.clip.stem, "origin": "pseudo+review",
                                                 "weights": str(a.weights), "images": len(frames)}, indent=1))
    (review / "review.json").write_text(json.dumps({
        "clip": str(a.clip), "source_dir": str(src), "frames": len(frames), "fps": a.fps,
        "flagged": [{**f, "image": frames[f["frame"]].name} for f in flags if f["frame"] < len(frames)],
        "classes": classes}, indent=1))
    print(f"{len(frames)} frames -> {src}; {len(flags)} flagged for review in {review / 'review.json'}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_pseudolabel.py -q`
Expected: `3 passed`

- [ ] **Step 5: Smoke the CLI on any short clip available locally**

Run: `ml/.venv/bin/python -m ml.spark.pseudolabel --clip ml/data/open_video/CakjTJXij7g.mp4 --weights models/followcam_smoke.pt --fps 0.2 --imgsz 640 --device cpu`
Expected: prints `N frames -> data/spark/sources/CakjTJXij7g; K flagged ...`, sheets under `out/spark/review_CakjTJXij7g/`. Then delete that source so it does not pollute the dataset: `rm -rf data/spark/sources/CakjTJXij7g` (the bootstrap set already has this video, and it would leak between train and val by name).

- [ ] **Step 6: Commit**

```bash
git add ml/spark/pseudolabel.py ml/spark/tests/test_pseudolabel.py
git commit -m "spark: pseudo-label loop with review sheet and cross-view priority

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 7: Core ML export (Mac only)

**Files:**
- Create: `ml/spark/export_coreml.py`, `ml/spark/tests/test_export_coreml.py`

**Interfaces:**
- Produces: `export_coreml.require_macos(platform_name: str) -> None` (raises `SystemExit` off macOS), `export_coreml.output_paths(weights: Path, sizes: list[int]) -> list[Path]`; files `models/<weights-stem>_<size>.mlpackage`.

- [ ] **Step 1: Write the failing tests**

```python
# ml/spark/tests/test_export_coreml.py
from pathlib import Path

import pytest

from ml.spark.export_coreml import output_paths, require_macos


def test_require_macos_rejects_linux():
    with pytest.raises(SystemExit, match="macOS"):
        require_macos("Linux")
    require_macos("Darwin")


def test_output_paths_follow_naming():
    assert output_paths(Path("models/followcam_s_1280.pt"), [640, 960]) == [
        Path("models/followcam_s_1280_640.mlpackage"), Path("models/followcam_s_1280_960.mlpackage")]
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_export_coreml.py -q`
Expected: FAIL with `ModuleNotFoundError`

- [ ] **Step 3: Implement `export_coreml.py`**

```python
# ml/spark/export_coreml.py
"""Export a trained .pt to Core ML at 640 and 960 input. Runs on macOS only.

  ml/.venv/bin/python -m ml.spark.export_coreml --weights models/followcam_s_1280.pt [--sizes 640 960]
"""
from __future__ import annotations

import argparse
import platform
import shutil
from pathlib import Path


def require_macos(platform_name: str = platform.system()) -> None:
    if platform_name != "Darwin":
        raise SystemExit("Core ML export needs macOS (coremltools compiles the model); run this on the Mac")


def output_paths(weights: Path, sizes: list[int]) -> list[Path]:
    return [weights.with_name(f"{weights.stem}_{s}.mlpackage") for s in sizes]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", type=Path, required=True)
    ap.add_argument("--sizes", type=int, nargs="+", default=[640, 960])
    a = ap.parse_args()
    require_macos()
    from ultralytics import YOLO

    for size, out in zip(a.sizes, output_paths(a.weights, a.sizes)):
        exported = Path(YOLO(str(a.weights)).export(format="coreml", imgsz=size, nms=True, half=True))
        if out.exists():
            shutil.rmtree(out)
        shutil.move(str(exported), out)
        mb = sum(p.stat().st_size for p in out.rglob("*") if p.is_file()) / 1e6
        print(f"{out}  {mb:.1f} MB")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `ml/.venv/bin/python -m pytest ml/spark/tests/test_export_coreml.py -q`
Expected: `2 passed`

- [ ] **Step 5: Export the smoke weights to prove the toolchain**

Run: `ml/.venv/bin/python -m pip install coremltools && make spark-export WEIGHTS=models/followcam_smoke.pt`
Expected: two lines `models/followcam_smoke_640.mlpackage  X MB` and `..._960.mlpackage  Y MB`. If coremltools refuses Python 3.14, create `ml/.venv-export` with Python 3.12 (`brew install python@3.12; python3.12 -m venv ml/.venv-export; ml/.venv-export/bin/pip install ultralytics coremltools`) and run `make spark-export MLPY=ml/.venv-export/bin/python`; record which one worked in `SETUP.md`.

- [ ] **Step 6: Commit**

```bash
git add ml/spark/export_coreml.py ml/spark/tests/test_export_coreml.py ml/spark/SETUP.md
git commit -m "spark: Core ML export at 640 and 960 (Mac only)

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

### Task 8: Round one on the Spark and the results record

This task is operational: it runs the committed tooling on the real hardware and records the numbers. Nothing new is coded except the results section.

**Files:**
- Modify: `ml/spark/sources.yaml` (real Roboflow entries), `ml/spark/configs/{s,m}_1280.yaml` (pin `batch`), `ml/spark/SETUP.md` (pin the container tag, hostname), `docs/RESULTS.md` (new section)

- [ ] **Step 1: Pick the public sources**

Open `https://universe.roboflow.com/search?q=basketball%20ball%20hoop%20referee`, choose two or three datasets whose class list includes ball, hoop and referee (or aliases in `classes.yaml`), preferring gym and sideline angles. Fill `workspace`, `project`, `version` for `rf_basketball_a` and `rf_basketball_b` in `sources.yaml` (add a third entry `rf_basketball_c` if useful). Put `ROBOFLOW_API_KEY=...` in `.env`.

- [ ] **Step 2: Build the dataset on the Mac and sync**

```bash
set -a; . .env; set +a
make spark-data
cat data/spark/dataset/manifest.json | head -40
rsync -av --progress data/spark/ followcam@spark.local:~/followcam/data/spark/
```
Expected: manifest shows every source with nonzero `ball` and `hoop` boxes, val nonempty.

- [ ] **Step 3: Smoke on the Spark**

In the container (see `SETUP.md`): `make spark-smoke MLPY=python DEVICE=0`
Expected: `True GB10` from the torch check, one epoch completes, `models/followcam_smoke.pt` written on the Spark. Pin the container tag in `SETUP.md`.

- [ ] **Step 4: Train s and m**

```bash
make spark-train MLPY=python DEVICE=0 CONFIG=s_1280 2>&1 | tee runs/spark/s_1280.log
make spark-train MLPY=python DEVICE=0 CONFIG=m_1280 2>&1 | tee runs/spark/m_1280.log
```
Expected: each prints the per-class table at the end. Write the batch size ultralytics chose into the two configs.

- [ ] **Step 5: Ball reports and sync back**

```bash
python -m ml.spark.eval_ball --weights models/followcam_s_1280.pt --labeled data/spark/dataset --split val --device 0
python -m ml.spark.eval_ball --weights models/followcam_m_1280.pt --labeled data/spark/dataset --split val --device 0
# on the Mac:
rsync -av followcam@spark.local:~/followcam/models/ models/
rsync -av followcam@spark.local:~/followcam/runs/spark/ runs/spark/
rsync -av followcam@spark.local:~/followcam/out/spark/ out/spark/
```

- [ ] **Step 6: Record results**

Append to `docs/RESULTS.md`:

```markdown
## Spark round one (date)

Dataset: `data/spark/dataset/manifest.json` — <train N> train / <val M> val images from
<sources list>, classes 0 player 1 ball 2 hoop 3 referee, split by source video.
Hardware: DGX Spark, container `nvcr.io/nvidia/pytorch:<TAG>`, ultralytics 8.4.

| Run | imgsz | batch | epochs run | best epoch | mAP50 all | ball P | ball R | ball mAP50 | time |
|---|---|---|---|---|---|---|---|---|---|
| s_1280 | 1280 | | | | | | | | |
| m_1280 | 1280 | | | | | | | | |

Ball report (`out/spark/eval_followcam_{s,m}_1280_dataset_val.json`), IoU 0.3:

| Run | conf | recall | FP / frame |
|---|---|---|---|
| s_1280 | 0.25 / 0.35 / 0.45 | | |
| m_1280 | 0.25 / 0.35 / 0.45 | | |

Chosen for round two: <s or m>, because <one sentence>.
No rig clip existed yet, so the longest-gap metric is pending the first
multi-view session (plan 2026-09-16-multiview-sync).
```

Fill every cell from the logs and JSON. No cell stays blank.

- [ ] **Step 7: Commit**

```bash
git add ml/spark/sources.yaml ml/spark/configs ml/spark/SETUP.md docs/RESULTS.md
git commit -m "spark: round one results, pinned batch sizes and container tag

Co-Authored-By: WOZCODE <contact@withwoz.com>"
```

---

## Round two (after the first multi-view session exists)

Not a coded task; the loop to run once `out/<session>/fused_ball.json` and per-view clips exist:

1. Copy the newest rig clip to `data/clips/` and pseudo-label it: `python -m ml.spark.pseudolabel --clip data/clips/<view>.mp4 --weights models/followcam_<chosen>.pt --fused out/<session>/fused_ball.json --view <view_id>`.
2. Review `out/spark/review_<view>/review.json`, fix the flagged labels in `data/spark/sources/<view>/labels/`.
3. Hold out the newest session's primary view: `make spark-data HOLDOUT="<view>/<view_key>"`, rsync, retrain s and m.
4. Gate: `make spark-eval WEIGHTS=models/followcam_<chosen>.pt CLIP=data/clips/<holdout view>.mp4`, `longest_gap_s` under 2.0.
5. Court test with the rig. Record both in `docs/RESULTS.md` under "Spark round two".
6. `make spark-export WEIGHTS=models/followcam_<chosen>.pt` on the Mac. Hand `models/*.mlpackage` to the iOS plan.

## Self-review notes

- Spec A1 layout: Tasks 1 to 7 create every listed file; `SETUP.md` in Task 4, `requirements.txt` in Task 1.
- Spec A3 sources and split: Task 2 and Task 3, holdout in Task 3, manifest in Task 3.
- Spec A4 recipe: configs in Task 4 with the exact hyperparameters.
- Spec A5 evaluation: Task 5 (both layers; the ultralytics val table is printed by `train.py`).
- Spec A6 pseudo-label loop and the cross-view rule: Task 6.
- Spec A7 export: Task 7.
- Acceptance and round two: Task 8 and the round-two section.
