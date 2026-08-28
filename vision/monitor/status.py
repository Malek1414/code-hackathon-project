"""Collect the pipeline state from disk. Every section degrades to a small
dict with ``ok: False`` (or ``error``) instead of raising, so the board never
goes down because one artifact is half written."""
from __future__ import annotations

import csv
import json
import os
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
FRAMES_DIR = ROOT / "data" / "frames"
DATASET_DIR = ROOT / "data" / "dataset"
LABELS_DIR = DATASET_DIR / "labels"
RUNS_DIR = ROOT / "runs"
OUT_DIR = ROOT / "out"

CLASSES = {0: "player", 1: "ball", 2: "hoop", 3: "referee"}
IMG_SUFFIXES = {".jpg", ".jpeg", ".png"}
TAIL_LINES = 200
LOG_LINES = 14


def _safe(fn):
    def wrapped(*args, **kwargs):
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - the board must never crash
            return {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    wrapped.__name__ = fn.__name__
    return wrapped


def _stat(path: Path):
    try:
        st = path.stat()
        return st.st_mtime, st.st_size
    except OSError:
        return None


def _hhmm(ts: float | None) -> str:
    return time.strftime("%H:%M:%S", time.localtime(ts)) if ts else ""


# ---------------------------------------------------------------- LABEL ----

_label_counts: dict[str, tuple[float, int, dict[int, int]]] = {}


def label_files() -> list[Path]:
    if not LABELS_DIR.is_dir():
        return []
    files: list[Path] = []
    for split in LABELS_DIR.iterdir():
        if split.is_dir():
            files.extend(p for p in split.iterdir() if p.suffix == ".txt")
    return files


def _count_boxes(path: Path) -> dict[int, int]:
    st = _stat(path)
    if st is None:
        return {}
    cached = _label_counts.get(str(path))
    if cached and cached[0] == st[0] and cached[1] == st[1]:
        return cached[2]
    counts: dict[int, int] = {}
    with open(path, encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.split()
            if not parts:
                continue
            try:
                cls = int(float(parts[0]))
            except ValueError:
                continue
            counts[cls] = counts.get(cls, 0) + 1
    _label_counts[str(path)] = (st[0], st[1], counts)
    return counts


def newest_label() -> Path | None:
    files = label_files()
    if not files:
        return None
    return max(files, key=lambda p: p.stat().st_mtime if p.exists() else 0)


def image_for_label(txt: Path) -> Path | None:
    """YOLO layout: labels/<split>/x.txt <-> images/<split>/x.jpg, else data/frames."""
    split = txt.parent.name
    candidates = [DATASET_DIR / "images" / split / (txt.stem + s) for s in IMG_SUFFIXES]
    candidates += [FRAMES_DIR / (txt.stem + s) for s in IMG_SUFFIXES]
    for c in candidates:
        if c.exists():
            return c
    return None


def newest_results_csv() -> Path | None:
    if not RUNS_DIR.is_dir():
        return None
    found = [p for p in RUNS_DIR.rglob("results.csv") if p.is_file()]
    if not found:
        return None
    return max(found, key=lambda p: p.stat().st_mtime)


def read_results_csv(path: Path) -> dict:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if not rows:
        return {"epochs": 0}
    keys = [k.strip() for k in rows[0].keys() if k]
    cleaned = [{k.strip(): (v or "").strip() for k, v in r.items() if k} for r in rows]

    def col(name):
        if name not in keys:
            return None
        vals = []
        for r in cleaned:
            try:
                vals.append(float(r[name]))
            except ValueError:
                vals.append(None)
        return vals

    map_cols = [k for k in keys if "mAP50" in k and "mAP50-95" not in k]
    series = {}
    box = col("train/box_loss")
    if box:
        series["box_loss"] = box
    for k in map_cols:
        short = k.replace("metrics/", "").replace("(B)", "")
        series[short] = col(k)
    epochs = col("epoch") or list(range(1, len(cleaned) + 1))
    last = cleaned[-1]
    return {
        "epochs": len(cleaned),
        "epoch_values": epochs,
        "series": series,
        "last": {k: last.get(k) for k in keys if k in ("epoch", "train/box_loss", "metrics/mAP50(B)", "metrics/mAP50-95(B)")},
        "run": str(path.parent.relative_to(ROOT)),
        "mtime": _hhmm(path.stat().st_mtime),
    }


@_safe
def label_section() -> dict:
    frames = 0
    if FRAMES_DIR.is_dir():
        frames = sum(1 for p in os.scandir(FRAMES_DIR) if p.is_file() and Path(p.name).suffix.lower() in IMG_SUFFIXES)
    files = label_files()
    per_class: dict[int, int] = {}
    per_split: dict[str, int] = {}
    for f in files:
        per_split[f.parent.name] = per_split.get(f.parent.name, 0) + 1
        for cls, n in _count_boxes(f).items():
            per_class[cls] = per_class.get(cls, 0) + n
    newest = newest_label()
    res_csv = newest_results_csv()
    training = read_results_csv(res_csv) if res_csv else None
    best = ROOT / "models" / "best.pt"
    best_st = _stat(best)
    return {
        "ok": True,
        "frames": frames,
        "labels": len(files),
        "per_split": per_split,
        "per_class": {CLASSES.get(c, str(c)): n for c, n in sorted(per_class.items())},
        "boxes_total": sum(per_class.values()),
        "newest": newest.name if newest else None,
        "newest_time": _hhmm(newest.stat().st_mtime) if newest else "",
        "training": training,
        "best_pt": {"exists": best_st is not None, "time": _hhmm(best_st[0]) if best_st else "", "mb": round(best_st[1] / 1e6, 1) if best_st else 0},
    }


# ---------------------------------------------------------------- TRACK ----


class _LineCounter:
    """Counts complete lines incrementally; only new bytes are read."""

    def __init__(self, path: Path):
        self.path = path
        self.offset = 0
        self.lines = 0
        self.size = 0

    def count(self) -> int:
        st = _stat(self.path)
        if st is None:
            self.offset = self.lines = self.size = 0
            return 0
        size = st[1]
        if size < self.size:
            self.offset = self.lines = 0
        self.size = size
        if size > self.offset:
            with open(self.path, "rb") as fh:
                fh.seek(self.offset)
                chunk = fh.read(size - self.offset)
            self.lines += chunk.count(b"\n")
            self.offset = size
        return self.lines


_tracks_counter = _LineCounter(OUT_DIR / "tracks.jsonl")


def tail_lines(path: Path, n: int, max_bytes: int = 800_000) -> list[str]:
    st = _stat(path)
    if st is None or st[1] == 0:
        return []
    size = st[1]
    start = max(0, size - max_bytes)
    with open(path, "rb") as fh:
        fh.seek(start)
        data = fh.read(size - start)
    text = data.decode("utf-8", errors="replace")
    lines = text.split("\n")
    if start > 0:
        lines = lines[1:]  # first line is probably cut in the middle
    if not text.endswith("\n"):
        lines = lines[:-1]  # last line still being written
    return [ln for ln in lines if ln.strip()][-n:]


def load_json(path: Path):
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


@_safe
def track_section() -> dict:
    path = OUT_DIR / "tracks.jsonl"
    st = _stat(path)
    if st is None:
        return {"ok": False}
    total = _tracks_counter.count()
    recent = []
    for ln in tail_lines(path, TAIL_LINES):
        try:
            recent.append(json.loads(ln))
        except ValueError:
            continue
    n = len(recent)
    ball_hits = sum(1 for r in recent if r.get("ball"))
    team_boxes = {"0": 0, "1": 0, "-1": 0}
    team_ids: dict[str, set] = {"0": set(), "1": set(), "-1": set()}
    hoops = 0
    for r in recent:
        for p in r.get("players") or []:
            t = str(p.get("team", -1))
            if t not in team_boxes:
                t = "-1"
            team_boxes[t] += 1
            team_ids[t].add(p.get("id"))
        hoops += len(r.get("hoops") or [])
    last = recent[-1] if recent else {}
    meta = None
    meta_path = OUT_DIR / "tracks_meta.json"
    if meta_path.exists():
        try:
            meta = load_json(meta_path)
        except ValueError:
            meta = None
    progress = None
    if meta and meta.get("last_frame") and last.get("frame") is not None:
        progress = round(100.0 * last["frame"] / max(1, meta["last_frame"]), 1)
    overlay = _stat(OUT_DIR / "overlay.mp4")
    return {
        "ok": True,
        "lines": total,
        "window": n,
        "ball_hits": ball_hits,
        "ball_rate": round(ball_hits / n, 3) if n else None,
        "team_boxes": team_boxes,
        "team_ids": {k: len(v) for k, v in team_ids.items()},
        "players_per_frame": round(sum(len(r.get("players") or []) for r in recent) / n, 1) if n else None,
        "hoops_per_frame": round(hoops / n, 2) if n else None,
        "last_frame": last.get("frame"),
        "last_t": last.get("t"),
        "progress_pct": progress,
        "clip": (meta or {}).get("clip"),
        "weights": (meta or {}).get("weights"),
        "file_time": _hhmm(st[0]),
        "file_mb": round(st[1] / 1e6, 2),
        "overlay": {"exists": overlay is not None, "mb": round(overlay[1] / 1e6, 1) if overlay else 0, "time": _hhmm(overlay[0]) if overlay else ""},
    }


# ---------------------------------------------------------------- STATS ----


@_safe
def stats_section() -> dict:
    out: dict = {"ok": False, "events": None, "stats": None}
    ev_path = OUT_DIR / "events.json"
    if ev_path.exists():
        ev = load_json(ev_path)
        shots = ev.get("shots") or []
        made = sum(1 for s in shots if s.get("made"))
        per_team: dict[str, dict[str, int]] = {}
        for s in shots:
            t = str(s.get("team", -1))
            d = per_team.setdefault(t, {"fga": 0, "fgm": 0})
            d["fga"] += 1
            d["fgm"] += 1 if s.get("made") else 0
        out["events"] = {
            "clip": ev.get("clip"),
            "fps": ev.get("fps"),
            "count": len(shots),
            "made": made,
            "per_team": per_team,
            "shots": [
                {"t": s.get("t"), "frame": s.get("frame"), "team": s.get("team"), "player_id": s.get("player_id"), "made": s.get("made"), "unconfirmed": s.get("unconfirmed")}
                for s in shots[-30:]
            ][::-1],
            "time": _hhmm(ev_path.stat().st_mtime),
        }
        out["ok"] = True
    st_path = OUT_DIR / "stats.json"
    if st_path.exists():
        data = load_json(st_path)
        players = data.get("players") or []
        players = sorted(players, key=lambda p: (-(p.get("fga") or 0), p.get("id") or 0))[:24]
        out["stats"] = {"players": players, "teams": data.get("teams") or [], "time": _hhmm(st_path.stat().st_mtime)}
        out["ok"] = True
    return out


# ---------------------------------------------------------------- COURT ----


@_safe
def court_section() -> dict:
    calib = OUT_DIR / "court_calib.json"
    preview = OUT_DIR / "court_propagate_preview.mp4"
    minimap = OUT_DIR / "minimap.mp4"
    dashboard = OUT_DIR / "dashboard.html"
    res: dict = {
        "ok": calib.exists(),
        "calib": None,
        "preview": _stat(preview) is not None,
        "minimap": _stat(minimap) is not None,
        "dashboard": _stat(dashboard) is not None,
    }
    if calib.exists():
        data = load_json(calib)
        frames = data.get("frames")
        if isinstance(frames, dict) and frames:
            keyframes = len(frames)
        elif data.get("H_px_to_m"):
            keyframes = 1
        else:
            keyframes = 0
        res["calib"] = {
            "clip": data.get("clip"),
            "frame": data.get("frame"),
            "points": len(data.get("points") or []),
            "keyframes": keyframes,
            "reproj_err_px": data.get("reproj_err_px"),
            "court_m": data.get("court_m"),
            "time": _hhmm(calib.stat().st_mtime),
        }
    return res


# ----------------------------------------------------------------- LOGS ----


@_safe
def logs_section() -> dict:
    if not OUT_DIR.is_dir():
        return {"ok": False, "logs": []}
    logs = sorted((p for p in OUT_DIR.glob("*.log") if p.is_file()), key=lambda p: p.stat().st_mtime, reverse=True)
    items = []
    for p in logs[:8]:
        st = p.stat()
        items.append({"name": p.name, "time": _hhmm(st.st_mtime), "bytes": st.st_size, "lines": tail_lines(p, LOG_LINES, max_bytes=16_000)})
    return {"ok": bool(items), "logs": items}


# ------------------------------------------------------------- COLLECT ----

_lock = threading.Lock()
_memo: tuple[float, dict] | None = None
MEMO_SECONDS = 1.0


def collect() -> dict:
    """Full snapshot. Memoised for one second so several tabs stay cheap."""
    global _memo
    with _lock:
        now = time.time()
        if _memo and now - _memo[0] < MEMO_SECONDS:
            return _memo[1]
        from vision.monitor import images  # local import: cv2 only when needed

        snap = {
            "now": time.strftime("%H:%M:%S"),
            "root": str(ROOT),
            "label": label_section(),
            "track": track_section(),
            "stats": stats_section(),
            "court": court_section(),
            "logs": logs_section(),
            "images": images.tokens(),
        }
        _memo = (now, snap)
        return snap
