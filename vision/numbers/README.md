# NUMBERS: jersey numbers per track, tracks merged into players

    .venv/bin/python -m vision.numbers.read            # OCR + vote  -> out/numbers_reads.json, out/numbers_preview.jpg
    .venv/bin/python -m vision.numbers.merge           # players     -> out/identities.json (contract in docs/ORCHESTRATION.md)
    .venv/bin/python -m vision.numbers.watch           # both, again whenever out/tracks.jsonl changes
    .venv/bin/python -m pytest vision/numbers/tests -q

Reads `out/tracks.jsonl` (or `--tracks out/dev60/tracks.jsonl`; the clip path
comes from the `tracks_meta.json` next to it). EasyOCR runs on the CPU only,
digits allowlist, crops rescaled to 256 px longest side (0.8 s per crop).
`out/numbers_cache.json` caches every OCR read by clip + frame + bbox, so a
re-run after TRACK appended frames only reads the new crops.

Rules: up to 12 crops per track spread over its lifetime, torso = rows 15-60 %
of the box, reads of 1-2 digits with conf >= 0.4, vote = sum of confidences,
a number needs >= 2 reads and >= 60 % of the vote mass. Players = same
(team, number) with non-overlapping lifetimes; a second overlapping group gets
the key `A12~<id>`, tracks without a number keep `A?<id>` (`X?<id>` for team -1).
