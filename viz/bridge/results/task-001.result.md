# Task 001 result

- Output: `fx/task-001/court_panel.mov`
- Toolchain: Python 3 + Pillow 12.2.0 for deterministic 2× RGBA frame rendering; FFmpeg 8.1.1/libx264 for final assembly.

## Commands used

```bash
python3 fx/task-001/render.py
ffmpeg -y -framerate 30 -start_number 0 -i fx/task-001/frames/%05d.png -frames:v 240 -an -c:v libx264 -preset slow -crf 14 -pix_fmt yuv420p -r 30 -movflags +faststart fx/task-001/court_panel.mov
ffprobe -v error -count_frames -select_streams v:0 -show_entries stream=codec_name,width,height,pix_fmt,r_frame_rate,avg_frame_rate,nb_read_frames:format=duration -of json fx/task-001/court_panel.mov
ffmpeg -v error -i fx/task-001/court_panel.mov -f null -
```

## Verification

- Codec: H.264 (`libx264`)
- Dimensions: 960×1080
- Duration: 8.000000 seconds
- Frame rate: 30/1 fps
- Decoded frame count: 240
- Pixel format: yuv420p (opaque, as requested)
- Full-stream decode check: passed

## Caveats or deviations

None.
