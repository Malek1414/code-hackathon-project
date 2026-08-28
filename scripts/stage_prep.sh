#!/bin/zsh
# Big Ball Baller stage prep: free the GPU and CPU for the live demo, print the commands.
# Run at 15:50: ./scripts/stage_prep.sh
cd "$(dirname "$0")/.." || exit 1
echo "stopping model jobs and watchers (monitor board stays)"
pkill -f "vision/track/run.py" 2>/dev/null
pkill -f "vision/track/compare.py" 2>/dev/null
pkill -f "vision/label/train" 2>/dev/null
pkill -f "vision/label/autolabel" 2>/dev/null
pkill -f "vision.numbers.watch" 2>/dev/null
pkill -f "vision.qa.watch" 2>/dev/null
pkill -f "vision/court/propagate.py" 2>/dev/null
pkill -f "vision/qa/ball_check" 2>/dev/null
pkill -f "imageio_ffmpeg" 2>/dev/null
touch out/.stop_autopush
sleep 2
echo "remaining .venv python processes:"
ps -axo pid,%cpu,command | grep "\.venv/bin/python" | grep -v grep | cut -c1-100
echo
echo "cameras:"
.venv/bin/python -m vision.live.live --list-sources 2>/dev/null | tail -6
echo
echo "serial ports:"; ls /dev/cu.usb* 2>/dev/null || echo "  none (Arduino not connected)"
echo
echo "LIVE (phone = --source 1 on this Mac, servo = --serial <port>):"
echo "  .venv/bin/python -m vision.live.live --source 1 --team-a 'Team A' --team-b 'Team B' --serial /dev/cu.usbXXXX"
echo "REPLAY fallback (no GPU):"
echo "  .venv/bin/python -m vision.live.live --source data/clips/dev60.mp4 --realtime --loop --replay out/dev60_v5/tracks.jsonl"
echo "first action: click the window, press 1 (Team A +2), then z."
