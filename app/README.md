# FollowCam iOS app

SwiftUI camera app that runs ON the phone mounted in the rig: records the
game, tracks a tapped subject with Vision, and steers the servo by sending
pan angles to the laptop bridge, which forwards them to the Arduino.

```
phone (this app) --ws://laptop:8765--> software/pan_bridge.py --serial--> Arduino (servo_pan.ino)
```

## Run it (needs Xcode + an iPhone)

1. Xcode → New Project → iOS App, name **FollowCam**, interface SwiftUI.
2. Delete the template `ContentView.swift`/`FollowCamApp.swift`; drag the five
   files from `FollowCamApp/` into the project.
3. Target → Info: add
   - `NSCameraUsageDescription` — "Films the game and tracks the ball."
   - `NSPhotoLibraryAddUsageDescription` — "Saves recordings."
   - `NSLocalNetworkUsageDescription` — "Steers the FollowCam rig."
4. Signing: your personal team. Build to the phone.

## Use it (court test flow)

1. Laptop: `python3 software/pan_bridge.py --port /dev/cu.usbmodem*`
   (or `--dry-run` to test without the Arduino). Phone and laptop on the
   same Wi-Fi / hotspot.
2. App: enter the laptop's IP → **link rig** (turns green).
3. Mount the phone in the rig, **tap the subject** on screen (ball or the
   person carrying it) — orange box appears, servo starts following.
4. Big red button records; clips land in Photos (these are tier-3 training
   data — see `docs/ML_DATA_PLAN.md`).
5. If the rig pans the wrong way, flip the **inv** toggle.

Angles are clamped to 40–140° in both the app and the firmware; the
firmware's slew limiter smooths whatever the app sends.

## Today's demo note

The laptop OpenCV path (`software/ball_tracker.py` → serial) remains the
primary demo pipeline — it has no signing/Wi-Fi dependencies. The app is the
product-architecture demo (no laptop in the vision loop) and the recorder
for the court test. Don't let app debugging eat integration time; the 14:30
freeze applies.
