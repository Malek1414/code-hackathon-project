# FollowCam iOS app

SwiftUI camera app that runs ON the phone mounted in the rig: records the
game, tracks a tapped subject with Vision, and steers the servo by sending
pan angles to the laptop bridge, which forwards them to the Arduino.

```
phone (this app) --ws://laptop:8765--> software/pan_bridge.py --serial--> Arduino (servo_pan.ino)
```

## Run it (needs Xcode + an iPhone)

1. Open `app/FollowCam.xcodeproj` (already generated; verified with an
   unsigned `xcodebuild` — BUILD SUCCEEDED).
2. Target FollowCam → Signing & Capabilities → pick your personal team.
3. Plug in the phone, hit Run. Camera/Photos/network permission strings are
   already configured.

(To regenerate the project after adding files: `brew install xcodegen`,
then `xcodegen generate` in `app/`. The generated `.xcodeproj` is committed;
`app/build/` is not — it is Xcode's scratch space and is gitignored.)

Verify without a phone or a signing identity:

```bash
cd app
xcodebuild -scheme FollowCam -destination 'generic/platform=iOS' \
  CODE_SIGNING_ALLOWED=NO build                              # compiles the app
xcodebuild test -scheme FollowCam \
  -destination 'platform=iOS Simulator,name=iPhone 17 Pro'  # pan control law + HR packet parser
```

## Use it (court test flow)

1. Laptop: `python3 software/pan_bridge.py --port /dev/cu.usbmodem*`
   (or `--dry-run` to test without the Arduino). Phone and laptop on the
   same Wi-Fi / hotspot.
2. App: enter the laptop's IP → **link rig** (turns green).
3. Mount the phone in the rig, **tap the subject** on screen (ball or the
   person carrying it) — orange box appears, servo starts following.
4. Big red button records; clips land in Photos (these are tier-3 training
   data — see `docs/ML_DATA_PLAN.md`). The first recording asks for the
   microphone; allow it and the clip has sound, deny it and the clip is
   silent. The summary card says whether the save to Photos succeeded.
5. If the rig pans the wrong way, flip the **inv** toggle. The laptop
   address and the toggle are remembered between launches.

The RIG lamp turns green only once the socket to the bridge is actually
open, and the link reconnects by itself (backoff up to 8 s) until you tap
Disconnect. HR: tap to search for a strap, tap again to stop; a strap that
drops out clears the reading instead of freezing on its last value.

Angles are clamped to 40–140° in both the app and the firmware; the
firmware's slew limiter smooths whatever the app sends.

## Today's demo note

The laptop OpenCV path (`software/ball_tracker.py` → serial) remains the
primary demo pipeline — it has no signing/Wi-Fi dependencies. The app is the
product-architecture demo (no laptop in the vision loop) and the recorder
for the court test. Don't let app debugging eat integration time; the 14:30
freeze applies.
