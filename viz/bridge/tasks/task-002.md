---
task: 002
toolchain: python
inputs:
  - manual_assets/IMG_6614.jpg
  - manual_assets/IMG_6615.jpg
  - manual_assets/IMG_6616.jpg
  - manual_assets/IMG_6617.jpg
  - manual_assets/IMG_6619.jpg
  - manual_assets/IMG_6621.jpg
  - manual_assets/IMG_6603.jpg
  - manual_assets/IMG_6604.jpg
  - manual_assets/IMG_6603_annotated.jpg
  - manual_assets/followcam_assembled_poster.png
  - manual_assets/app_screen.png
output: fx/task-002/assembly_manual.mov
specs:
  width: 1920
  height: 1080
  duration: 55.0
  fps: 30
  alpha: false
---
# Goal
A LEGO-instruction-style assembly manual for the FollowCam rig: 11 static
pages (below), ALSO saved individually as fx/task-002/pages/page_00.png ...
page_10.png, then paced into the output video: each page held 5.0s
(11 x 5 = 55.0s exactly, 30fps, libx264 crf 14, yuv420p).

## LEGO grammar (every page identical layout)
- Canvas 1920x1080, background #FAFAF8.
- Top-left: step number in a filled square badge 110x110, #EB6414, white
  bold numeral (Helvetica, /System/Library/Fonts/Helvetica.ttc), except
  page 00 which shows "FC-01" small.
- Next to badge: step title, Helvetica bold ~44px, #1B1C1E; below it one
  short instruction line ~28px #6b6b6b. Text NEVER overlaps images.
- Right edge: a vertical "parts for this step" strip 320px wide, thin
  #1B1C1E border, listing part names with quantity "x1"/"x2" in
  monospace ~24px (SF Mono or Menlo). Omit the strip when empty.
- Main area: the listed photo(s), fitted large and centered, 12px white
  border + thin #1B1C1E outline. Two photos = side by side.
- Simple thick arrows (#EB6414, drawn with PIL polygons) where the step
  says so. No other decoration.
- Bottom-right footer: "FollowCam assembly - page N/10", mono 20px, #9a9a9a.

## Pages
00 TITLE/PARTS: photo IMG_6614.jpg large. Title "FollowCam - build it".
   Parts strip: printed column clamp x1, printed fork arm x1, metal-gear
   servo x1, servo horn + screws x1 set, Arduino Uno R3 x1, USB cable x1,
   M3 bolts x2, zip ties x6, tripod x1, phone x1.
01 "Flash the firmware": IMG_6615.jpg. Line: "Arduino IDE -> open
   software/servo_pan/servo_pan.ino -> board Arduino Uno -> Upload."
02 "Wire the servo": IMG_6615.jpg left + a drawn wiring diagram right
   (draw with PIL on white: three labeled colored lines orange/red/brown
   from a servo plug rectangle to three labeled pins "PIN 9" "5V" "GND"
   on an Uno rectangle). Line: "orange->9, red->5V, brown->GND."
03 "Bench test before mounting": IMG_6615.jpg. Line: "python3
   software/servo_test.py -> press s -> one smooth 40-140 sweep."
   Parts strip: none.
04 "Clamp onto the column": IMG_6619.jpg + IMG_6603.jpg side by side,
   arrow from clamp photo toward the column area of the tripod photo.
   Line: "Raise the column. Ring around it just below the head, shelf
   pointing the SAME side as the handle. 2x M3 through the slit bosses -
   snug." Parts strip: column clamp x1, M3 bolt x2.
05 "Seat the servo ON the rails": IMG_6619.jpg + IMG_6621.jpg. Line:
   "Big servo does not drop in the pocket - it sits ON TOP of the two
   rails, output shaft UP and toward the column. 2 zip ties through the
   ledge slots, cinch tight." Parts strip: metal-gear servo x1,
   zip ties x2. Add a small #EB6414 note chip: "pocket was sized for
   SG90 - rails carry it, ties hold it".
06 "Center, then fit the arm": IMG_6616.jpg. Line: "servo_test.py ->
   press c (90 deg). Only then press the fork arm onto the horn pointing
   straight at the pan handle; fix the horn screw." Parts strip: fork
   arm x1, horn + screw x1.
07 "Catch the handle shaft": IMG_6603_annotated.jpg. Line: "Drop the tall
   U-fork over the chrome shaft from below - the window is forgiving,
   +/-10mm everywhere." Arrow pointing at the chrome shaft in the photo.
08 "Friction + phone": IMG_6604.jpg. Line: "Loosen pan friction to a
   two-finger glide. Lock tilt HARD. Phone into the clamp, camera facing
   the court."
09 "Link and go": app_screen.png (phone frame, centered, not stretched).
   Line: "Hotspot on -> laptop: python3 software/pan_bridge.py --port
   /dev/cu.usbmodem* -> app: settings, laptop IP, RIG turns blue -> tap
   a player -> record."
10 "The goal": followcam_assembled_poster.png. Line: "Every clip you
   record trains the model. Film wide, vary distance and speed."

## Quality bar
- Draw at 2x (3840x2160) and LANCZOS-downscale each page to 1920x1080.
- Photos never distorted (fit, letterbox inside their frame).
- All text inside a 90% safe area; deterministic (no randomness).

## Feedback
