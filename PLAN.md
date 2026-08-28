# CODE Hackathon Battle Plan — Fri Aug 28, 10:00–17:00, Donaustraße 44

Goal: **win**. The winning shape: a robot that visibly follows a ball on stage + a pitch that scales it into the position-data platform.

## Tonight (Thu Aug 27)

| Time | Do |
|---|---|
| 21:00–21:20 | Photograph tripod + handle (see `cad/MEASUREMENTS.md`), take the 5 measurements, drop everything in `tripod-photos/`, send to Claude |
| 21:20–22:15 | CAD session with Claude: finalize `cad/handle-clamp.scad` with real dimensions → export both STLs, copy onto a USB stick AND email to self |
| 22:15–23:00 | Software spine: run `software/ball_tracker.py` on laptop webcam with any bright object; confirm tracking box + angle output works. Do NOT polish |
| 23:00–23:20 | Hardware hunt: confirm what the makerspace has (message lab staff / student Slack tonight): hobby servo (MG996R/SG90 class), Arduino or ESP32, jumper wires, USB cable, M3 bolts, zip ties. Anything missing → plan to buy at Conrad/Obi or ask in the 10:00 crowd |
| 23:20–23:40 | Pack bag (list below) + skim `pitch/pitch.md` once |
| 23:40–00:00 | Wind down. **Sleep by 00:00 — non-negotiable. A sharp brain beats 2 more hours of prep.** |

**Packing list:** tripod + pan handle, phone + phone tripod mount, laptop + chargers (laptop AND phone), USB-A/C cables + any USB-serial adapter, ball (or bright substitute), roll of tape + zip ties if you have them, USB stick with STLs, water/snacks.

## Tomorrow (Fri Aug 28)

| Time | Do |
|---|---|
| 07:45 | Wake. Real breakfast |
| 08:45 | Leave for Donaustraße 44, Neukölln |
| 09:30 | Arrive early. **Straight to the makerspace**: confirm printer access during the event, get a printer reserved/queued, confirm filament, load your STLs into the slicer, check estimated print time (target <90 min; if longer, cut infill to 20% / drop the bracket and zip-tie the servo) |
| 10:00–10:30 | Kickoff. Scan the room for 2–3 teammates while listening |
| 10:30–11:00 | **Pick your problem / team formation.** Recruit with one sentence: "I'm building a robot cameraman that follows the ball — hardware is designed, I need a vision person, a pitch/design person, and a builder. We demo live." Take the first 2–3 solid yeses |
| 11:00–11:15 | **START THE PRINT.** This is the critical path — clamp first, bracket second. Then ideate phase with the team on top of `IDEA.md` (don't reopen the core decision; refine the pitch) |
| 11:15–12:30 | Parallel build: (1) you/builder: wire servo + microcontroller, flash `servo_pan.ino`, servo sweeps from serial commands; (2) vision person: tune `ball_tracker.py` HSV on the actual ball in the actual room light; (3) pitch person: deck skeleton + starts filming b-roll of the build |
| 12:30–13:00 | Eat while working. Laptop→serial→servo loop closed (servo follows the ball in the air, no tripod yet) |
| 13:00–13:30 | Print done → fit clamp on handle, mount bracket + servo on tripod leg, connect pushrod (wire/rod from makerspace). If fit fails: file/dremel it or fall back to zip ties (Fallback B). Do not reprint unless <30 min fix |
| 13:30–14:30 | **End-to-end integration**: ball moves → camera pans. Tune deadband + smoothing so it doesn't oscillate. Define the demo choreography: one person walks the ball left–right at 5m distance |
| 14:30–15:15 | **Film the video pitch** (script in `pitch/pitch.md`): 60–90s, phone-shot, the money shot is the rig following the ball with the phone screen visible. Cut it fast (CapCut). Submit-ready by 15:15 |
| 15:15–15:50 | Finish deck (5 slides max), rehearse pitch twice out loud, assign who says what, **submit everything before the deadline** — whatever it is, treat 16:00 as yours |
| 15:50–16:00 | Set up the stage demo: rig placed, ball ready, laptop screen mirrored if possible |
| 16:00–17:00 | Pitches. Open with the live demo, not the slides. Close with the platform story + roadmap. Win |

## Makerspace inventory (confirmed by photos, Thu ~21:30)

- **Prusa CORE One** — free as of 20:00 (prior job: 33g PLA in 1h25m ≈ our part size)
- **Prusa MK4S** — busy until ~22:30 (CODE HACK cube). House rule: no adhesives
  on bed; use a brim for adhesion issues
- **xTool S1 laser cutter** — supervised use Friday. Cuts plywood + dark acrylic;
  CANNOT cut clear acrylic (diode laser). Backup plan: fork arm from 2×4-6mm ply
- Filament: PLA + Rapid PETG · **M3 bolts** (full box) · nut heat inserts ·
  zip ties · glues · heat shrink · DuPont jumpers · WAGO · resistors/LEDs/
  transistors · piezos · grommets
- Servo + microcontroller + USB cable: **sourcing confirmed for Friday** (Malek).
  Backup if it falls through: Segor Electronics, Kaiserin-Augusta-Allee 94,
  Charlottenburg, opens 10:00, tel 030/4399843 — call first, send Sami.
  Spec: servo ≥8 kg·cm (MG996R ideal, SG90 spare), Uno/Nano clone + its USB
  cable, 4xAA holder + batteries for servo power (never the Arduino 5V pin).
  If it's an ESP32 instead: tell Claude, sketch needs the ESP32Servo library
- Print-tonight option: if tripod measurements land Thursday night, slice on
  PrusaSlicer → USB stick → CORE One overnight; Friday's 11:15 print rule
  becomes moot

## Non-negotiables / decision rules

1. **Print starts by 11:15** or you switch to Fallback B immediately at 13:00. The printer is the only resource you don't control.
2. **Demo beats features.** Any hour spent on a second feature is stolen from rehearsing the one moment that wins.
3. **Freeze integration at 14:30.** Whatever works at 14:30 is the demo; film it.
4. **The pitch is a story, not a spec**: problem (games nobody films) → magic moment (live follow) → price collapse (€2,000 → €20 retrofit) → platform (position data: broadcast, analytics, scorekeeping roadmap) → ask/next step.
5. If the room's judging vibe is community/values-driven (CODE: freedom, community, initiative, responsibility), lean on the accessibility angle: kids' games, viewers with disabilities, volunteer-run leagues.

## Risk table

| Risk | Mitigation |
|---|---|
| No printer slot / print queue full | Ask at 09:30, not 11:00. Fallback B (zip ties) keeps the robot alive |
| No servo/Arduino at makerspace | Confirm TONIGHT. Backup: buy SG90+Uno clone tomorrow 09:00 at a Conrad, or borrow from any robotics-inclined student in the 10:00 crowd |
| Phone-as-webcam fights you | macOS Continuity Camera usually just works in OpenCV; else use the laptop webcam for the demo and the phone as a prop |
| Tracking loses the ball in gym light | Fallback D: neon ball / tag sticker — and sell the sticker as product ("tag any ball") |
| Servo oscillates/overshoots | Deadband + EMA already in the script; if still bad, slow the walk in the demo choreography |
| Team wants to pivot the idea | You brought the hardware, the CAD, and the plan — offer them ownership of pitch/vision/roles instead of the core concept |
