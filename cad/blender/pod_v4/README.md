# FollowCam Pod v4 — print & assembly handoff

Direct-drive pan pod that **replaces the MACTREM tripod head**. Phone sits on a
rotor that turns on a Ø88 mm printed slew ring wrapped around the servo; the
Arduino lives in its own case clipped to a tripod leg.

Regenerate everything (STLs + renders) from the repo root:

```bash
blender --background --python cad/blender/pod_v4/build_v4.py
FOLLOWCAM_SKIP_RENDER=1 blender --background --python cad/blender/pod_v4/build_v4.py   # STLs only, ~20 s
```

Video walkthroughs:

| File | Spline offset | Matches `STL/`? |
|---|---|---|
| `FollowCam_Pod_v4.mov` | **9.85 mm** (measured) | **yes — use this one** |
| `FollowCam_Pod_v4_rev1.mov` | 10.25 mm (assumed) | no — 0.4 mm stale, kept for reference |

Both are 27.8 s / 1280×720. The 0.4 mm difference is not visible; rev1 is retained
only so the earlier review can be reproduced.

---

## Print list — `STL/`

Slicing is **not** done here. Numbers below are solid part volume, not filament use.

| # | File | Bed footprint (mm) | Vol | Orientation | Supports |
|---|---|---|---:|---|---|
| 01 | `01_stator_shell.stl` | 112 × 112 × 35 | 161 cm³ | as exported | none |
| 02 | `02_nut_retainer.stl` | 11.3 × 13 × 1.5 | 0.1 cm³ | as exported | none |
| 03 | `03_servo_carrier_MG996R.stl` | 96 × 96 × 5 | 30 cm³ | as exported | none |
| 04 | `04_rotor.stl` | 110 × 110 × 55 | 191 cm³ | **already inverted** — deck on the bed | none |
| 05 | `05_drive_dog_42.9-47.6.stl` | 37 × 13.5 × 8 | 1.1 cm³ | as exported | none |
| 05b | `05b_drive_dog_SHORT_42.9.stl` | 37 × 13.5 × 12.7 | 1.3 cm³ | as exported | none |
| 06 | `06_cradle_portrait.stl` | 173 × 87.5 × 22 | 67 cm³ | **already laid back-down** | yes — front lips |
| 07 | `07_cradle_landscape.stl` | 94 × 166.5 × 22 | 55 cm³ | **already laid back-down** | yes — front lips |
| 08 | `08_uno_tray.stl` | 96 × 80 × 30 | 85 cm³ | as exported — cavity opens up | none |
| 09 | `09_uno_lid.stl` | 96 × 80 × 5 | 34 cm³ | as exported | none |
| 10 | `10_leg_saddle.stl` | 96 × 66 × 36 | 155 cm³ | **already inverted** — flat face down, V opens up | none |
| 11 | `11_aa_bay.stl` | 68 × 41 × 21 | 19 cm³ | as exported — cavity opens up | none |

Pod is 01–07, leg case is 08–11. **Print only ONE of 05 / 05b — see below.**

### Which drive dog?

TowerPro's datasheet contradicts itself on how tall the MG996R is. The spec bullet
says 42.9 mm; the dimensioned drawing on the same sheet says **47.6 mm overall**
(36.6 mm case + output hub, 26.6 mm from the base to the mounting flange).

The pod is built to **47.6 mm**, because designing to the shorter figure and being
wrong means the servo jams into the rotor deck and the pod will not close. Being
wrong the other way just leaves air.

**Put a ruler on the actual servo, base to the top of the splined shaft:**

- **≈47.6 mm → print `05_drive_dog_42.9-47.6.stl`**
- **≈42.9 mm → print `05b_drive_dog_SHORT_42.9.stl`** (4.7 mm taller blade to reach up)

Both are ~1 cm³ and print in minutes. Print both if you want to decide at assembly.

**Every STL is pre-oriented and already sits on z = 0. Drop them on the plate as-is —
do not auto-orient.** Parts 04, 06 and 07 in particular are deliberately rotated:
printing the rotor deck-up would bridge a Ø97 mm hole, standing a cradle on its
tongue puts the one load-bearing joint directly across the layer lines, and the leg
saddle printed groove-down starts as two thin islands.

Only 06 and 07 need supports, and only under the four front retention lips.

### Filament

Baseline: **0.2 mm layers, 4 perimeters, 30–40 % infill.** Part 01 carries the whole
load into the tripod — do not drop its perimeter count.

Everything below is a recommendation, not a constraint: all twelve parts print in
plain PLA if that is what is on the shelf. It is what makes the pod read as a product
rather than as a print.

**Print it in PETG, in two colours.** PETG's ~80 °C glass transition survives a car
boot in summer, where PLA creeps at 55–60 °C — and 01 is under constant load from the
phone and cradle. Matching the atlas colours makes the printed pod match the renders:

| Parts | Filament |
|---|---|
| 01 stator, 04 rotor, 08 tray, 09 lid, 10 saddle | matte PETG, porcelain / bone |
| 03 carrier, 05 drive dog, 11 AA bay | matte PETG, vermilion |
| 06 / 07 cradles | either, but match the shells — these are what a hand touches |

Buy the **matte** grades. Gloss PETG highlights every layer line; matte diffuses them
and reads moulded. Cheapest upgrade on this list.

**If the pod lives outdoors, print ASA instead.** Tg ~100 °C, UV-stable so it will not
chalk on an outdoor court, and it leaves the bed with an eggshell finish. It needs the
CORE One's enclosure, and 01 is a Ø112 mm flat part — give it a brim.

Four settings carry most of the perceived quality:

- **0.15 mm layers on 01, 04, 06 and 07 only.** Everything else stays at 0.2 mm.
- **Fuzzy skin on 01's outer wall.** The shell is already ribbed; fuzzy skin turns that
  into a soft-touch texture and hides the layer lines entirely.
- **Seam set to rear or aligned**, never random — random speckles a curved shell.
- **Dry the PETG.** Wet PETG strings and blooms, and no other setting rescues it.

**On the slew ring, material is mechanical rather than cosmetic.** 04 turns directly
against the 01 journal and the 03 rim on the Ø88 ring, and PETG on PETG is the worst
same-material pairing for stick-slip — it squeaks and can gall under load. Either print
**04 in PLA** against PETG shells, or keep it all PETG and work **PTFE dry lubricant**
into the ring before first assembly. Silicone grease also works but collects court dust.

Avoid silk PLA (weak layer adhesion, cheap up close), gloss black (shows every defect),
and wood or marble fills (abrasive, and they read novelty).

### Verification

All twelve pass an automated gate before export — single connected component, zero
non-manifold edges, positive volume — then an independent re-read of the written
binary confirms every edge is shared by exactly two triangles and the part fits the
Prusa CORE One bed. If a part fails the gate the export is **refused** rather than
writing an empty file. (The previous `cad/blender/exports/` generation silently wrote
five 84-byte zero-triangle STLs; that is what this gate exists to stop.)

---

## Bill of materials

| Item | Qty | Notes |
|---|---:|---|
| MG996R servo (180°) | 1 | 40.7 × 20 × **47.6** mm per TowerPro's drawing; with supplied horn |
| **500 mm 3-pin servo extension lead** | 1 | **see wiring note — the stock 300 mm lead is NOT long enough** |
| 1/4-20 hex nut, 7/16" AF | 1 | steel |
| M3 × 10 screws | 4 | carrier plate → journal top |
| M3 × 8 screws | 4 | servo flange → carrier (holes opened to Ø5.0 for clone drift) |
| M4 × 25 thumbscrew | 1 | locks the cradle tongue |
| Zip tie, 2.5 mm | 1–2 | umbilical strain relief at the pod exit |
| M3 × 16 screws | 2 | leg saddle → Uno tray |
| Strap / velcro tie, ≤25 mm wide | 2 | saddle to leg, and AA bay |
| 4 × AA battery holder | 1 | servo power — **never off the Uno 5 V pin** |
| Arduino Uno R3 | 1 | lives in the leg case |
| Foam/adhesive shim strip | — | takes up phone-pocket slack |

---

## Assembly

1. Press the **1/4-20 nut** up into the hex pocket in the underside of `01`, then
   press `02_nut_retainer` in below it, flush with the seating face.
2. Bolt the **MG996R** to `03_servo_carrier` (4 × M3), servo body hanging *down*
   through the plate, flange resting on top.
3. Drop `04_rotor` over the Ø88 journal on `01`. It lands on the thrust land.
4. Screw `03` down onto the journal top (4 × M3). **The carrier plate is also the
   capture ring** — its Ø96 rim traps the rotor's Ø89 shoulder with ~1 mm of lift play.
5. Centre the servo at 90° *before* fitting `05_drive_dog`. Blade into the rotor
   deck slot at r = 18 mm.
6. Route the servo lead out the exit slot at az 210°, zip-tie it through the two
   tie slots, run it down to the leg case.
6a. Bolt `10_leg_saddle` to the underside of `08_uno_tray` (2 × M3 × 16), strap the
   saddle to a tripod leg, seat the Uno on its four pedestals, clip `09_uno_lid` on.
   `11_aa_bay` straps on separately so neither box has to shrink to fit the other.
7. Slide a cradle tongue into the deck socket, lock with the M4 thumbscrew.
8. Mount the whole pod on the MACTREM's 1/4-20.

---

## Wiring

The servo is on the **stator**, not the rotor — **the wires never rotate.** No service
loop, no twist limit, no cable wrap.

**Length matters and the stock lead is too short:**

```
servo → pod wire exit (internal routing)       ~ 80 mm
pod exit → leg case (straight line)             181 mm
elevator column raise (repo: column_free_h)   + 160 mm
routing slack + service loop (25 %)           + ~86 mm
                                              ─────────
required servo-to-Uno                           507 mm
```

The MG996R's own lead is 300 mm, which leaves only 220 mm outside the pod. Add **one
500 mm 3-pin extension** → 800 mm total, 293 mm of margin. Anything shorter will pull
tight the moment the centre column is raised.

Circuit is unchanged from `hardware/WIRING.md`: signal → D9, grounds commoned,
servo V+ from the 4×AA pack. Do not run an MG996R off the Uno's 5 V pin.

---

## Design decisions worth knowing

- **Bearing load path.** The phone's weight and tipping moment go through the printed
  slew ring into `01` and straight down the 1/4-20 into the tripod. The servo spline
  transmits torque only — the sliding drive dog is deliberately free to float axially
  so it can't be turned into a bearing.
- **Phone centre of mass is on the pan axis.** The cradle tongue is offset forward so
  the axis bisects the phone's *thickness* rather than sitting behind its back face.
  No static yaw imbalance, and portrait↔landscape doesn't change the balance.
- **Friction dominates, not inertia.** Landscape yaw inertia is ~4.7 × 10⁻⁴ kg·m²;
  at 120°/s² that is ~0.01 kg·cm. Bearing friction is ~0.2 kg·cm. The MG996R makes 11.
- **Clearances are deliberately loose** (phone pocket 1.5 mm/side, journal 0.5 mm/side,
  M3 at 3.6). Loose is recoverable with a shim or tape; tight is not recoverable at all.
- **Why Ø112 and not Pivo's Ø73.** The MG996R's output spline is not in the middle of
  the servo — it sits **9.85 mm off the body centre** (measured off TowerPro's own top
  view: 24.1 % of the body length). Since the pan axis must pass through the spline,
  the body hangs off to one side and sweeps a **30.18 mm radius** before any wall can
  exist. That single number, not the electronics, sets the pod's minimum diameter.
  Pivo uses a custom motor; Ø100 is about the floor for a standard-size hobby servo.

---

## Assumptions still to confirm against the real hardware

| Assumption | Used | If wrong |
|---|---|---|
| MG996R overall height | **47.6 mm** (TowerPro drawing, not the 42.9 spec bullet) | print `05b` instead of `05` |
| Phone + case envelope 76.5 × 155.5 × 13.0 mm | bare phone is 149.6 × 71.5 × 8.25 (Apple) | foam shims take up slack; else change `PHONE_W/H/T` and reprint 06/07 |
| MG996R spline offset **9.85 mm** from body centre | **measured off TowerPro's own top view** (spline sits 24.1 % of body length off centre); no datasheet dimensions it | 7.82 mm radial margin + Ø5.0 flange holes absorb it — reprint 03 only if truly wrong |
| Tripod leg up to **50 mm** | PT55's own legs are 20 mm tube | none — saddle is deliberately oversized |
| Tripod screw protrudes 6.5 mm | standard 1/4-20 | currently 5.0 mm engagement, 3.9 threads |
