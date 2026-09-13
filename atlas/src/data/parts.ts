// Knowledge layer for the FollowCam Pod v4 atlas.
//
// Every number here is traceable to cad/blender/pod_v4/README.md or build_v4.py.
// Geometry (sizes, bounding boxes, explode vectors) is NOT duplicated here - it is
// generated into src/generated/parts.geometry.json by scripts/export_atlas_glb.py
// and joined by id at load time, so the atlas can never drift from the STLs.

export type SystemId =
  // FollowCam Pod v4
  | 'stator' | 'rotor' | 'portrait' | 'landscape' | 'hardware' | 'external'
  // head-mounted displays
  | 'optics' | 'display' | 'sensing' | 'compute' | 'power' | 'audio' | 'chassis'

export interface PartInfo {
  /** matches the `id` in parts.geometry.json */
  id: string
  /** printed-part number, shown as the big index in the detail panel */
  code?: string
  name: string
  /** one line: what this thing IS */
  role: string
  /** what it DOES, and why it is shaped the way it is */
  does: string
  specs?: [string, string][]
  /** what it touches, mechanically */
  interfaces?: string[]
  /** the failure mode worth knowing before you print or assemble */
  watch?: string
  /** where a contested number came from */
  source?: string
  /** extra dimension callouts to draw on the 3D view, beyond the bounding box */
  callouts?: { label: string; value: string }[]
  bom?: string
  /**
   * Where this component sits in the photosensitivity chain. 'source' means it
   * emits or modulates light in a way that can flicker; 'lever' means it is a
   * place the project can intervene; 'path' means light passes through it and
   * its properties change what reaches the eye.
   */
  flicker?: { role: 'source' | 'lever' | 'path'; note: string }
}

export const PARTS: PartInfo[] = [
  // ------------------------------------------------------------------ stator
  {
    id: '01-stator-shell',
    code: '01',
    name: 'Stator shell',
    role: 'The pod body. Everything else is bolted to this, and it never turns.',
    does:
      'Floor, lower shell, thrust land and the Ø88 journal are printed as one piece, so the slew ring cannot come apart from the body it rides on. The phone’s weight and its tipping moment run through the journal, into this shell, and straight down the 1/4-20 into the tripod. The servo lead leaves through a slot at azimuth 210°, aimed at the leg case.',
    specs: [
      ['Footprint', 'Ø112 × 35 mm'],
      ['Solid volume', '161 cm³'],
      ['Material', 'PETG or PLA, 0.2 mm layers'],
      ['Perimeters', '4 — do not reduce'],
      ['Supports', 'none'],
    ],
    callouts: [
      { label: 'Pod outer', value: 'Ø112.0' },
      { label: 'Journal', value: 'Ø88.0' },
      { label: 'Journal bore', value: 'Ø76.0' },
    ],
    interfaces: [
      '1/4-20 hex nut pocket in the underside → tripod screw',
      'Ø88 journal → rotor slew ring rides on it',
      'Journal top → servo carrier, 4 × M3 × 10',
      'Wire exit slot at az 210° + two zip-tie slots',
    ],
    watch:
      'This is the one part carrying the whole load into the tripod. Dropping its perimeter count to save time is the way to break the pod.',
  },
  {
    id: '02-nut-retainer',
    code: '02',
    name: 'Nut retainer',
    role: 'A 0.1 cm³ plug that stops the tripod nut falling out.',
    does:
      'The 1/4-20 hex nut presses up into a pocket in the underside of the stator shell; this retainer presses in below it, flush with the seating face. Without it the nut drops out the moment you lift the pod off the tripod, and spins in its pocket while you are trying to thread it on.',
    specs: [
      ['Footprint', '11.3 × 13 × 1.5 mm'],
      ['Solid volume', '0.1 cm³'],
      ['Nut', '1/4-20, 7/16" AF, steel'],
    ],
    interfaces: ['Stator shell nut pocket', '1/4-20 hex nut'],
    watch: 'Fit the nut first, then this. It cannot be inserted afterwards.',
  },
  {
    id: '03-servo-carrier-mg996r',
    code: '03',
    name: 'Servo carrier',
    role: 'A replaceable plate that holds the servo — and quietly doubles as the rotor capture ring.',
    does:
      'The MG996R bolts to this plate with its body hanging down through the cut-out and its flange resting on top; the plate then screws down onto the journal. Its Ø96 rim overhangs the rotor’s Ø89 shoulder, so the same part that mounts the servo also traps the rotor with about 1 mm of lift play. One plate, two jobs, and a servo swap needs no other part reprinted.',
    specs: [
      ['Footprint', '96 × 96 × 5 mm'],
      ['Solid volume', '30 cm³'],
      ['To journal', '4 × M3 × 10'],
      ['To servo', '4 × M3 × 8'],
    ],
    callouts: [
      { label: 'Plate', value: 'Ø96.0' },
      { label: 'Rotor shoulder', value: 'Ø89.0' },
    ],
    interfaces: ['Journal top of the stator shell', 'MG996R mounting flange', 'Rotor shoulder (capture)'],
    watch:
      'Flange holes are opened to Ø5.0 rather than 3.2 because MG996R clones drift. If a clone still will not line up, this is the only part you reprint.',
  },

  // ------------------------------------------------------------------- rotor
  {
    id: '04-rotor',
    code: '04',
    name: 'Rotor',
    role: 'The turntable. Everything above it pans; everything below it does not.',
    does:
      'Wraps around the servo on a Ø88 printed slew ring and lands on the thrust land of the stator. It carries the cradle and the phone, takes the tipping moment, and hands it to the journal — which is why the servo spline only ever sees torque, never weight. The deck has the keyed cradle socket and the drive-dog slot at r = 18 mm.',
    specs: [
      ['Footprint', 'Ø110 × 55 mm'],
      ['Solid volume', '191 cm³'],
      ['Gap to shell', '1 mm shadow line'],
      ['Journal clearance', '0.5 mm per side'],
      ['Orientation', 'pre-inverted — deck on the bed'],
    ],
    callouts: [
      { label: 'Rotor outer', value: 'Ø110.0' },
      { label: 'Bore', value: 'Ø97.0' },
      { label: 'Dog slot at', value: 'r 18.0' },
    ],
    interfaces: ['Ø88 journal (rides on)', 'Servo carrier rim (captured by)', 'Drive dog slot', 'Cradle tongue socket'],
    watch:
      'The STL is already rotated deck-down. Do not let the slicer auto-orient it — printing deck-up asks the machine to bridge a Ø97 mm hole.',
  },
  {
    id: '05-drive-dog-42.9-47.6',
    code: '05',
    name: 'Drive dog',
    role: 'The 1.1 cm³ part that turns servo rotation into camera pan.',
    does:
      'Sits on the servo spline and reaches up into the slot in the rotor deck. Because it is deliberately free to float along its axis, it can only ever push sideways — it cannot be pressed into service as a bearing, which is what keeps the phone’s weight off the servo output shaft. The drive is 1:1, so the firmware’s existing 40–140° range maps straight onto ±50° of camera pan with no calibration change.',
    specs: [
      ['Footprint', '37 × 13.5 × 8 mm'],
      ['Solid volume', '1.1 cm³'],
      ['Blade radius', 'r = 18 mm on the deck'],
      ['Drive ratio', '1:1'],
      ['Print when', 'servo measures ≈ 47.6 mm'],
    ],
    interfaces: ['MG996R output spline', 'Rotor deck slot'],
    watch:
      'Centre the servo at 90° before this goes on, or the pan envelope ends up rotated and the software limits point the wrong way.',
  },
  {
    id: '05b-drive-dog-short-42.9',
    code: '05b',
    name: 'Drive dog, short-servo variant',
    role: 'The same part with a 4.7 mm taller blade, for the other MG996R height.',
    does:
      'Identical job to 05. The pod is built around a 47.6 mm servo, so if yours is actually the shorter 42.9 mm variant the spline sits lower and the standard dog cannot reach the rotor deck. This one has a longer blade to close that gap.',
    specs: [
      ['Footprint', '37 × 13.5 × 12.7 mm'],
      ['Solid volume', '1.3 cm³'],
      ['Print when', 'servo measures ≈ 42.9 mm'],
    ],
    source:
      'TowerPro’s datasheet contradicts itself: the spec bullet says 42.9 mm, the dimensioned drawing on the same sheet says 47.6 mm overall. The pod is built to 47.6, because being wrong that way leaves air and being wrong the other way means the pod will not close.',
    interfaces: ['MG996R output spline', 'Rotor deck slot'],
    watch:
      'Put a ruler on the actual servo, base to the top of the splined shaft, and print whichever dog matches. Both are about 1 cm³ and print in minutes, so printing both and deciding at assembly costs nothing.',
  },

  // ---------------------------------------------------------------- portrait
  {
    id: '06-cradle-portrait',
    code: '06',
    name: 'Portrait cradle',
    role: 'Full-wrap phone shell with a keyed tongue, for portrait shooting.',
    does:
      'Holds the phone by its back and four front corner lips, and drops into the rotor deck socket on a keyed tongue locked by one M4 thumbscrew — so switching orientation is a thumbscrew, not a reprint. The tongue is deliberately offset forward so the pan axis bisects the phone’s thickness rather than passing behind its back face, which is what keeps portrait and landscape equally balanced.',
    specs: [
      ['Footprint', '173 × 87.5 × 22 mm'],
      ['Solid volume', '67 cm³'],
      ['Phone envelope', '76.5 × 155.5 × 13.0 mm'],
      ['Pocket clearance', '1.5 mm per side'],
      ['Supports', 'yes — under the four front lips only'],
    ],
    interfaces: ['Rotor deck socket (keyed tongue)', 'M4 × 25 thumbscrew', 'Phone'],
    watch:
      'Pre-oriented back-down. Standing it on its tongue would run the one load-bearing joint straight across the layer lines. Take up any slack with foam shim rather than reprinting.',
  },
  {
    id: 'phone',
    name: 'Phone (portrait)',
    role: 'The camera, and the reason the whole pod exists.',
    does:
      'Placeholder at 76.5 × 155.5 × 13.0 mm — a phone plus its everyday case, not a bare phone. A bare iPhone is 149.6 × 71.5 × 8.25 mm; the difference is absorbed by the 1.5 mm-per-side pocket clearance and foam shim.',
    specs: [['Envelope', '76.5 × 155.5 × 13.0 mm'], ['Assumed payload', '≈ 250 g']],
    watch:
      'Change PHONE_W / PHONE_H / PHONE_T at the top of build_v4.py and reprint 06 and 07 if your phone is genuinely outside the shim range.',
  },
  { id: 'screen', name: 'Phone screen (portrait)', role: 'The display face of the portrait placeholder.', does: 'Marks which way the phone faces in the cradle. Geometry only — it has no function in the assembly.' },

  // --------------------------------------------------------------- landscape
  {
    id: '07-cradle-landscape',
    code: '07',
    name: 'Landscape cradle',
    role: 'The same shell rotated 90°, for landscape shooting.',
    does:
      'Interchangeable with the portrait cradle through the same keyed socket and thumbscrew. Because the tripod screw, servo output, slew ring, deck socket and the phone’s horizontal centre are all coaxial, swapping orientation introduces no static yaw imbalance — the servo does not have to work harder in one mode than the other.',
    specs: [
      ['Footprint', '94 × 166.5 × 22 mm'],
      ['Solid volume', '55 cm³'],
      ['Supports', 'yes — under the four front lips only'],
    ],
    interfaces: ['Rotor deck socket (keyed tongue)', 'M4 × 25 thumbscrew', 'Phone'],
    watch: 'Pre-oriented back-down, same reason as the portrait cradle.',
  },
  { id: 'phone.001', name: 'Phone (landscape)', role: 'The camera, mounted the other way up.', does: 'Same 76.5 × 155.5 × 13.0 mm envelope as the portrait placeholder, rotated into the landscape cradle.' },
  { id: 'screen.001', name: 'Phone screen (landscape)', role: 'The display face of the landscape placeholder.', does: 'Marks which way the phone faces. Geometry only.' },

  // ---------------------------------------------------------------- hardware
  {
    id: 'mg996r-body',
    name: 'MG996R servo',
    role: 'The muscle — and the single component that set the pod’s diameter.',
    does:
      'Bolts under the carrier plate and drives the rotor through the dog at 1:1. It makes roughly 11 kg·cm; the pod needs about 0.2 kg·cm to overcome bearing friction and about 0.01 kg·cm to accelerate the phone. Friction dominates, not inertia, so the servo is not close to working hard.',
    specs: [
      ['Body', '53.6 × 20 × 47.6 mm'],
      ['Torque', '≈ 11 kg·cm'],
      ['Friction to beat', '≈ 0.2 kg·cm'],
      ['Landscape yaw inertia', '4.7 × 10⁻⁴ kg·m²'],
      ['Lead', '300 mm — too short, see below'],
    ],
    callouts: [
      { label: 'Spline offset', value: '9.85' },
      { label: 'Sweep radius', value: 'r 30.18' },
    ],
    source:
      'The spline is not in the middle of the servo. It sits 9.85 mm off the body centre — 24.1 % of the body length, measured off TowerPro’s own top view, because no datasheet dimensions it.',
    interfaces: ['Servo carrier plate, 4 × M3 × 8', 'Drive dog on the output spline', 'Signal → Uno D9', '4 × AA pack for V+'],
    watch:
      'Never run an MG996R from the Uno’s 5 V pin — use the AA pack and common the grounds. And the stock 300 mm lead will not reach: you need one 500 mm extension, or it pulls tight the moment the centre column goes up.',
    bom: 'MG996R servo (180°), 1 off, with supplied horn',
  },
  {
    id: 'spline',
    name: 'Output spline',
    role: 'The 5.8 mm splined shaft where torque leaves the servo.',
    does:
      'The pan axis passes through this spline, not through the middle of the servo. Because the spline sits 9.85 mm off centre, the servo body hangs to one side and sweeps a 30.18 mm radius before any wall can exist — and that one number, not the electronics, is what forces the pod to Ø112 instead of Pivo’s Ø73.',
    specs: [['Spline', 'Ø5.8 × 4.2 mm'], ['Offset from body centre', '9.85 mm'], ['Swept radius', '30.18 mm']],
    interfaces: ['Drive dog'],
    watch:
      'Pivo gets to Ø73 with a custom motor. Ø100 is about the floor for any standard-size hobby servo, so this is a consequence of the component choice, not a design slip.',
  },

  // ---------------------------------------------------------------- external
  {
    id: '08-uno-tray',
    code: '08',
    name: 'Uno tray',
    role: 'The box the Arduino lives in, down on a tripod leg.',
    does:
      'The controller is deliberately not in the pod. Keeping it on a leg means the pod only has to be big enough for the servo, and — more importantly — the servo lead leaves from the stator, so no wire ever rotates. No service loop, no twist limit, no cable wrap.',
    specs: [['Footprint', '96 × 80 × 30 mm'], ['Solid volume', '85 cm³'], ['Uno R3', '68.6 × 53.4 mm']],
    interfaces: ['Uno R3 on four pedestals', 'Lid clips on top', 'Leg saddle bolts underneath, 2 × M3 × 16'],
  },
  { id: '09-uno-lid', code: '09', name: 'Uno lid', role: 'The clip-on cover for the Uno tray.', does: 'Closes the controller box so the board is not exposed on a tripod leg at a sports ground.', specs: [['Footprint', '96 × 80 × 5 mm'], ['Solid volume', '34 cm³']] },
  {
    id: '10-leg-saddle',
    code: '10',
    name: 'Leg saddle',
    role: 'The V-block that straps the controller box to a tripod leg.',
    does:
      'Bolts to the underside of the Uno tray and straps around a leg. Deliberately oversized: it takes legs up to 50 mm, where the PT55’s own legs are 20 mm tube — so it fits tripods this project has never seen.',
    specs: [['Footprint', '96 × 66 × 36 mm'], ['Solid volume', '155 cm³'], ['Leg capacity', 'up to 50 mm'], ['Orientation', 'pre-inverted — flat face down, V opens up']],
    interfaces: ['Uno tray underside, 2 × M3 × 16', 'Tripod leg + strap'],
    watch: 'Printed groove-down it would start as two thin islands, so the STL arrives already flipped.',
  },
  {
    id: '11-aa-bay',
    code: '11',
    name: 'AA bay',
    role: 'Servo power, strapped on as its own box.',
    does:
      'Holds the 4 × AA pack that feeds the servo. It straps on separately from the Uno tray specifically so neither box has to shrink to accommodate the other — two small boxes beat one awkward one.',
    specs: [['Footprint', '68 × 41 × 21 mm'], ['Solid volume', '19 cm³'], ['Cells', '4 × AA']],
    interfaces: ['Servo V+', 'Ground commoned with the Uno'],
    watch: 'This exists because the Uno’s 5 V pin cannot feed an MG996R. Do not be tempted to skip it.',
  },
  { id: 'uno-r3', name: 'Arduino Uno R3', role: 'The controller.', does: 'Receives A<angle> commands over USB serial from the laptop and drives the servo signal line on D9, slew-rate limited. Unchanged from the original handle-fork prototype — the pod is a mechanical redesign, not a software one.', specs: [['Board', '68.6 × 53.4 mm'], ['Servo signal', 'D9'], ['Protocol', 'A<angle>\\n over USB serial']], interfaces: ['USB-B to laptop', 'D9 → servo signal', 'GND commoned with the AA pack'] },
  { id: 'usb', name: 'USB-B port', role: 'The link to the laptop running the tracker.', does: 'Carries the serial angle commands in and powers the board. The servo does not draw through it.' },
  { id: '1-4-20', name: '1/4-20 tripod screw', role: 'The universal interface the pod deliberately standardises on.', does: 'The pod threads onto the tripod’s own 1/4-20 rather than reproducing the PT55’s proprietary dovetail — because replacement-plate listings disagree with each other by a few millimetres, and a standard screw fits every tripod instead of one.', specs: [['Thread', '1/4-20'], ['Engagement', '5.0 mm, ≈ 3.9 threads'], ['Screw protrusion assumed', '6.5 mm']], watch: 'Fit-check against the real plate. This is the assumption most likely to differ on your tripod.' },
  { id: 'mactrem-qr-plate', name: 'MACTREM quick-release plate', role: 'The tripod’s own plate — replaced by the pod, not used by it.', does: 'Shown for context. The pod replaces the MACTREM head entirely, so this plate is not in the load path; only its 1/4-20 screw is.', specs: [['Plate top', '65 × 45 mm'], ['Receiver footprint', '≈ 44 × 44 mm']] },
  { id: 'crown', name: 'Tripod crown', role: 'Where the three legs meet the centre column.', does: 'Context geometry for the MACTREM PT55. It is also the obstruction that sets how far the centre column can rise — which is where the extra 160 mm in the cable-length budget comes from.' },
  { id: 'leg', name: 'Tripod leg', role: 'One of three aluminium legs.', does: 'MACTREM PT55, 20 mm tube, listed at 4–5 kg payload. The leg saddle straps to one of these.', specs: [['Tube', '20 mm'], ['Payload', '4–5 kg listed']] },
  { id: 'leg.001', name: 'Tripod leg', role: 'One of three aluminium legs.', does: 'MACTREM PT55, 20 mm tube.' },
  { id: 'leg.002', name: 'Tripod leg', role: 'One of three aluminium legs.', does: 'MACTREM PT55, 20 mm tube — this is the one the Uno case clips to, at azimuth 210°.' },
]

export const BY_ID: Record<string, PartInfo> = Object.fromEntries(PARTS.map((p) => [p.id, p]))
