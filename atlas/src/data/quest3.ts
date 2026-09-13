import type { PartInfo } from './parts'
import type { Preset, System } from './systems'

// Meta Quest 3. Envelope and component counts follow published figures and the
// teardown record; internal placements are schematic. Sources are named on the
// parts whose numbers matter.

export const QUEST3_SYSTEMS: System[] = [
  { id: 'optics', name: 'Optics', note: 'What folds and carries the light.', color: '#7FA3B5' },
  { id: 'display', name: 'Display', note: 'What emits it. The flicker lives here.', color: '#C9A227' },
  { id: 'sensing', name: 'Sensing', note: 'Cameras, depth, tracking.', color: '#4C6472' },
  { id: 'compute', name: 'Compute', note: 'SoC, memory, mainboard.', color: '#3B4650' },
  { id: 'power', name: 'Power', note: 'The cell.', color: '#8A7A4E' },
  { id: 'audio', name: 'Audio', note: 'Open-ear speakers.', color: '#6E7A84' },
  { id: 'chassis', name: 'Chassis', note: 'Shell, facial interface, strap.', color: '#5B646C', ghost: true },
]

export const QUEST3_PRESETS: Preset[] = [
  { id: 'whole', name: 'Assembled', question: 'What is the whole thing?', systems: ['optics', 'display', 'sensing', 'compute', 'power', 'audio', 'chassis'], explode: 0 },
  {
    id: 'flicker', name: 'Flicker chain', question: 'What can actually flash in the wearer’s eye?',
    systems: ['display', 'optics', 'sensing', 'compute'],
    only: ['led-backlight-l', 'led-backlight-r', 'lcd-panel-l', 'lcd-panel-r', 'pancake-lens-l', 'pancake-lens-r', 'rgb-passthrough-camera-l', 'rgb-passthrough-camera-r', 'snapdragon-xr2-gen-2'],
    explode: 0.85,
  },
  {
    id: 'lightpath', name: 'Light path', question: 'World to retina, in order.',
    systems: ['optics', 'display', 'sensing'],
    only: ['rgb-passthrough-camera-l', 'rgb-passthrough-camera-r', 'led-backlight-l', 'led-backlight-r', 'lcd-panel-l', 'lcd-panel-r', 'optical-stack-l', 'optical-stack-r', 'pancake-lens-l', 'pancake-lens-r'],
    explode: 1,
  },
  { id: 'cameras', name: 'Sensor array', question: 'Which camera does what?', systems: ['sensing'], explode: 0.7 },
  { id: 'guts', name: 'Compute + power', question: 'What drives it?', systems: ['compute', 'power'], explode: 0.6 },
]

const pair = (base: Omit<PartInfo, 'id' | 'name'>, idL: string, idR: string, name: string): PartInfo[] => [
  { ...base, id: idL, name: `${name} (left)` },
  { ...base, id: idR, name: `${name} (right)` },
]

export const QUEST3_PARTS: PartInfo[] = [
  // ------------------------------------------------------------- display
  ...pair({
    role: 'The light source — and the origin of every flicker concern in this headset.',
    does:
      'An LCD does not emit; this LED array behind it does. In a headset it is not run continuously but strobed: it flashes on for a fraction of each frame and sits dark for the rest, so that the image does not smear when the head turns. That strobe is the flicker the photosensitivity warning is really about, and its frequency is the display refresh rate — nothing else.',
    specs: [['Drive', 'low persistence, strobed'], ['Strobe frequency', '= refresh rate'], ['Peak display brightness', '≈ 100 nits']],
    flicker: {
      role: 'source',
      note: 'At 72 Hz the backlight pulses 72 times a second. Photoparoxysmal responses peak around 15–25 Hz and the guidelines flag 3–60 Hz, so even the lowest Quest refresh rate already sits above the provocative band — and 90 or 120 Hz puts it further clear. This is why refresh rate, not brightness, is the first lever.',
    },
    watch:
      'Never let the refresh rate vary at runtime. A variable rate against a low-persistence backlight is perceived as the whole view brightening and dimming, which manufactures exactly the low-frequency modulation you are trying to avoid.',
    source: 'Low-persistence strobing and the variable-refresh brightness artefact are described in Meta’s own display patents and in the VR flicker literature (elaTCSF, 2025).',
  }, 'led-backlight-l', 'led-backlight-r', 'LED backlight'),

  ...pair({
    role: 'The panel that shapes the backlight into an image.',
    does:
      'A 2064 × 2208 RGB-stripe LCD per eye, about 30 % more pixels per eye than Quest 2. It modulates light rather than producing it, so it sets what the image is but not whether it flickers. Anything your shader does to clamp luminance or desaturate a flash happens in the content that reaches this panel.',
    specs: [['Resolution', '2064 × 2208 per eye'], ['Subpixels', 'RGB stripe'], ['Type', 'LCD, backlit']],
    flicker: {
      role: 'lever',
      note: 'This is where render-side filtering lands: cap luminance transitions frame to frame, limit how much of the field may flash at once, and treat saturated red separately. That maps directly onto the flash-rate, flash-area and red-flash criteria in the 2024 photosensitivity gap analysis your pitch already cites.',
    },
  }, 'lcd-panel-l', 'lcd-panel-r', 'LCD panel'),

  // -------------------------------------------------------------- optics
  ...pair({
    role: 'The folded optic that makes the headset thin.',
    does:
      'A pancake lens bounces light back and forth inside a thin stack — through a quarter-wave plate, off a reflective polarizer, off a half mirror — so a long focal path fits in a short one. That is what makes Quest 3 roughly 40 % slimmer than Quest 2.',
    specs: [['Type', 'pancake (folded)'], ['Stack', 'half mirror + QWP + reflective polarizer'], ['Thickness saving', '≈ 40 % vs Quest 2']],
    flicker: {
      role: 'path',
      note: 'Every fold costs light — a pancake stack throws away most of what the backlight makes, which is why peak brightness is only about 100 nits. Low peak brightness is genuinely protective here: there is a hard ceiling on how bright a flash can be, and it is far below a television.',
    },
  }, 'pancake-lens-l', 'pancake-lens-r', 'Pancake lens'),

  ...pair({
    role: 'The polarizer and wave-plate layers in front of the panel.',
    does: 'The polarization management that makes the pancake fold work. Light leaves the panel linearly polarized, gets rotated by the quarter-wave plate on each pass, and is either reflected or transmitted depending on its state.',
    flicker: { role: 'path', note: 'Passive. It attenuates, it never modulates — so it cannot introduce flicker of its own.' },
  }, 'optical-stack-l', 'optical-stack-r', 'Optical stack'),

  {
    id: 'ipd-motor', name: 'IPD motor', role: 'Motorised interpupillary distance adjustment.',
    does: 'Drives the two optical assemblies together or apart so the lens centres line up with the wearer’s pupils.',
    watch: 'Worth setting properly before any session: an off-axis pupil sees more edge distortion and a dimmer, less uniform field, which makes the display harder to tolerate for longer wear.',
  },

  // ------------------------------------------------------------- sensing
  ...pair({
    role: 'The colour camera that lets the headset show you the real room.',
    does:
      'A 4 MP RGB camera per side. These are the reason Quest 3 is the hackathon device rather than the roadmap one: because the wearer sees a *video* of the world rather than the world itself, software can darken, desaturate or filter the room before it is ever displayed. A pair of AR glasses physically cannot do that.',
    specs: [['Resolution', '4 MP each'], ['Count', '2 (one per eye)'], ['Function', 'full-colour video passthrough']],
    flicker: {
      role: 'source',
      note: 'A second, independent flicker path that is easy to miss. Mains lighting in Berlin runs at 50 Hz, and a rolling-shutter camera sampling a 50 Hz-flickering room can beat against it and produce banding or a low-frequency wobble in passthrough that is not present to the naked eye. Lock exposure to a multiple of 10 ms when passthrough is on.',
    },
  }, 'rgb-passthrough-camera-l', 'rgb-passthrough-camera-r', 'RGB passthrough camera'),

  ...[1, 2, 3, 4].map((n): PartInfo => ({
    id: `ir-tracking-camera-${n}`, name: `IR tracking camera ${n}`,
    role: 'One of four infrared cameras that work out where the headset and the hands are.',
    does: 'Monochrome IR cameras running inside-out SLAM: they watch fixed features in the room to place the headset in space, and they track hands and controllers. They feed tracking, not the image you see.',
    flicker: { role: 'path', note: 'Infrared and never shown to the wearer, so it contributes nothing to photosensitivity risk. Listed so the count is honest: six cameras in total, only two of which you ever look through.' },
  })),

  {
    id: 'depth-projector', name: 'Depth projector', role: 'The dot projector that measures room geometry.',
    does: 'Throws a structured infrared pattern onto the room so the headset can measure distance and build a mesh of the space. Ten times the resolution of Quest 2’s approach and about twice Quest Pro’s for passthrough quality.',
    flicker: { role: 'path', note: 'Infrared, outward-facing, never enters the wearer’s visible field.' },
  },
  { id: 'camera-pill-left', name: 'Camera pill (left)', role: 'One of the three sensor housings on the face of the visor.', does: 'Houses one RGB passthrough camera and IR tracking cameras behind a single IR-transparent window.' },
  { id: 'camera-pill-centre', name: 'Camera pill (centre)', role: 'The middle sensor housing.', does: 'Carries the depth projector and tracking optics.' },
  { id: 'camera-pill-right', name: 'Camera pill (right)', role: 'The third sensor housing.', does: 'Mirrors the left pill: one RGB passthrough camera plus IR tracking cameras.' },

  // ------------------------------------------------------------- compute
  {
    id: 'snapdragon-xr2-gen-2', name: 'Snapdragon XR2 Gen 2', role: 'The SoC — and the component that actually holds the flicker lever.',
    does:
      'Qualcomm’s XR2 Gen 2, with roughly double Quest 2’s graphics performance. It runs the compositor, so it is what decides the display refresh rate, the passthrough pipeline and every shader that touches the image before it reaches the panel.',
    specs: [['Standard refresh rates', '72 / 80 / 90 / 120 Hz'], ['Hardware range', 'any integer 72–207 Hz via API'], ['RAM', '8 GB LPDDR5']],
    flicker: {
      role: 'lever',
      note: 'The single most useful fact for this project: the display hardware accepts any integer refresh rate from 72 to 207 Hz through standard APIs, and the app requests it. Pick one high rate, hold it, and the backlight strobe sits far above any provocative frequency. The catch is Meta’s own rule — an app that cannot sustain the rate it asks for fails store review, so the rate you pick has to be one your render loop can actually hold.',
    },
    watch: 'Requesting a high refresh rate and then dropping frames is worse than asking for a low one, because inconsistent frame delivery is itself perceived as flicker.',
  },
  { id: 'lpddr5-8-gb', name: 'LPDDR5 memory', role: '8 GB of system memory.', does: 'Shared between the OS, the app and the passthrough pipeline. Double Quest 2’s 4 GB.', specs: [['Capacity', '8 GB'], ['Type', 'LPDDR5']] },
  { id: 'mainboard', name: 'Mainboard', role: 'The board everything else lands on.', does: 'Carries the SoC, memory, camera interfaces and power management.' },

  // --------------------------------------------------------- power/audio
  { id: 'battery', name: 'Battery', role: 'The cell, in the front of the visor.', does: 'Powers roughly two to three hours of mixed-reality use. Its position in front is part of why the headset needs the strap to counterbalance.' },
  ...pair({ role: 'One of two open-ear speakers.', does: 'Fires down toward the ear without covering it, so the wearer still hears the room. For this project that matters more than audio quality: a person having a seizure must still be audible to whoever is with them, and a caregiver must be able to talk to the wearer.' }, 'speaker-l', 'speaker-r', 'Speaker'),

  // ------------------------------------------------------------- chassis
  { id: 'visor-shell', name: 'Visor shell', role: 'The body of the headset.', does: 'Holds the optics, boards and sensors. About 30–40 % thinner than Quest 2 thanks to the pancake optics, at roughly 515 g all in.', specs: [['Mass', '≈ 515 g'], ['Depth vs Quest 2', '30–40 % thinner']] },
  { id: 'facial-interface', name: 'Facial interface', role: 'The foam that meets the face and blocks stray light.', does: 'Seals the optics from ambient light and spreads the load across the cheeks and forehead.', watch: 'The light seal is also the comfort limit. For an always-on accessibility device meant to be tolerable for hours, this is the part most likely to end the session — the pitch’s own argument is that devices get abandoned for comfort, not accuracy.' },
  { id: 'head-strap', name: 'Head strap', role: 'What holds it on and counterbalances the front-heavy visor.', does: 'Carries load onto the crown and back of the head. The stock strap is the usual first upgrade for long sessions.' },
]
