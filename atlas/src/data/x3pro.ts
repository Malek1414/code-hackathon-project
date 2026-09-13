import type { PartInfo } from './parts'
import type { Preset, System } from './systems'

// RayNeo X3 Pro. Published specifications where stated; internal placements are
// schematic. The point of this device in the project is what it *cannot* do.

export const X3PRO_SYSTEMS: System[] = [
  { id: 'optics', name: 'Optics', note: 'Waveguides in the lenses.', color: '#7FA3B5' },
  { id: 'display', name: 'Display', note: 'MicroLED engines in the frame.', color: '#C9A227' },
  { id: 'sensing', name: 'Sensing', note: 'Camera, SLAM, microphones.', color: '#4C6472' },
  { id: 'compute', name: 'Compute', note: 'AR1 and the board.', color: '#3B4650' },
  { id: 'power', name: 'Power', note: 'Two cells, one per temple.', color: '#8A7A4E' },
  { id: 'audio', name: 'Audio', note: 'Open-ear drivers.', color: '#6E7A84' },
  { id: 'chassis', name: 'Chassis', note: 'Titanium frame and temples.', color: '#5B646C', ghost: true },
]

export const X3PRO_PRESETS: Preset[] = [
  { id: 'whole', name: 'Assembled', question: 'What is the whole thing?', systems: ['optics', 'display', 'sensing', 'compute', 'power', 'audio', 'chassis'], explode: 0 },
  {
    id: 'lightpath', name: 'Light path', question: 'How does an image get into the eye?',
    systems: ['display', 'optics'],
    only: ['microled-engine-l', 'microled-engine-r', 'waveguide-l', 'waveguide-r'],
    explode: 1,
  },
  {
    id: 'flicker', name: 'Flicker chain', question: 'What can flash, and how bright?',
    systems: ['display', 'optics', 'compute'],
    only: ['microled-engine-l', 'microled-engine-r', 'waveguide-l', 'waveguide-r', 'snapdragon-ar1-gen-1'],
    explode: 0.85,
  },
  { id: 'temples', name: 'What is in the temples', question: 'Where does everything fit in 76 g?', systems: ['compute', 'power', 'audio', 'chassis'], explode: 0.75 },
]

const pair = (base: Omit<PartInfo, 'id' | 'name'>, idL: string, idR: string, name: string): PartInfo[] => [
  { ...base, id: idL, name: `${name} (left)` },
  { ...base, id: idR, name: `${name} (right)` },
]

export const X3PRO_PARTS: PartInfo[] = [
  // ------------------------------------------------------------- display
  ...pair({
    role: 'The light engine — RayNeo’s “Firefly”, one per eye.',
    does:
      'A full-colour microLED projector small enough to hide in the frame above the lens. Unlike the Quest’s LCD it emits directly, with no backlight and no polarizer stack, which is how it reaches 3,500 nits typical and 6,000 nits peak from something this size.',
    specs: [['Type', 'full-colour microLED'], ['Typical brightness', '3,500 nits'], ['Peak brightness', '6,000 nits'], ['Refresh rate', '60 Hz'], ['Colours', '16.77 million']],
    flicker: {
      role: 'source',
      note:
        'Two things differ from the Quest and both cut the wrong way. The refresh rate is 60 Hz rather than 72–120, which sits at the very top of the 3–60 Hz band the photosensitivity guidelines flag rather than clear of it. And microLED brightness is normally set by pulse-width modulation, so dimming the display does not reduce flicker — it deepens the modulation. A dimmed microLED can flicker harder than a bright one.',
    },
    watch: 'Before this device is used with a photosensitive wearer, the PWM dimming frequency needs to be measured, not assumed. It is not in the published specifications, and it is the number that decides whether dimming helps or hurts.',
  }, 'microled-engine-l', 'microled-engine-r', 'MicroLED engine'),

  // -------------------------------------------------------------- optics
  ...pair({
    role: 'The lens that carries the image to the eye — and the reason this device cannot dim the world.',
    does:
      'A waveguide co-developed with Applied Materials. Light from the engine is coupled into the lens, bounced along inside it by total internal reflection, and coupled back out in front of the pupil, giving about a 30° field of view — roughly a 43-inch screen seen from two metres.',
    specs: [['Type', 'diffractive waveguide'], ['Field of view', '30°'], ['Apparent image', '43" at 2 m'], ['Partner', 'Applied Materials']],
    flicker: {
      role: 'path',
      note:
        'This is the single most important line in the whole comparison. A waveguide is *additive*: it can only add light to what the eye already sees. It cannot subtract, attenuate or darken the real world. So the headline intervention in this project — dim the flashing thing in front of the wearer — is physically impossible on this device. It can show the breathing cue and the alert, and that is all.',
    },
    watch: 'If a photosensitive trigger is in the room, these glasses cannot take it away. Any protection they offer has to come from telling the wearer, not from filtering what they see.',
  }, 'waveguide-l', 'waveguide-r', 'Waveguide'),

  // ------------------------------------------------------------- compute
  {
    id: 'snapdragon-ar1-gen-1', name: 'Snapdragon AR1 Gen 1', role: 'The SoC, in the right temple.',
    does: 'Qualcomm’s AR1, the part designed for glasses rather than headsets, with 4 GB of RAM and 32 GB of storage. It runs the display pipeline, the camera and the on-device assistant work.',
    specs: [['SoC', 'Snapdragon AR1 Gen 1'], ['RAM', '4 GB'], ['Storage', '32 GB']],
    flicker: {
      role: 'lever',
      note: 'A much narrower lever than the Quest’s. Refresh rate is fixed at 60 Hz, so the only thing controllable here is content: what is drawn, how bright, how much of the 30° field it covers, and how fast it may change.',
    },
  },
  { id: 'mainboard', name: 'Mainboard', role: 'The board running the length of the temple.', does: 'Long and narrow because that is the only shape available inside a temple arm. Carries the SoC, camera interface and power management.' },

  // ------------------------------------------------------------- sensing
  { id: 'rgb-camera-12-mp', name: 'RGB camera', role: 'The 12 MP forward camera.', does: 'A Sony IMX681 for stills, video and the visual input the on-device assistant works from.', specs: [['Sensor', 'Sony IMX681'], ['Resolution', '12 MP']], flicker: { role: 'path', note: 'Outward-facing only. Nothing it captures is re-displayed to the wearer as a live view, so unlike the Quest’s passthrough cameras it adds no flicker path into the eye.' } },
  ...pair({ role: 'One of two spatial tracking cameras.', does: 'Feeds 6DoF and SLAM through RayNeo’s Falcon Image positioning, so overlaid content can stay anchored to the room rather than drifting with the head.', specs: [['Tracking', '6DoF + SLAM']] }, 'slam-camera-l', 'slam-camera-r', 'SLAM camera'),
  ...[1, 2].map((n): PartInfo => ({ id: `microphone-${n}`, name: `Microphone ${n}`, role: 'One of the array microphones in the frame front.', does: 'Picks up voice for the assistant and for calls, with beamforming across the array to reject room noise.' })),

  // --------------------------------------------------------- power/audio
  ...pair({
    role: 'One of two cells, one per temple.',
    does: 'Split left and right so the mass balances across the head rather than hanging off one side.',
    watch: 'Battery life is the honest weakness of this class of device and reviewers have been blunt about it. For an always-on accessibility aid meant to run all day, this is a harder problem than anything optical.',
  }, 'battery-l', 'battery-r', 'Battery'),
  ...pair({ role: 'One of two open-ear drivers.', does: 'Fires toward the ear without occluding it, so the wearer keeps full awareness of the room — the same reasoning as on the Quest, and the same requirement for a device a caregiver has to talk through.' }, 'speaker-l', 'speaker-r', 'Speaker'),

  // ------------------------------------------------------------- chassis
  { id: 'frame-front', name: 'Frame front', role: 'The titanium front that carries the optics.', does: 'Holds both waveguides, both light engines, the camera and the microphone array — and keeps the whole device at 76 g, roughly a seventh of a Quest 3.', specs: [['Material', 'titanium alloy'], ['Total mass', '76 g']] },
  { id: 'bridge', name: 'Bridge', role: 'The nose bridge.', does: 'Takes most of the weight. At 76 g total, it can do that without a strap — which is the entire argument for this form factor as the roadmap device.' },
  ...pair({ role: 'One of two temple arms.', does: 'Not just an arm: each temple is a sealed enclosure holding a battery, a speaker, and on the right the SoC and mainboard. There is no other volume in a pair of glasses to put them in.' }, 'temple-l', 'temple-r', 'Temple'),
]
