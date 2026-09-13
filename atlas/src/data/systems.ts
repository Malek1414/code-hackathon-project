import type { SystemId } from './parts'

export interface System {
  id: SystemId
  name: string
  /** what the layer means, shown under the toggle */
  note: string
  /** hue used for the part when nothing is selected */
  color: string
  /** render as a translucent enclosure so the parts inside stay visible */
  ghost?: boolean
}

// Mirrors the six collections in cad/blender/pod_v4/build_v4.py, so the layer list
// and the Blender scene cannot disagree.
export const SYSTEMS: System[] = [
  { id: 'stator', name: 'Stator', note: 'Bolted to the tripod. Never turns.', color: '#7E8C97' },
  { id: 'rotor', name: 'Rotating assembly', note: 'Everything the servo pans.', color: '#2E3A42' },
  { id: 'portrait', name: 'Portrait attachment', note: 'Cradle and phone, upright.', color: '#B9BEC2' },
  { id: 'landscape', name: 'Landscape attachment', note: 'The same cradle, turned 90°.', color: '#B9BEC2' },
  { id: 'hardware', name: 'Purchased hardware', note: 'The parts you buy, not print.', color: '#3C4148' },
  { id: 'external', name: 'Tripod + leg case', note: 'Context, and the controller box.', color: '#6E7A84' },
]

export interface Preset {
  id: string
  name: string
  /** the question this preset answers */
  question: string
  systems: SystemId[]
  /** optional explicit part allowlist within those systems */
  only?: string[]
  explode?: number
}

export const PRESETS: Preset[] = [
  {
    id: 'assembled',
    name: 'Assembled',
    question: 'What does it look like built?',
    systems: ['stator', 'rotor', 'portrait', 'hardware', 'external'],
    explode: 0,
  },
  {
    id: 'printed',
    name: 'What you print',
    question: 'Which parts come off the printer?',
    systems: ['stator', 'rotor', 'portrait', 'landscape'],
    only: [
      '01-stator-shell', '02-nut-retainer', '03-servo-carrier-mg996r',
      '04-rotor', '05-drive-dog-42.9-47.6', '05b-drive-dog-short-42.9',
      '06-cradle-portrait', '07-cradle-landscape',
    ],
    explode: 1,
  },
  {
    id: 'legcase',
    name: 'Leg case',
    question: 'What lives on the tripod leg?',
    systems: ['external'],
    only: ['08-uno-tray', '09-uno-lid', '10-leg-saddle', '11-aa-bay', 'uno-r3', 'usb'],
    explode: 0.6,
  },
  {
    id: 'loadpath',
    name: 'Load path',
    question: 'Where does the phone’s weight actually go?',
    systems: ['stator', 'rotor', 'portrait', 'external'],
    only: [
      'phone', 'screen', '06-cradle-portrait', '04-rotor',
      '01-stator-shell', '02-nut-retainer', '1-4-20',
      'crown', 'leg', 'leg.001', 'leg.002',
    ],
    explode: 0.35,
  },
  {
    id: 'drive',
    name: 'Drive train',
    question: 'How does servo rotation become camera pan?',
    systems: ['stator', 'rotor', 'hardware'],
    only: ['mg996r-body', 'spline', '05-drive-dog-42.9-47.6', '04-rotor', '03-servo-carrier-mg996r'],
    explode: 0.8,
  },
]

/** The eight steps from the README, in order, each naming the parts it touches. */
export interface AssemblyStep {
  n: number
  text: string
  parts: string[]
  explode: number
}

export const ASSEMBLY: AssemblyStep[] = [
  { n: 1, text: 'Press the 1/4-20 nut up into the hex pocket in the underside of 01, then press 02 in below it, flush with the seating face.', parts: ['01-stator-shell', '02-nut-retainer', '1-4-20'], explode: 0.7 },
  { n: 2, text: 'Bolt the MG996R to 03 with four M3, servo body hanging down through the plate, flange resting on top.', parts: ['03-servo-carrier-mg996r', 'mg996r-body', 'spline'], explode: 0.7 },
  { n: 3, text: 'Drop 04 over the Ø88 journal on 01. It lands on the thrust land.', parts: ['04-rotor', '01-stator-shell'], explode: 0.5 },
  { n: 4, text: 'Screw 03 down onto the journal top with four M3. The carrier plate is also the capture ring — its Ø96 rim traps the rotor’s Ø89 shoulder with about 1 mm of lift play.', parts: ['03-servo-carrier-mg996r', '04-rotor', '01-stator-shell'], explode: 0.3 },
  { n: 5, text: 'Centre the servo at 90° before fitting 05. Blade into the rotor deck slot at r = 18 mm.', parts: ['05-drive-dog-42.9-47.6', 'spline', '04-rotor'], explode: 0.45 },
  { n: 6, text: 'Route the servo lead out the exit slot at az 210°, zip-tie it through the two tie slots, and run it down to the leg case.', parts: ['01-stator-shell', 'mg996r-body'], explode: 0.15 },
  { n: 7, text: 'Bolt 10 to the underside of 08 with two M3 × 16, strap the saddle to a leg, seat the Uno on its four pedestals, clip 09 on. 11 straps on separately.', parts: ['08-uno-tray', '09-uno-lid', '10-leg-saddle', '11-aa-bay', 'uno-r3'], explode: 0.55 },
  { n: 8, text: 'Slide a cradle tongue into the deck socket, lock with the M4 thumbscrew, and mount the whole pod on the MACTREM’s 1/4-20.', parts: ['06-cradle-portrait', '04-rotor', '1-4-20', 'phone'], explode: 0.25 },
]
