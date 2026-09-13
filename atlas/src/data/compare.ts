/**
 * The layer that actually answers the project question: which capability survives
 * the move from the hackathon device to the roadmap device?
 *
 * `verdict` is about this project's needs, not about which product is better.
 */
export type Verdict = 'quest' | 'rayneo' | 'even'

export interface CompareRow {
  system: string
  property: string
  quest: string
  rayneo: string
  verdict: Verdict
  /** why this row matters for an epilepsy accessibility device */
  soWhat: string
}

export const COMPARE: CompareRow[] = [
  {
    system: 'Optics', property: 'Can it darken the real world?',
    quest: 'Yes — video passthrough',
    rayneo: 'No — additive waveguide',
    verdict: 'quest',
    soWhat:
      'The whole intervention is “dim the flashing thing in front of the wearer”. A waveguide can only add light to what the eye already sees, so on the X3 Pro that intervention does not exist at any brightness. This single row is why the Quest is the hackathon device.',
  },
  {
    system: 'Display', property: 'Refresh rate',
    quest: '72 / 80 / 90 / 120 Hz, any integer 72–207 via API',
    rayneo: '60 Hz, fixed',
    verdict: 'quest',
    soWhat:
      'The backlight or PWM strobe sits at the refresh rate. Photoparoxysmal responses peak around 15–25 Hz and the guidelines flag 3–60 Hz. The Quest can be pushed far clear of that band and held there; the X3 Pro sits at its upper edge with no way to move.',
  },
  {
    system: 'Display', property: 'Emission and dimming',
    quest: 'LCD + strobed LED backlight, ≈ 100 nits peak',
    rayneo: 'microLED, 3,500 nits typical / 6,000 peak',
    verdict: 'quest',
    soWhat:
      'Low peak brightness is protective: it caps how bright any flash can be. The X3 Pro is 35–60× brighter and normally dims by pulse-width modulation, which deepens the modulation rather than reducing it — so turning it down may make flicker worse, not better. That PWM frequency is unpublished and needs measuring.',
  },
  {
    system: 'Sensing', property: 'Cameras in the wearer’s visual path',
    quest: '2 × 4 MP RGB passthrough (live, re-displayed)',
    rayneo: '1 × 12 MP, capture only',
    verdict: 'rayneo',
    soWhat:
      'The Quest’s advantage costs it a second flicker path: a rolling shutter sampling 50 Hz mains lighting can beat against it and produce banding in passthrough that is not there to the naked eye. Lock exposure to a multiple of 10 ms. The X3 Pro never re-displays what it captures, so it has no such path.',
  },
  {
    system: 'Compute', property: 'SoC and headroom',
    quest: 'Snapdragon XR2 Gen 2, 8 GB',
    rayneo: 'Snapdragon AR1 Gen 1, 4 GB',
    verdict: 'quest',
    soWhat:
      'Per-frame shader filtering of the whole visual field is cheap on XR2 and not obviously affordable on AR1. On the roadmap device, protection has to come from what is drawn, not from reprocessing the view.',
  },
  {
    system: 'Chassis', property: 'Mass and wearability',
    quest: '≈ 515 g, strap, sealed facial interface',
    rayneo: '76 g, titanium, no strap',
    verdict: 'rayneo',
    soWhat:
      'The pitch’s own argument is that monitoring devices get abandoned for comfort, not accuracy — about a third of device owners in the Charité SUDEP cohort had stopped, false alarms first among the reasons. A 515 g sealed headset is not an all-day device. 76 g of glasses is. That is the whole reason for the roadmap.',
  },
  {
    system: 'Power', property: 'Runtime',
    quest: '≈ 2–3 h mixed reality',
    rayneo: 'Short; reviewers have been blunt about it',
    verdict: 'even',
    soWhat:
      'Neither device currently runs a full day, and an always-on arousal-sensing layer needs one. This is the honest open problem on both sides, and it is worth saying so on stage before someone else does.',
  },
  {
    system: 'Audio', property: 'Does it block the room?',
    quest: 'Open-ear speakers',
    rayneo: 'Open-ear speakers',
    verdict: 'even',
    soWhat:
      'Both keep the wearer audible to, and able to hear, whoever is with them. For a device whose escalation path ends in a caregiver or an emergency call, that is a requirement, not a feature.',
  },
]
