import type { GeometryRecord } from '../viewer/Viewer'
import { BY_ID as FOLLOWCAM_BY_ID, PARTS as FOLLOWCAM_PARTS, type PartInfo, type SystemId } from './parts'
import { ASSEMBLY, PRESETS as FOLLOWCAM_PRESETS, SYSTEMS as FOLLOWCAM_SYSTEMS, type AssemblyStep, type Preset, type System } from './systems'
import { QUEST3_PARTS, QUEST3_PRESETS, QUEST3_SYSTEMS } from './quest3'
import { X3PRO_PARTS, X3PRO_PRESETS, X3PRO_SYSTEMS } from './x3pro'

import followcamGeometry from '../generated/parts.geometry.json'
import quest3Geometry from '../generated/quest3.geometry.json'
import x3proGeometry from '../generated/x3pro.geometry.json'

export interface Device {
  id: string
  name: string
  /** the tab label */
  short: string
  /** what this device is *for*, in this project */
  role: string
  model: string
  geometry: GeometryRecord[]
  parts: PartInfo[]
  byId: Record<string, PartInfo>
  systems: System[]
  presets: Preset[]
  assembly?: AssemblyStep[]
  defaultSystems: SystemId[]
  /** true when the geometry is a block schematic rather than the real CAD */
  schematic: boolean
  provenance: string
}

const index = (parts: PartInfo[]) => Object.fromEntries(parts.map((p) => [p.id, p]))

export const DEVICES: Device[] = [
  {
    id: 'followcam',
    name: 'FollowCam Pod v4',
    short: 'FollowCam',
    role: 'The direct-drive pan pod that replaces a tripod head. Twelve printed parts and a hobby servo.',
    model: 'models/followcam_pod_v4.glb',
    geometry: followcamGeometry as unknown as GeometryRecord[],
    parts: FOLLOWCAM_PARTS,
    byId: FOLLOWCAM_BY_ID,
    systems: FOLLOWCAM_SYSTEMS,
    presets: FOLLOWCAM_PRESETS,
    assembly: ASSEMBLY,
    defaultSystems: ['stator', 'rotor', 'portrait', 'hardware', 'external'],
    schematic: false,
    provenance:
      'Geometry is exported straight from cad/blender/pod_v4/build_v4.py — the same script that writes the printable STLs. Every dimension shown is measured from that model.',
  },
  {
    id: 'quest3',
    name: 'Meta Quest 3',
    short: 'Quest 3',
    role: 'The hackathon device. Video passthrough means software can darken the real world — the only device here that can.',
    model: 'models/quest3.glb',
    geometry: quest3Geometry as unknown as GeometryRecord[],
    parts: QUEST3_PARTS,
    byId: index(QUEST3_PARTS),
    systems: QUEST3_SYSTEMS,
    presets: QUEST3_PRESETS,
    defaultSystems: ['optics', 'display', 'sensing', 'compute', 'power', 'audio', 'chassis'],
    schematic: true,
    provenance:
      'Block schematic. Component counts, display, SoC and optics figures follow published specifications and the teardown record; internal placement is representative, not measured. Numbers that matter are sourced on the part itself.',
  },
  {
    id: 'x3pro',
    name: 'RayNeo X3 Pro',
    short: 'X3 Pro',
    role: 'The roadmap device. 76 g of titanium that can show a cue and an alert — but cannot dim anything.',
    model: 'models/x3pro.glb',
    geometry: x3proGeometry as unknown as GeometryRecord[],
    parts: X3PRO_PARTS,
    byId: index(X3PRO_PARTS),
    systems: X3PRO_SYSTEMS,
    presets: X3PRO_PRESETS,
    defaultSystems: ['optics', 'display', 'sensing', 'compute', 'power', 'audio', 'chassis'],
    schematic: true,
    provenance:
      'Block schematic. Display, optics, SoC and mass figures follow RayNeo’s published specifications and independent reviews; internal placement is representative, not measured.',
  },
]

export const DEVICE_BY_ID = Object.fromEntries(DEVICES.map((d) => [d.id, d]))
