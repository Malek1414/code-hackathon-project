"""Build block-schematic geometry for the two head-mounted displays in the
Charite / Google accessibility hackathon project, and export them the same way
the FollowCam pod is exported.

IMPORTANT - what this geometry is and is not.

The FollowCam pod's atlas geometry comes from the CAD that produces its printable
STLs, so it is dimensionally exact. Nothing equivalent exists for a Quest 3 or a
RayNeo X3 Pro: there is no public CAD, and a teardown photo is not a mesh. So the
parts below are BLOCK SCHEMATICS - each component is a primitive sized and placed
to be roughly right and topologically honest about what sits where and what it
connects to. Outside envelopes and component counts follow published figures; the
internal placements are representative, not measured. The atlas labels them as
such, and no number is presented as a measurement unless it is sourced.

    blender --background --python atlas/scripts/build_glasses.py
"""
import json
import math
from pathlib import Path

import bpy
from mathutils import Vector

OUT = Path(__file__).resolve().parents[1]
MODELS = OUT / 'public' / 'models'
GEN = OUT / 'src' / 'generated'

PARTS = []   # (object, system, explode_mm)


# ------------------------------------------------------------------ helpers
def reset():
    bpy.ops.object.select_all(action='SELECT')
    bpy.ops.object.delete(use_global=False)
    for block in (bpy.data.meshes, bpy.data.materials):
        for b in list(block):
            if b.users == 0:
                block.remove(b)
    PARTS.clear()


def mat(name, rgb, rough=0.42, metal=0.0):
    m = bpy.data.materials.get(name)
    if m:
        return m
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes['Principled BSDF']
    b.inputs['Base Color'].default_value = (*rgb, 1)
    b.inputs['Roughness'].default_value = rough
    b.inputs['Metallic'].default_value = metal
    return m


def box(name, size, at, m, system, explode=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=at)
    o = bpy.context.object
    o.scale = Vector(size)
    bpy.ops.object.transform_apply(scale=True)
    return _reg(o, name, m, system, explode)


def cyl(name, d, h, at, m, system, explode=(0, 0, 0), axis='Z', n=48):
    bpy.ops.mesh.primitive_cylinder_add(radius=d / 2, depth=h, location=at, vertices=n)
    o = bpy.context.object
    if axis == 'X':
        o.rotation_euler = (0, math.pi / 2, 0)
    elif axis == 'Y':
        o.rotation_euler = (math.pi / 2, 0, 0)
    bpy.ops.object.transform_apply(rotation=True)
    return _reg(o, name, m, system, explode)


def _reg(o, name, m, system, explode):
    o.name = name
    o.data.materials.clear()
    o.data.materials.append(m)
    PARTS.append((o, system, Vector(explode)))
    return o


def node_name(name):
    out = [c if (c.isalnum() or c in '_-') else '-' for c in name]
    n = ''.join(out)
    while '--' in n:
        n = n.replace('--', '-')
    return n.strip('-')


def slugify(name):
    n = ''.join(c if c.isalnum() else '-' for c in name.lower())
    while '--' in n:
        n = n.replace('--', '-')
    return n.strip('-')


# ------------------------------------------------------------------ Quest 3
def build_quest3():
    """Meta Quest 3. Outer envelope ~184 x 160 x 98 mm with the facial interface;
    the visor itself is the slim part, because pancake optics fold the light path."""
    shell = mat('Quest polymer', (.10, .11, .13), .48)
    soft = mat('Facial interface foam', (.20, .21, .24), .82)
    glass = mat('Optical stack', (.06, .12, .20), .10)
    lens = mat('Pancake lens', (.55, .70, .82), .06)
    panel = mat('LCD panel', (.85, .87, .90), .30)
    lamp = mat('LED backlight', (.95, .72, .30), .35)
    silicon = mat('Silicon', (.16, .18, .22), .40, .5)
    cell = mat('Li-ion cell', (.30, .26, .18), .45)
    metalm = mat('Metal', (.52, .57, .62), .25, .85)

    # chassis ------------------------------------------------------------
    box('visor shell', (184, 52, 82), (0, 0, 0), shell, 'chassis')
    box('facial interface', (170, 46, 78), (0, -49, -4), soft, 'chassis', (0, -70, 0))
    box('head strap', (188, 8, 10), (0, -20, 58), shell, 'chassis', (0, 0, 60))

    # optics + display ---------------------------------------------------
    for side, x in (('L', -32), ('R', 32)):
        cyl(f'pancake lens {side}', 44, 9, (x, -20, 0), lens, 'optics', (0, -34, 0), axis='Y')
        cyl(f'optical stack {side}', 46, 4, (x, -13, 0), glass, 'optics', (0, -20, 0), axis='Y')
        box(f'LCD panel {side}', (40, 2.4, 43), (x, -4, 0), panel, 'display', (0, 14, 0))
        box(f'LED backlight {side}', (40, 3.0, 43), (x, 0, 0), lamp, 'display', (0, 34, 0))
    box('IPD motor', (26, 12, 10), (0, -8, -30), metalm, 'optics', (0, 0, -34))

    # sensing ------------------------------------------------------------
    for side, x in (('left', -66), ('right', 66)):
        box(f'camera pill {side}', (26, 7, 44), (x, 27, 4), glass, 'sensing', (x * .5, 40, 0))
    box('camera pill centre', (24, 7, 40), (0, 27, 4), glass, 'sensing', (0, 40, 0))
    cyl('RGB passthrough camera L', 9, 6, (-66, 29, 14), glass, 'sensing', (34, 62, -14), axis='Y')
    cyl('RGB passthrough camera R', 9, 6, (66, 29, 14), glass, 'sensing', (-34, 62, -14), axis='Y')
    for i, (x, z) in enumerate(((-66, -8), (66, -8), (-8, 16), (8, 16))):
        cyl(f'IR tracking camera {i + 1}', 7, 6, (x, 29, z), glass, 'sensing', (x * .6, 52, z), axis='Y')
    cyl('depth projector', 11, 7, (0, 29, -8), lamp, 'sensing', (0, 56, -12), axis='Y')

    # compute + power + audio -------------------------------------------
    box('Snapdragon XR2 Gen 2', (18, 3, 18), (0, 12, 22), silicon, 'compute', (0, 34, 34))
    box('LPDDR5 8 GB', (12, 2.4, 12), (24, 12, 22), silicon, 'compute', (34, 34, 34))
    box('mainboard', (120, 2, 46), (0, 14, 10), mat('PCB', (.05, .20, .16), .5), 'compute', (0, 44, 18))
    box('battery', (92, 14, 30), (0, 16, -26), cell, 'power', (0, 30, -52))
    for side, x in (('L', -80), ('R', 80)):
        box(f'speaker {side}', (16, 12, 10), (x, -6, 30), shell, 'audio', (x * .4, 0, 46))

    return 'quest3'


# ------------------------------------------------------------- RayNeo X3 Pro
def build_x3pro():
    """RayNeo X3 Pro. A 76 g titanium-frame pair of glasses: the optics are
    waveguides in the lenses, so there is no visor volume to put anything in -
    everything lives in the temples."""
    ti = mat('Titanium frame', (.46, .47, .50), .34, .80)
    wg = mat('Waveguide', (.58, .74, .80), .08)
    engine = mat('Firefly optical engine', (.88, .55, .18), .30)
    silicon = mat('Silicon', (.16, .18, .22), .40, .5)
    cell = mat('Li-ion cell', (.30, .26, .18), .45)
    glass = mat('Optical stack', (.06, .12, .20), .10)
    shell = mat('Temple shell', (.12, .13, .15), .46)

    # chassis ------------------------------------------------------------
    box('frame front', (146, 9, 16), (0, 0, 22), ti, 'chassis')
    box('bridge', (18, 8, 10), (0, 0, 14), ti, 'chassis', (0, 0, 22))
    for side, x in (('L', -63), ('R', 63)):
        box(f'temple {side}', (10, 128, 12), (x, -66, 20), shell, 'chassis', (x * .6, -40, 26))
        box(f'waveguide {side}', (46, 2.2, 30), (x * .55, 0, 6), wg, 'optics', (x * 1.1, 0, -16))
        box(f'microLED engine {side}', (13, 13, 12), (x * .86, -3, 22), engine, 'display', (x * .95, 0, 34))
        box(f'battery {side}', (8, 44, 10), (x, -104, 20), cell, 'power', (x * .5, -70, 30))
        box(f'speaker {side}', (7, 12, 8), (x, -116, 12), shell, 'audio', (x * .5, -80, -14))
        cyl(f'SLAM camera {side}', 6, 5, (x * .82, 4, 22), glass, 'sensing', (x, 30, 26), axis='Y')

    # sensing + compute --------------------------------------------------
    cyl('RGB camera 12 MP', 9, 6, (-52, 4, 22), glass, 'sensing', (-30, 34, 30), axis='Y')
    box('Snapdragon AR1 Gen 1', (11, 11, 8), (63, -34, 20), silicon, 'compute', (40, -10, 34))
    box('mainboard', (8, 56, 10), (63, -44, 20), mat('PCB', (.05, .20, .16), .5), 'compute', (44, -20, 26))
    for i, x in enumerate((-30, 30)):
        cyl(f'microphone {i + 1}', 3.5, 3, (x, 3, 26), shell, 'sensing', (x, 26, 34), axis='Y')

    return 'x3pro'


# -------------------------------------------------------------------- export
def export(device):
    scene = bpy.context.scene
    scene.unit_settings.system = 'METRIC'

    for o in list(scene.objects):
        o.location *= .001
        o.scale *= .001
    bpy.context.view_layer.update()

    records = []
    for o, system, ex in PARTS:
        deps = bpy.context.evaluated_depsgraph_get()
        m = o.evaluated_get(deps).to_mesh()
        mw = o.matrix_world
        pts = [mw @ v.co for v in m.vertices]
        tris = len(m.loop_triangles) or sum(max(0, len(p.vertices) - 2) for p in m.polygons)
        o.evaluated_get(deps).to_mesh_clear()
        lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))
        records.append({
            'id': slugify(o.name),
            'object': o.name,
            'node': node_name(o.name),
            'system': system,
            'explode': [ex.x * .001, ex.y * .001, ex.z * .001],
            'explodeGl': [ex.x * .001, ex.z * .001, -ex.y * .001],
            'bboxMin': [lo.x, lo.y, lo.z],
            'bboxMax': [hi.x, hi.y, hi.z],
            'sizeMm': [(hi.x - lo.x) * 1000, (hi.y - lo.y) * 1000, (hi.z - lo.z) * 1000],
            'centerM': [(lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2],
            'triangles': tris,
            'printed': False,
            'printOrientation': None,
            'defaultHidden': False,
            'material': o.data.materials[0].name if o.data.materials else None,
            'schematic': True,
        })

    seen = {}
    for r in records:
        if r['node'] in seen:
            raise SystemExit(f"node collision: {r['object']} / {seen[r['node']]}")
        seen[r['node']] = r['object']
    for r in records:
        bpy.data.objects[r['object']].name = r['node']

    records.sort(key=lambda r: (r['system'], r['object']))
    MODELS.mkdir(parents=True, exist_ok=True)
    GEN.mkdir(parents=True, exist_ok=True)
    glb = MODELS / f'{device}.glb'

    bpy.ops.object.select_all(action='DESELECT')
    for o, _, _ in PARTS:
        o.select_set(True)
    bpy.ops.export_scene.gltf(filepath=str(glb), export_format='GLB',
                              use_selection=True, export_apply=True, export_yup=True)

    (GEN / f'{device}.geometry.json').write_text(json.dumps(records, indent=2) + '\n')
    print(f'[atlas] {device}: {len(records)} parts -> {glb.name} ({glb.stat().st_size // 1024} KB)')
    return records


for builder in (build_quest3, build_x3pro):
    reset()
    export(builder())
