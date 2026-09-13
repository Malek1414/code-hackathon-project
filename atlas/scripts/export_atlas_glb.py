"""Export the FollowCam Pod v4 assembly as a browser-ready GLB + parts manifest.

Reuses cad/blender/pod_v4/build_v4.py as the single source of geometric truth:
this script never re-models anything, it imports build() and serialises the
result.  Run from the repo root:

    blender --background --python atlas/scripts/export_atlas_glb.py

Outputs
    atlas/public/models/followcam_pod_v4.glb   named node per part, metres
    atlas/src/generated/parts.geometry.json    system, explode vector, bbox, volume
"""
import json
import math
import runpy
import sys
from pathlib import Path

import bpy
from mathutils import Vector

REPO = Path(__file__).resolve().parents[2]
BUILD = REPO / 'cad' / 'blender' / 'pod_v4' / 'build_v4.py'
GLB = REPO / 'atlas' / 'public' / 'models' / 'followcam_pod_v4.glb'
MANIFEST = REPO / 'atlas' / 'src' / 'generated' / 'parts.geometry.json'

# Objects that exist for rendering only and carry no product meaning.
SKIP = {'ground', 'key', 'fill', 'rim', 'cam', 'PAN', 'leg case rig'}


def node_name(name):
    """A name that survives the round trip into three.js.

    GLTFLoader runs every node name through PropertyBinding.sanitizeNodeName, which
    strips the reserved characters [ ] . : / entirely. Blender's own names use dots
    for duplicates ('leg.001') and slashes for threads ('1/4-20'), so left alone they
    arrive in the browser as 'leg001' and '1420' and never match the manifest.
    Renaming before export keeps the GLB and the manifest in agreement.
    """
    out = []
    for ch in name:
        out.append(ch if (ch.isalnum() or ch in '_-') else '-')
    n = ''.join(out)
    while '--' in n:
        n = n.replace('--', '-')
    return n.strip('-')


def slugify(name):
    out = []
    for ch in name.lower():
        out.append(ch if (ch.isalnum() or ch == '.') else '-')
    slug = ''.join(out)
    while '--' in slug:
        slug = slug.replace('--', '-')
    return slug.strip('-')


def main():
    # build_v4.py guards its own main() behind __name__ == '__main__', so running
    # it under any other name gives us the module namespace without side effects.
    ns = runpy.run_path(str(BUILD), run_name='followcam_build_v4')
    build = ns['build']
    parts_registry = ns['PARTS']
    orient = ns['ORIENT']

    scene, cam, pivot, rotor, c_port, c_land, c_stat, c_rot, c_hw, c_ext = build()

    systems = {
        'stator': c_stat,
        'rotor': c_rot,
        'portrait': c_port,
        'landscape': c_land,
        'hardware': c_hw,
        'external': c_ext,
    }
    obj_system = {}
    for key, coll in systems.items():
        for o in coll.objects:
            obj_system[o.name] = key

    # build() already converted the scene to metres; explode vectors in PARTS are
    # still in millimetres because they are authored alongside the mm geometry.
    explode = {o.name: Vector(v) for o, v in parts_registry}

    # main()'s exploded shot adds two rules on top of the PARTS registry: the phone
    # rides clear of its cradle, and the servo swings out sideways. Mirror them so
    # the atlas explodes the same way the render does, and cover landscape too.
    for coll, prefix in ((c_port, '06_'), (c_land, '07_')):
        for o in coll.objects:
            if not o.name.startswith(prefix):
                explode.setdefault(o.name, Vector((0, 0, 110)))
    for o in c_hw.objects:
        explode.setdefault(o.name, Vector((0, 98, 18)))

    # Everything still unassigned is the leg case or tripod context: it is bolted to
    # a leg, not stacked on the pan axis, so it stays put rather than exploding into
    # the pod's column.

    records = []
    for o in list(scene.objects):
        if o.type != 'MESH' or o.name in SKIP:
            continue
        was_hidden = o.hide_render or o.hide_viewport
        # hide_viewport excludes the object from the depsgraph entirely, which leaves
        # matrix_world stale from before build()'s millimetre-to-metre conversion.
        # Clear it and flush before measuring, or the part reads 1000x oversized.
        o.hide_render = False
        o.hide_viewport = False
        o.hide_set(False)
        bpy.context.view_layer.update()

        deps = bpy.context.evaluated_depsgraph_get()
        ev = o.evaluated_get(deps)
        mesh = ev.to_mesh()
        mw = o.matrix_world
        pts = [mw @ v.co for v in mesh.vertices]
        tri_count = len(mesh.loop_triangles) or sum(max(0, len(p.vertices) - 2) for p in mesh.polygons)
        ev.to_mesh_clear()

        if not pts:
            continue
        lo = Vector((min(p.x for p in pts), min(p.y for p in pts), min(p.z for p in pts)))
        hi = Vector((max(p.x for p in pts), max(p.y for p in pts), max(p.z for p in pts)))

        ex = explode.get(o.name, Vector((0, 0, 0)))
        records.append({
            'id': slugify(o.name),
            'object': o.name,
            'node': node_name(o.name),
            'system': obj_system.get(o.name, 'external'),
            # Human-facing numbers stay in Blender's Z-up millimetres so they match
            # the README's bed-footprint table. 'explodeGl' is the same vector in the
            # GLB's Y-up metres, because export_yup=True rotates the geometry:
            # Blender (x, y, z) -> glTF (x, z, -y).
            'explode': [ex.x * .001, ex.y * .001, ex.z * .001],
            'explodeGl': [ex.x * .001, ex.z * .001, -ex.y * .001],
            'bboxMin': [lo.x, lo.y, lo.z],
            'bboxMax': [hi.x, hi.y, hi.z],
            'sizeMm': [(hi.x - lo.x) * 1000, (hi.y - lo.y) * 1000, (hi.z - lo.z) * 1000],
            'centerM': [(lo.x + hi.x) / 2, (lo.y + hi.y) / 2, (lo.z + hi.z) / 2],
            'triangles': tri_count,
            'printed': o.name in orient,
            'printOrientation': orient.get(o.name, (None,))[0],
            'defaultHidden': bool(was_hidden),
            'material': o.data.materials[0].name if o.data.materials else None,
        })

    records.sort(key=lambda r: r['object'])

    seen = {}
    for r in records:
        if r['node'] in seen:
            raise SystemExit(f"node name collision: {r['object']!r} and {seen[r['node']]!r} "
                             f"both sanitise to {r['node']!r}")
        seen[r['node']] = r['object']

    # Rename last, so everything above reads the original Blender names.
    for r in records:
        bpy.data.objects[r['object']].name = r['node']

    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    GLB.parent.mkdir(parents=True, exist_ok=True)
    MANIFEST.write_text(json.dumps(records, indent=2) + '\n')

    bpy.ops.object.select_all(action='DESELECT')
    for o in scene.objects:
        if o.type == 'MESH' and o.name not in SKIP:
            o.select_set(True)

    kw = dict(filepath=str(GLB), export_format='GLB', use_selection=True,
              export_apply=True, export_yup=True)
    for optional in ({'export_materials': 'EXPORT'}, {}):
        try:
            bpy.ops.export_scene.gltf(**kw, **optional)
            break
        except TypeError:
            continue

    print(f'[atlas] {len(records)} parts -> {GLB.name} ({GLB.stat().st_size // 1024} KB)')
    print(f'[atlas] manifest -> {MANIFEST.relative_to(REPO)}')
    for r in records:
        print(f"    {r['system']:<10} {r['object']:<28} {r['triangles']:>6} tris"
              f"{'  (printed)' if r['printed'] else ''}")


main()
