"""Generate the FollowCam Pod v3 Blender assembly, renders, and print STLs.

Run from the repository root:
    blender --background --python cad/blender/followcam_pod_v3.py

All modelling dimensions are millimetres. The model is intentionally
parameterised near the top of this file so a measured phone envelope can replace
the conservative universal placeholder before printing.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

import bpy
from mathutils import Vector


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "cad" / "blender"
EXPORT = OUT / "exports"
RENDER = OUT / "renders"
OUT.mkdir(parents=True, exist_ok=True)
EXPORT.mkdir(parents=True, exist_ok=True)
RENDER.mkdir(parents=True, exist_ok=True)

# Core design envelope (mm)
BASE_D = 112.0
LOWER_H = 22.0
UPPER_BOTTOM = 22.0
UPPER_TOP = 66.0
ROTOR_BOTTOM = 70.5
ROTOR_H = 10.0
PHONE_W = 78.5          # conservative case-on phone width
PHONE_H = 165.0         # conservative case-on phone height
PHONE_T = 11.5          # conservative case-on phone thickness
PHONE_CLEARANCE = 0.8
CASE_WALL = 2.8
CASE_BACK = 2.6


def clean_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)
    for datablocks in (bpy.data.meshes, bpy.data.curves, bpy.data.materials, bpy.data.cameras, bpy.data.lights):
        pass


def collection(name: str) -> bpy.types.Collection:
    col = bpy.data.collections.get(name)
    if col is None:
        col = bpy.data.collections.new(name)
        bpy.context.scene.collection.children.link(col)
    return col


def move_to_collection(obj: bpy.types.Object, col: bpy.types.Collection) -> None:
    for old in list(obj.users_collection):
        old.objects.unlink(obj)
    col.objects.link(obj)


def material(name: str, color: tuple[float, float, float, float], metallic=0.0, roughness=0.45):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = color
    mat.use_nodes = True
    bsdf = mat.node_tree.nodes.get("Principled BSDF")
    bsdf.inputs["Base Color"].default_value = color
    bsdf.inputs["Metallic"].default_value = metallic
    bsdf.inputs["Roughness"].default_value = roughness
    return mat


def assign(obj, mat):
    if mat is not None:
        obj.data.materials.clear()
        obj.data.materials.append(mat)
    return obj


def apply_modifier(obj, modifier):
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)
    bpy.ops.object.modifier_apply(modifier=modifier.name)
    obj.select_set(False)


def bevel(obj, amount=1.5, segments=3):
    mod = obj.modifiers.new("manufacturing fillets", "BEVEL")
    mod.width = amount
    mod.segments = segments
    mod.limit_method = "ANGLE"
    apply_modifier(obj, mod)
    return obj


def rounded_box(name, dims, loc=(0, 0, 0), radius=2.0, mat=None, col=None):
    bpy.ops.mesh.primitive_cube_add(location=loc)
    obj = bpy.context.object
    obj.name = name
    obj.dimensions = dims
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if radius > 0:
        bevel(obj, min(radius, min(dims) * 0.45), 4)
    assign(obj, mat)
    if col:
        move_to_collection(obj, col)
    return obj


def cylinder(name, radius, depth, loc=(0, 0, 0), vertices=96, mat=None, col=None, edge=0.0):
    bpy.ops.mesh.primitive_cylinder_add(vertices=vertices, radius=radius, depth=depth, location=loc)
    obj = bpy.context.object
    obj.name = name
    if edge:
        bevel(obj, edge, 3)
    assign(obj, mat)
    if col:
        move_to_collection(obj, col)
    return obj


def boolean(target, cutter, operation="DIFFERENCE"):
    mod = target.modifiers.new(f"{operation.lower()}_{cutter.name}", "BOOLEAN")
    mod.operation = operation
    mod.solver = "EXACT"
    mod.object = cutter
    apply_modifier(target, mod)
    bpy.data.objects.remove(cutter, do_unlink=True)
    return target


def join_objects(name, objects):
    bpy.ops.object.select_all(action="DESELECT")
    for obj in objects:
        obj.select_set(True)
    bpy.context.view_layer.objects.active = objects[0]
    bpy.ops.object.join()
    obj = bpy.context.object
    obj.name = name
    return obj


def duplicate(obj, name, col=None):
    dup = obj.copy()
    dup.data = obj.data.copy()
    dup.name = name
    (col or bpy.context.scene.collection).objects.link(dup)
    return dup


def set_smooth(obj):
    if obj.type == "MESH":
        for poly in obj.data.polygons:
            poly.use_smooth = True


def make_lower_shell(col, mat):
    outer = cylinder("01_BASE_LOWER", BASE_D / 2, LOWER_H, (0, 0, LOWER_H / 2), mat=mat, col=col, edge=2.4)
    inner = cylinder("lower cavity", BASE_D / 2 - 4.0, LOWER_H, (0, 0, 14.0), col=col)
    boolean(outer, inner)
    # Universal tripod interface: 1/4-20 through hole and captive nut/insert pocket.
    boolean(outer, cylinder("quarter-inch clearance", 3.4, 10, (0, 0, 3.0), col=col))
    boolean(outer, cylinder("quarter-inch insert pocket", 5.65, 4.2, (0, 0, 5.7), vertices=6, col=col))
    # USB-B access and a smaller power/strain-relief slot.
    boolean(outer, rounded_box("USB access", (18, 12, 11), (0, -54.0, 12.5), 2.0, col=col))
    boolean(outer, rounded_box("power access", (12, 10, 8), (27, -54.5, 11.0), 1.5, col=col))
    # Arduino Uno R3 standoffs: 68.6 x 53.4 board, 3.2 mm holes.
    standoff_xy = [(-31.0, -24.0), (31.0, -24.0), (-31.0, 24.0), (31.0, 24.0)]
    for i, (x, y) in enumerate(standoff_xy):
        post = cylinder(f"Uno standoff {i+1}", 3.4, 5.0, (x, y, 6.5), vertices=48, mat=mat, col=col)
        bore = cylinder(f"M3 pilot {i+1}", 1.35, 7.0, (x, y, 7.0), vertices=32, col=col)
        boolean(post, bore)
        boolean(outer, post, "UNION")
    outer.name = "01_BASE_LOWER"
    return outer


def make_upper_shell(col, mat):
    h = UPPER_TOP - UPPER_BOTTOM
    outer = cylinder("02_BASE_UPPER", BASE_D / 2, h, (0, 0, (UPPER_BOTTOM + UPPER_TOP) / 2), mat=mat, col=col, edge=2.4)
    inner = cylinder("upper cavity", BASE_D / 2 - 4.0, h - 3.2, (0, 0, UPPER_BOTTOM + (h - 3.2) / 2 - 0.1), col=col)
    boolean(outer, inner)
    # Servo coupler clearance and thrust-bearing seat.
    boolean(outer, cylinder("servo hub clearance", 5.0, 10.0, (0, 0, 63.0), col=col))
    boolean(outer, cylinder("51100 bearing seat", 12.25, 4.7, (0, 0, 64.0), col=col))
    # Four service screws on a square bolt pattern.
    for i, (x, y) in enumerate(((-39, -30), (39, -30), (-39, 30), (39, 30))):
        boolean(outer, cylinder(f"lid M3 {i+1}", 1.7, h + 4, (x, y, 44), vertices=32, col=col))
    # Vent bank and status LED window.
    for i, x in enumerate((-30, -20, -10, 0, 10, 20, 30)):
        boolean(outer, rounded_box(f"vent {i+1}", (5, 9, 2.8), (x, 55.0, 39), 1.2, col=col))
    boolean(outer, rounded_box("status light", (18, 5, 4), (0, -55.0, 51), 1.8, col=col))
    return outer


def make_servo_carrier(col, mat):
    plate = rounded_box("03_SERVO_CARRIER", (58.0, 36.0, 4.0), (0, 0, 2.0), 2.2, mat, col)
    boolean(plate, rounded_box("MG996R body window", (41.6, 20.7, 7.0), (10.2, 0, 2.0), 1.4, col=col))
    # MG996R flange holes; slotted slightly for clone tolerances.
    for i, (x, y) in enumerate(((-24.5, -5.4), (-24.5, 5.4), (24.5, -5.4), (24.5, 5.4))):
        boolean(plate, cylinder(f"servo M3 {i+1}", 1.7, 8, (x, y, 2), vertices=32, col=col))
    return plate


def make_rotor(col, mat, accent):
    rotor = cylinder("04_ROTATING_PLATTER", 44.0, ROTOR_H, (0, 0, ROTOR_BOTTOM + ROTOR_H / 2), mat=mat, col=col, edge=2.0)
    boolean(rotor, cylinder("upper bearing seat", 12.25, 4.7, (0, 0, ROTOR_BOTTOM + 1.7), col=col))
    boolean(rotor, cylinder("horn screw access", 2.2, ROTOR_H + 3, (0, 0, ROTOR_BOTTOM + ROTOR_H / 2), vertices=32, col=col))
    # Three horn attachment holes on a 16 mm PCD.
    for i, a in enumerate((0, 120, 240)):
        x, y = 8 * math.cos(math.radians(a)), 8 * math.sin(math.radians(a))
        boolean(rotor, cylinder(f"horn screw {i+1}", 1.25, ROTOR_H + 3, (x, y, ROTOR_BOTTOM + ROTOR_H / 2), vertices=32, col=col))
    # Keyed cradle receiver. It is centred, so either cradle keeps its CG on-axis.
    receiver = rounded_box("cradle receiver", (32, 22, 8), (0, 0, ROTOR_BOTTOM + ROTOR_H + 4), 2.5, accent, col)
    boolean(receiver, rounded_box("receiver tongue cavity", (26.4, 12.4, 8), (0, 0, ROTOR_BOTTOM + ROTOR_H + 6), 1.2, col=col))
    return rotor, receiver


def make_phone_case(name, phone_w, phone_h, col, mat, accent, bottom_z=91.0):
    outer_w = phone_w + 2 * CASE_WALL
    outer_h = phone_h + 2 * CASE_WALL
    outer_d = PHONE_T + CASE_BACK + 2.4
    center_z = bottom_z + outer_h / 2
    shell = rounded_box(name, (outer_w, outer_d, outer_h), (0, 0, center_z), 7.0, mat, col)
    cavity_y = -outer_d / 2 + CASE_BACK + (PHONE_T + PHONE_CLEARANCE) / 2
    cavity = rounded_box(
        f"{name} phone cavity",
        (phone_w + PHONE_CLEARANCE, PHONE_T + PHONE_CLEARANCE + 2.5, phone_h + PHONE_CLEARANCE),
        (0, cavity_y + 1.3, center_z),
        5.2,
        col=col,
    )
    boolean(shell, cavity)
    # Back camera window: deliberately generous for current multi-lens phones.
    camera_x = -phone_w * 0.29
    camera_z = center_z + phone_h * 0.32
    boolean(shell, rounded_box(f"{name} camera window", (39, outer_d + 4, 44), (camera_x, -outer_d / 2, camera_z), 7.0, col=col))
    # Charging/speaker access at the lower edge.
    boolean(shell, rounded_box(f"{name} charge access", (28, outer_d + 3, 9), (0, 1.0, bottom_z + 1), 3.2, col=col))
    # Four compliant front corner lips make this a true wraparound case rather than a clamp.
    front_y = outer_d / 2 + 0.75
    clips = []
    for sx in (-1, 1):
        for sz in (-1, 1):
            x = sx * (outer_w / 2 - 8.5)
            z = center_z + sz * (outer_h / 2 - 10.0)
            clip = rounded_box(f"{name} retention clip", (17, 2.8, 20), (x, front_y, z), 3.0, accent, col)
            clips.append(clip)
    # Keyed tongue, centred below the phone for equal portrait/landscape balance.
    tongue = rounded_box(f"{name} keyed tongue", (26.0, 12.0, 10.0), (0, 0, bottom_z - 5.0), 1.8, accent, col)
    boolean(tongue, cylinder(f"{name} tongue M3", 1.65, 18, (0, 0, bottom_z - 5), vertices=32, col=col))
    return shell, clips, tongue


def make_phone_dummy(name, w, h, bottom_z, col, glass, metal):
    d = PHONE_T
    body = rounded_box(name, (w, d, h), (0, 0.8, bottom_z + h / 2), 6.5, metal, col)
    screen = rounded_box(f"{name} screen", (w - 4.0, 0.8, h - 4.0), (0, d / 2 + 0.9, bottom_z + h / 2), 5.3, glass, col)
    return [body, screen]


def make_electronics(col, pcb, metal, black, accent):
    parts = []
    # Arduino Uno R3 placeholder: exact board plan dimensions and hole pattern.
    board = rounded_box("Arduino Uno R3 — 68.6 x 53.4", (68.6, 53.4, 1.6), (0, 0, 10.0), 1.8, pcb, col)
    parts.append(board)
    for x, y in ((-31, -24), (31, -24), (-31, 24), (31, 24)):
        hole = cylinder("Uno mounting hole", 1.6, 3, (x, y, 10), vertices=32, col=col)
        boolean(board, hole)
    parts += [
        rounded_box("Uno USB-B", (16, 12, 11), (0, -22.0, 16.3), 1.4, metal, col),
        rounded_box("Uno DC jack", (14, 9, 11), (23, -21.5, 16.3), 1.4, black, col),
        rounded_box("ATmega328P", (35, 10, 4), (3, 6, 13), 1.0, black, col),
    ]
    # MG996R-class servo, shaft on the world pan axis. Body centre offset is accurate by intent.
    servo_body = rounded_box("MG996R body — 40.7 x 19.7", (40.7, 19.7, 37.0), (10.15, 0, 35.5), 2.2, black, col)
    flange = rounded_box("MG996R mounting flange", (54.5, 20.0, 3.0), (10.15, 0, 50.5), 1.5, black, col)
    gear = cylinder("MG996R gear cap", 7.0, 4.0, (0, 0, 55.0), mat=black, col=col, edge=0.8)
    spline = cylinder("MG996R spline", 3.0, 6.0, (0, 0, 59.5), vertices=48, mat=metal, col=col, edge=0.4)
    carrier = make_servo_carrier(col, accent)
    carrier.location.z += 50.0
    parts += [servo_body, flange, gear, spline, carrier]
    # 51100 thrust bearing and aluminium horn/hub keep phone load off the servo shaft.
    bearing_outer = cylinder("51100 thrust bearing 10 x 24 x 9", 12.0, 9.0, (0, 0, 66.0), mat=metal, col=col, edge=0.5)
    boolean(bearing_outer, cylinder("bearing bore", 5.0, 12.0, (0, 0, 66.0), col=col))
    hub = cylinder("servo-to-rotor hub", 5.0, 10.5, (0, 0, 65.0), vertices=48, mat=accent, col=col, edge=0.6)
    parts += [bearing_outer, hub]
    return parts


def add_text(name, body, loc, size, col, mat, extrude=0.22):
    curve = bpy.data.curves.new(name, "FONT")
    curve.body = body
    curve.align_x = "CENTER"
    curve.size = size
    curve.extrude = extrude
    obj = bpy.data.objects.new(name, curve)
    col.objects.link(obj)
    obj.location = loc
    assign(obj, mat)
    return obj


def aim_at(obj, point):
    direction = Vector(point) - obj.location
    obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def camera(name, loc, target, lens=58, ortho=None):
    data = bpy.data.cameras.new(name)
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = loc
    aim_at(obj, target)
    data.lens = lens
    if ortho:
        data.type = "ORTHO"
        data.ortho_scale = ortho
    return obj


def area_light(name, loc, energy, size, color=(1, 1, 1)):
    data = bpy.data.lights.new(name, "AREA")
    data.energy = energy
    data.shape = "DISK"
    data.size = size
    data.color = color
    obj = bpy.data.objects.new(name, data)
    bpy.context.scene.collection.objects.link(obj)
    obj.location = loc
    aim_at(obj, (0, 0, 110))
    return obj


def set_visible(objects, visible):
    for obj in objects:
        obj.hide_render = not visible


def render(path, camera_obj, resolution=(900, 900)):
    scene = bpy.context.scene
    scene.camera = camera_obj
    scene.render.resolution_x, scene.render.resolution_y = resolution
    scene.render.filepath = str(path)
    bpy.ops.render.render(write_still=True)


def prepare_export_copy(obj, name, export_col, rotate_x=0.0):
    dup = duplicate(obj, f"PRINT_{name}", export_col)
    dup.hide_render = True
    if rotate_x:
        dup.rotation_euler.x += math.radians(rotate_x)
    bpy.context.view_layer.update()
    corners = [dup.matrix_world @ Vector(corner) for corner in dup.bound_box]
    min_z = min(v.z for v in corners)
    dup.location.z -= min_z
    bpy.context.view_layer.update()
    return dup


def union_export_sources(name, sources, export_col, rotate_x=0.0):
    """Duplicate and boolean-union several assembly objects into one print mesh."""
    result = duplicate(sources[0], f"PRINT_{name}", export_col)
    result.hide_render = False
    for i, source in enumerate(sources[1:], 1):
        addition = duplicate(source, f"{name} union source {i}", export_col)
        addition.hide_render = False
        boolean(result, addition, "UNION")
    if rotate_x:
        result.rotation_euler.x += math.radians(rotate_x)
    bpy.context.view_layer.update()
    result.location.z -= min((result.matrix_world @ Vector(c)).z for c in result.bound_box)
    return result


def export_stl(obj, filename):
    bpy.ops.object.select_all(action="DESELECT")
    obj.hide_viewport = False
    obj.hide_set(False)
    obj.hide_render = False
    obj.select_set(True)
    bpy.context.view_layer.objects.active = obj
    path = str(EXPORT / filename)
    if hasattr(bpy.ops.wm, "stl_export"):
        bpy.ops.wm.stl_export(filepath=path, export_selected_objects=True)
    else:
        bpy.ops.export_mesh.stl(filepath=path, use_selection=True)
    obj.hide_viewport = True
    obj.select_set(False)


def main():
    clean_scene()
    scene = bpy.context.scene
    scene.unit_settings.system = "METRIC"
    scene.unit_settings.length_unit = "MILLIMETERS"
    scene.unit_settings.scale_length = 0.001
    # Blender 4.x used EEVEE_NEXT while the 5.x LTS package exposes EEVEE again.
    try:
        scene.render.engine = "BLENDER_EEVEE_NEXT"
    except TypeError:
        scene.render.engine = "BLENDER_EEVEE"
    scene.render.image_settings.file_format = "PNG"
    scene.render.film_transparent = False
    scene.render.resolution_percentage = 100
    scene.render.image_settings.color_mode = "RGBA"
    scene.world.use_nodes = True
    scene.world.node_tree.nodes["Background"].inputs["Color"].default_value = (0.055, 0.07, 0.095, 1)
    scene.world.node_tree.nodes["Background"].inputs["Strength"].default_value = 0.32

    # Pivo-inspired visual language, without copying its enclosure geometry.
    graphite = material("Graphite polymer", (0.075, 0.095, 0.13, 1), metallic=0.08, roughness=0.26)
    graphite_2 = material("Soft graphite", (0.12, 0.15, 0.20, 1), metallic=0.02, roughness=0.34)
    cradle_mat = material("Cradle slate", (0.19, 0.23, 0.29, 1), metallic=0.02, roughness=0.36)
    coral = material("FollowCam coral", (0.96, 0.19, 0.11, 1), metallic=0.02, roughness=0.28)
    silver = material("Machined aluminium", (0.36, 0.41, 0.47, 1), metallic=0.82, roughness=0.23)
    pcb = material("Arduino blue", (0.02, 0.20, 0.38, 1), metallic=0.08, roughness=0.33)
    glass = material("Phone glass", (0.004, 0.012, 0.022, 1), metallic=0.16, roughness=0.08)
    phone_metal = material("Phone edge", (0.14, 0.16, 0.19, 1), metallic=0.7, roughness=0.21)
    white = material("Label white", (0.9, 0.93, 0.97, 1), metallic=0.0, roughness=0.4)

    shell_col = collection("A — Enclosure")
    internals_col = collection("B — Electronics + drive")
    portrait_col = collection("C — Portrait full-wrap cradle")
    landscape_col = collection("D — Landscape full-wrap cradle")
    export_col = collection("Z — Printable exports (hidden)")

    lower = make_lower_shell(shell_col, graphite)
    upper = make_upper_shell(shell_col, graphite_2)
    rotor, receiver = make_rotor(shell_col, graphite, coral)
    internals = make_electronics(internals_col, pcb, silver, graphite, coral)

    portrait = list(make_phone_case("05_PORTRAIT_FULL_WRAP_CASE", PHONE_W, PHONE_H, portrait_col, cradle_mat, coral))
    portrait_flat = [portrait[0], *portrait[1], portrait[2]]
    portrait_phone = make_phone_dummy("Portrait phone placeholder", PHONE_W, PHONE_H, 93.5, portrait_col, glass, phone_metal)
    portrait_all = portrait_flat + portrait_phone

    # Landscape gets the same phone envelope rotated as a separate, print-ready attachment.
    landscape_bottom = 91.0
    landscape = list(make_phone_case("06_LANDSCAPE_FULL_WRAP_CASE", PHONE_H, PHONE_W, landscape_col, cradle_mat, coral, landscape_bottom))
    landscape_flat = [landscape[0], *landscape[1], landscape[2]]
    landscape_phone = make_phone_dummy("Landscape phone placeholder", PHONE_H, PHONE_W, 93.5, landscape_col, glass, phone_metal)
    landscape_all = landscape_flat + landscape_phone
    set_visible(landscape_all, False)

    # Branding remains an editable text object in the .blend file.
    brand = add_text("FollowCam mark", "FOLLOWCAM", (-0.0, -56.4, 37.0), 7.0, shell_col, white, 0.18)
    brand.rotation_euler = (math.radians(90), 0, 0)

    # Studio floor and lighting.
    floor_mat = material("Studio floor", (0.065, 0.08, 0.105, 1), metallic=0.0, roughness=0.52)
    floor = cylinder("studio plinth", 155, 4, (0, 0, -3), vertices=128, mat=floor_mat, col=collection("Studio"), edge=2)
    hero_cam = camera("Hero camera", (300, -390, 245), (0, 0, 132), lens=62)
    land_cam = camera("Landscape camera", (315, -420, 205), (0, 0, 112), lens=62)
    exploded_cam = camera("Exploded camera", (380, -500, 310), (0, 0, 155), lens=66)
    area_light("Key", (-210, -250, 390), 2200, 220, (1.0, 0.82, 0.72))
    area_light("Fill", (250, -110, 250), 1750, 180, (0.62, 0.78, 1.0))
    area_light("Rim", (0, 250, 330), 2400, 160, (1.0, 0.23, 0.12))

    # Renders: both orientation choices and an engineering exploded view.
    set_visible(internals, False)
    render(RENDER / "followcam_pod_v3_portrait.png", hero_cam)
    set_visible(portrait_all, False)
    set_visible(landscape_all, True)
    render(RENDER / "followcam_pod_v3_landscape.png", land_cam, (1100, 760))

    # Explode the portrait version to expose the actual Uno, servo carrier, bearing and rotor.
    set_visible(landscape_all, False)
    set_visible(portrait_all, True)
    set_visible(internals, True)
    moved = {obj: obj.location.copy() for obj in [upper, rotor, receiver, *portrait_all]}
    upper.location.x += 105
    upper.location.z += 18
    rotor.location.z += 58
    receiver.location.z += 58
    for obj in portrait_all:
        obj.location.z += 100
    render(RENDER / "followcam_pod_v3_exploded.png", exploded_cam, (1100, 900))
    for obj, loc in moved.items():
        obj.location = loc
    set_visible(internals, False)

    # Print exports. Shells are separate serviceable pieces; each phone attachment is one STL.
    exp_lower = prepare_export_copy(lower, "base_lower", export_col)
    exp_upper = prepare_export_copy(upper, "base_upper", export_col)
    exp_rotor = union_export_sources("rotating_platter", [rotor, receiver], export_col)
    carrier_source = next(obj for obj in internals if obj.name.startswith("03_SERVO_CARRIER"))
    carrier_copy = prepare_export_copy(carrier_source, "servo_carrier", export_col)

    portrait_part = union_export_sources("phone_case_portrait", portrait_flat, export_col, rotate_x=90)
    landscape_part = union_export_sources("phone_case_landscape", landscape_flat, export_col, rotate_x=90)

    for obj, filename in (
        (exp_lower, "followcam_v3_base_lower.stl"),
        (exp_upper, "followcam_v3_base_upper.stl"),
        (exp_rotor, "followcam_v3_rotating_platter.stl"),
        (carrier_copy, "followcam_v3_servo_carrier.stl"),
        (portrait_part, "followcam_v3_phone_case_portrait.stl"),
        (landscape_part, "followcam_v3_phone_case_landscape.stl"),
    ):
        export_stl(obj, filename)

    # Keep the useful portrait assembly visible when the file opens.
    set_visible(portrait_all, True)
    set_visible(landscape_all, False)
    set_visible(internals, False)
    scene.camera = hero_cam
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / "followcam_pod_v3.blend"))
    print(f"FollowCam Pod v3 generated at {OUT}")


if __name__ == "__main__":
    main()
