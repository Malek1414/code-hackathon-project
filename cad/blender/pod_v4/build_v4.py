"""FollowCam Pod v4 - design visualisation pass.

Builds the agreed v4 architecture and renders a .mov walkthrough.
NO STL EXPORT in this pass - geometry is for review only.

Run: blender --background --python cad/blender/pod_v4/build_v4.py
Construction units are mm; the scene is scaled to metres before rendering.
"""
import bpy, bmesh, math, sys, os, struct
from pathlib import Path
from mathutils import Vector, Matrix

OUT = Path(__file__).resolve().parent
IMAGES = OUT / 'renders'
IMAGES.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- parameters
PHONE_W, PHONE_H, PHONE_T = 76.5, 155.5, 13.0   # iPhone 16 Pro (149.6x71.5x8.25) + normal case
FIT      = 1.5        # phone pocket clearance PER SIDE  (v4: scaled up)
POD_D    = 112.0      # stator shell outer
ROTOR_D  = 110.0      # rotor outer (1 mm shadow gap to shell)
JOUR_D   = 88.0       # journal outer  - the printed slew ring
JOUR_CLR = 0.5        # journal clearance PER SIDE (v4: scaled up)
BORE_D   = JOUR_D + 2*JOUR_CLR          # 89.0
JOUR_ID  = 76.0       # journal inner bore (clears servo body 61.4)
PLATE_D  = 96.0       # carrier plate = doubles as rotor capture ring
TOWER_ID = 97.0       # rotor bore above the shoulder (clears the plate)

# MG996R
SV_L, SV_W, SV_H = 40.7, 20.0, 47.6   # 47.6 = TowerPro drawing overall (spec bullet says 42.9)
SV_FLANGE_W      = 53.6
SV_HOLE_X, SV_HOLE_Y, SV_HOLE_D = 49.5, 10.0, 5.0   # v4: holes opened to 5.0
SV_OFF           = 9.85      # spline offset from body centre. MEASURED off TowerPro's
                             # own top view: the spline sits 24.1% of the body length
                             # off centre. No datasheet dimensions it directly.
SV_FLANGE_Z      = 26.6      # body bottom -> flange bottom (drawing)
SV_FLANGE_T      = 2.5
SV_POCKET        = 0.8       # v4: +0.8 all round

# z stack
Z_SHELL_TOP = 15.0
Z_THRUST    = 16.0
Z_JOUR_TOP  = 35.0
Z_SHOULDER  = 34.0         # rotor bore step, 1 mm below the carrier
Z_PLATE0, Z_PLATE1 = 35.0, 40.0
Z_BODY0     = Z_PLATE1 - SV_FLANGE_Z      # 13.4
Z_BODY1     = Z_BODY0 + SV_H          # 59.4
Z_DECK0, Z_DECK1 = 63.0, 71.0
SOCK_L, SOCK_W, SOCK_CLR = 30.0, 12.0, 0.6
BLADE_R     = 18.0
WIRE_AZ     = 210.0        # umbilical exits toward the Uno leg case
NUT_AF      = 11.51        # 7/16" AF 1/4-20 nut + 0.4 fit
NUT_T       = 5.56         # nut thickness

PARTS = []
EXPORT = OUT / 'STL'

ORIENT = {                      # how each part lands on the bed
    '01_stator_shell':      ('as modelled', Matrix.Identity(4)),
    '02_nut_retainer':      ('as modelled', Matrix.Identity(4)),
    '03_servo_carrier_MG996R': ('as modelled', Matrix.Identity(4)),
    '04_rotor':             ('INVERTED - deck on the bed, bore opens up',
                             Matrix.Rotation(math.pi, 4, 'X')),
    '05_drive_dog_42.9-47.6': ('as modelled', Matrix.Identity(4)),
    '05b_drive_dog_SHORT_42.9': ('as modelled', Matrix.Identity(4)),
    '06_cradle_portrait':   ('BACK DOWN on the bed', Matrix.Rotation(-math.pi/2, 4, 'Y')),
    '07_cradle_landscape':  ('BACK DOWN on the bed', Matrix.Rotation(-math.pi/2, 4, 'Y')),
    '08_uno_tray':          ('as modelled - cavity opens up', Matrix.Identity(4)),
    '09_uno_lid':           ('as modelled', Matrix.Identity(4)),
    '10_leg_saddle':        ('INVERTED - flat face down, V opens up', Matrix.Rotation(math.pi, 4, 'X')),
    '11_aa_bay':            ('as modelled - cavity opens up', Matrix.Identity(4)),
}

# ---------------------------------------------------------------- helpers
def col(name):
    c = bpy.data.collections.new(name); bpy.context.scene.collection.children.link(c); return c

def move(o, c):
    for old in list(o.users_collection): old.objects.unlink(o)
    c.objects.link(o); return o

def mat(name, rgb, metal=0.0, rough=0.35):
    m = bpy.data.materials.new(name); m.use_nodes = True
    p = m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value = (*rgb, 1)
    p.inputs['Metallic'].default_value = metal
    p.inputs['Roughness'].default_value = rough
    return m

def material(o, m):
    o.data.materials.clear(); o.data.materials.append(m); return o

def bake(o):
    bpy.context.view_layer.objects.active = o
    bpy.ops.object.select_all(action='DESELECT'); o.select_set(True)
    bpy.ops.object.transform_apply(location=False, rotation=True, scale=True); return o

def boolean(a, b, operation='DIFFERENCE'):
    bpy.context.view_layer.objects.active = a
    m = a.modifiers.new(operation, 'BOOLEAN')
    m.operation = operation; m.solver = 'EXACT'; m.object = b
    bpy.ops.object.modifier_apply(modifier=m.name)
    bpy.data.objects.remove(b, do_unlink=True); return a

def cyl(name, d, z0, z1, xy=(0, 0), n=96):
    bpy.ops.mesh.primitive_cylinder_add(vertices=n, radius=d/2, depth=z1-z0, location=(*xy, (z0+z1)/2))
    o = bpy.context.object; o.name = name; return o

def box(name, size, xyz):
    bpy.ops.mesh.primitive_cube_add(size=1, location=xyz)
    o = bpy.context.object; o.name = name; o.dimensions = size; return bake(o)

def rr(name, w, h, r, z0, z1, xy=(0, 0), steps=12):
    r = max(0.01, min(r, w/2-.01, h/2-.01)); pts = []
    for cx, cy, a0 in ((w/2-r, h/2-r, 0), (-w/2+r, h/2-r, 90), (-w/2+r, -h/2+r, 180), (w/2-r, -h/2+r, 270)):
        for i in range(steps+1):
            a = math.radians(a0 + i*90/steps)
            pts.append((cx + r*math.cos(a) + xy[0], cy + r*math.sin(a) + xy[1]))
    n = len(pts); verts = [(x, y, z) for z in (z0, z1) for x, y in pts]
    faces = [tuple(reversed(range(n))), tuple(range(n, 2*n))]
    faces.extend((i, (i+1) % n, (i+1) % n + n, i+n) for i in range(n))
    me = bpy.data.meshes.new(name); me.from_pydata(verts, [], faces); me.update()
    o = bpy.data.objects.new(name, me); bpy.context.scene.collection.objects.link(o); return o

def bore(o, d, z0, z1, xy=(0, 0), n=48): return boolean(o, cyl('drill', d, z0, z1, xy, n))
def ring(name, od, id_, z0, z1): return bore(cyl(name, od, z0, z1), id_, z0-1, z1+1)
def slot(o, w, h, z0, z1, xy=(0, 0), r=1.5): return boolean(o, rr('cut', w, h, r, z0, z1, xy))

def part(o, name, c, m, explode=(0, 0, 0)):
    o.name = name; move(o, c); material(o, m); PARTS.append((o, Vector(explode))); return o

# ---------------------------------------------------------------- stator
def make_shell(c, m):
    """Floor + lower shell + thrust land + journal, printed as one piece."""
    o = cyl('stator shell', POD_D, 0, Z_SHELL_TOP)
    # Land is Ø100 and solid out to the journal; the Ø76 hollow below turns it into
    # an annulus. Do NOT relieve it with a cylinder that dips under Z_SHELL_TOP -
    # that leaves a hairline gap and severs the journal from the shell.
    boolean(o, cyl('thrust land', 100, Z_SHELL_TOP, Z_THRUST), 'UNION')
    boolean(o, cyl('journal', JOUR_D, Z_SHELL_TOP, Z_JOUR_TOP), 'UNION')
    bore(o, JOUR_ID, 12.0, Z_JOUR_TOP+1)               # hollow: servo lives inside
    # 1/4-20 interface. The nut pocket opens DOWNWARD and sits as low as possible:
    # a tripod screw only protrudes ~6.5 mm, so a nut any higher never engages.
    # The nut bears UP against the solid ceiling at z=7.3 - that is the load path.
    bore(o, 7.5, -1, 13)
    bore(o, NUT_AF/math.cos(math.pi/6), -1, 7.3, n=6)
    # wire exit + strain relief, aimed at the leg the Uno case clips to (az 210).
    # Sized for a 3-pin servo lead: ~7.5 x 2.5 mm, with room for the connector boot.
    we = rr('wire exit', 17, 9, 2.5, 7, 13, (POD_D/2 - 3, 0))
    we.rotation_euler = (0, 0, math.radians(WIRE_AZ)); bake(we); boolean(o, we)
    # Zip-tie strain relief: two radial slots straight THROUGH the wall (r36->r60),
    # leaving a 6 mm bridge. A blind pocket would seal an internal void and split
    # the mesh, and a tie needs somewhere to loop anyway.
    a = math.radians(WIRE_AZ)
    for sy in (-1, 1):
        px = 48*math.cos(a) - sy*11*math.sin(a)
        py = 48*math.sin(a) + sy*11*math.cos(a)
        t = box('tie slot', (26, 3.2, 4.0), (px, py, 10))
        t.rotation_euler = (0, 0, a); bake(t); boolean(o, t)
    # carrier-plate screw bosses on the journal top face
    for a in (45, 135, 225, 315):
        bore(o, 2.6, Z_JOUR_TOP-9, Z_JOUR_TOP+1, (41*math.cos(math.radians(a)), 41*math.sin(math.radians(a))))
    # weight-saving scallops in the skirt
    for a in range(0, 360, 30):
        bore(o, 9, -1.0, 5.0, (48*math.cos(math.radians(a)), 48*math.sin(math.radians(a))))
    return part(o, '01_stator_shell', c, m, (0, 0, 0))

def make_keeper(c, m):
    """Thin hex plug that press-fits BELOW the nut and stops it dropping out when the
    pod is off the tripod. 1.5 mm thick, so the screw still gets ~5 mm of thread."""
    o = cyl('nut retainer', (NUT_AF - 0.25)/math.cos(math.pi/6), 0.0, 1.5, n=6)
    bore(o, 7.6, -1, 2.5)
    return part(o, '02_nut_retainer', c, m, (0, 0, -30))

def make_carrier(c, m):
    """Swappable MG996R plate. Also the rotor capture ring - Ø96 over a Ø89 shoulder."""
    o = cyl('servo carrier', PLATE_D, Z_PLATE0, Z_PLATE1)
    # servo body pass-through, +0.8 all round, positioned so the SPLINE is on axis
    slot(o, SV_L + 2*SV_POCKET, SV_W + 2*SV_POCKET, Z_PLATE0-1, Z_PLATE1+1, (SV_OFF, 0), 1.5)
    # MG996R flange holes, opened to 5.0 for clone drift
    for sx in (-1, 1):
        for sy in (-1, 1):
            bore(o, SV_HOLE_D, Z_PLATE0-1, Z_PLATE1+1, (SV_OFF + sx*SV_HOLE_X/2, sy*SV_HOLE_Y/2))
    # fastening down to the journal top - well inside the Ø96 rim
    for a in (45, 135, 225, 315):
        bore(o, 3.6, Z_PLATE0-1, Z_PLATE1+1, (41*math.cos(math.radians(a)), 41*math.sin(math.radians(a))))
    # cable lanes
    for sy in (-1, 1):
        bore(o, 11, Z_PLATE0-1, Z_PLATE1+1, (0, sy*36))
    return part(o, '03_servo_carrier_MG996R', c, m, (0, 0, 34))

# ---------------------------------------------------------------- rotor
def make_rotor(c, m):
    o = cyl('rotor', ROTOR_D, Z_THRUST, Z_DECK1)
    bore(o, BORE_D, Z_THRUST-1, Z_SHOULDER)            # rides the journal
    bore(o, TOWER_ID, Z_SHOULDER, Z_DECK0)             # clears the capture plate
    # cradle socket - centred on the pan axis so the phone CoM lands on axis
    slot(o, SOCK_W + SOCK_CLR, SOCK_L + SOCK_CLR, Z_DECK0, Z_DECK1 + 1, (0, 0), 1.5)
    # drive-dog blade slot, offset to r=18 and perpendicular to the socket
    slot(o, 5.0, 15.0, Z_DECK0 - 1, Z_DECK1 - 1, (BLADE_R, 0), 1.0)
    # M4 thumbscrew that locks the cradle tongue
    b = cyl('thumbscrew', 4.6, 0, 62); b.rotation_euler = (0, math.radians(90), 0)
    bake(b); b.location = (31, 0, Z_DECK0 + 4); boolean(o, b)
    # grip flutes so it reads as a product, not a blank tube
    for a in range(0, 360, 15):
        bore(o, 3.2, Z_THRUST + 3, Z_THRUST + 26,
             ((ROTOR_D/2 + 0.4)*math.cos(math.radians(a)), (ROTOR_D/2 + 0.4)*math.sin(math.radians(a))), 16)
    return part(o, '04_rotor', c, m, (0, 0, 70))

def make_dog(c, m, short=False):
    """Sliding drive dog: torque only, free to float axially so the bearing takes load.
    TowerPro's drawing says the MG996R is 47.6 mm overall; the spec bullet on the same
    datasheet says 42.9. Both variants are exported - measure yours and print one."""
    top = Z_BODY1 - (47.6 - 42.9) if short else Z_BODY1
    o = rr('drive dog', 34, 13, 4, top, top + 2.0)
    bore(o, 6.4, top - 1, top + 3)                     # over the spline
    for sx in (-1, 1): bore(o, 2.6, top - 1, top + 3, (sx*13, 0))
    boolean(o, rr('blade', 4.0, 13.5, 0.4, top + 2.0, Z_DECK1 - 2.0, (BLADE_R, 0)), 'UNION')
    name = '05b_drive_dog_SHORT_42.9' if short else '05_drive_dog_42.9-47.6'
    return part(o, name, c, m, (0, 0, 52))

# ---------------------------------------------------------------- cradles
def make_cradle(c, m, landscape=False):
    """Back shell + tongue. Tongue sits at the phone's THICKNESS MID-PLANE so the
    phone's centre of mass lands on the pan axis - no static yaw imbalance."""
    w = (PHONE_H if landscape else PHONE_W) + 2*FIT
    h = (PHONE_W if landscape else PHONE_H) + 2*FIT
    t = PHONE_T + 2*FIT
    back = 3.0
    x_back = -(PHONE_T/2 + back)        # back face, axis bisects the phone
    o = box('cradle back', (back, w + 2*4, h), (x_back + back/2, 0, Z_DECK1 + 6 + h/2))
    # perimeter rails
    for sy in (-1, 1):
        boolean(o, box('rail', (t + back, 4.0, h), (x_back + (t + back)/2, sy*(w/2 + 2), Z_DECK1 + 6 + h/2)), 'UNION')
    boolean(o, box('chin', (t + back, w + 8, 9.0), (x_back + (t + back)/2, 0, Z_DECK1 + 6 + 4.5)), 'UNION')
    # four front retention lips
    for sx in (-1, 1):
        for sy in (-1, 1):
            boolean(o, box('lip', (3.0, 16, 10),
                    (PHONE_T/2 + 1.5, sy*(w/2 - 8), Z_DECK1 + 6 + h/2 + sx*(h/2 - 7))), 'UNION')
    # phone pocket
    boolean(o, box('phone pocket', (t, w, h + 20), (x_back + back + t/2, 0, Z_DECK1 + 6 + h/2 + 10)))
    # camera window / charge port
    boolean(o, box('camwin', (back + 2, 46, 46), (x_back + back/2, 12, Z_DECK1 + 6 + h - 34)))
    boolean(o, box('charge', (t + back + 2, 26, 10), (x_back + (t + back)/2, 0, Z_DECK1 + 6 + 3)))
    # footplate + tongue, centred on the pan axis
    boolean(o, box('footplate', (22, 62, 7.0), (-1.5, 0, Z_DECK1 + 3.5)), 'UNION')
    boolean(o, box('tongue', (SOCK_W, SOCK_L, 8.6), (0, 0, Z_DECK1 - 4.0)), 'UNION')
    b = cyl('thumbscrew', 4.6, -30, 30); b.rotation_euler = (0, math.radians(90), 0)
    b.location = (0, 0, Z_DECK0 + 4); bake(b); boolean(o, b)
    name = '07_cradle_landscape' if landscape else '06_cradle_portrait'
    return part(o, name, c, m, (0, 0, 110))

def dummy_phone(c, body, glass, landscape=False):
    w = PHONE_H if landscape else PHONE_W
    h = PHONE_W if landscape else PHONE_H
    z0 = Z_DECK1 + 6
    o = rr('phone', PHONE_T, w, 5, z0, z0 + h)
    material(o, body)
    g = box('screen', (0.6, w - 5, h - 6), (PHONE_T/2 + 0.2, 0, z0 + h/2)); material(g, glass)
    move(o, c); move(g, c)
    return o, g

# ---------------------------------------------------------------- purchased / context
def make_servo(c, black, metal):
    """MG996R visual. Spline on the pan axis, body offset SV_OFF - that offset is
    exactly why the pod cannot shrink below ~Ø100."""
    o = box('MG996R body', (SV_L, SV_W, SV_H), (SV_OFF, 0, Z_BODY0 + SV_H/2))
    boolean(o, box('flange', (SV_FLANGE_W, SV_W, SV_FLANGE_T),
                   (SV_OFF, 0, Z_BODY0 + SV_FLANGE_Z + SV_FLANGE_T/2)), 'UNION')
    boolean(o, cyl('boss', 13, Z_BODY1 - 5, Z_BODY1), 'UNION')
    material(o, black); move(o, c)
    sp = cyl('spline', 5.8, Z_BODY1 - 1, Z_BODY1 + 3.2); material(sp, metal); move(sp, c)
    return o, sp

def make_leg_case(c, shell, accent, blue, black):
    """Uno R3 case, v4 sizing: 82x66x32 interior for a 68.6x53.4 board.
    Deliberately roomy - the previous designs were all too tight to close."""
    OW, OD, OH = 96.0, 80.0, 30.0
    tray = rr('uno tray', OW, OD, 8, 0, OH)
    boolean(tray, rr('cavity', 82, 66, 6, 4, OH + 1))          # opens upward: no supports
    for x, y in ((-19.05, 24.13), (-20.32, -24.13), (31.75, 8.89), (31.75, -19.05)):
        boolean(tray, cyl('pedestal', 7.5, 4, 11, (x, y)), 'UNION')
        bore(tray, 2.7, 6, 13, (x, y))
    boolean(tray, box('usb-b',      (16, 20, 14), (-OW/2 + 1, 22, 13)))
    boolean(tray, box('barrel',     (16, 14, 13), (-OW/2 + 1, -6, 12)))
    boolean(tray, box('umbilical',  (16, 17, 12), ( OW/2 - 1, 0, 13)))
    for sx in (-1, 1): bore(tray, 3.6, -1, 5, (sx*32, 0))      # saddle bolts
    lid = rr('uno lid', OW, OD, 8, OH + 1, OH + 6)
    for a in range(-24, 25, 12):
        boolean(lid, box('vent', (3, 50, 9), (a, 0, OH + 3.5)))
    for sx in (-1, 1): bore(lid, 3.6, OH, OH + 7, (sx*42, 0))

    # Leg saddle. The V-groove cutter reaches the top face, which would leave TWO
    # loose rails - so a 6 mm web is unioned back on to rejoin them and give a flat
    # bolting face. Printed inverted: flat face on the bed, V opening upward.
    # 45 deg V, 36 mm across the opening, 18 mm deep, apex at z=-12 so everything
    # above it stays solid. A 52 mm cutter reached the top face and left two rails.
    # 90 deg V sized for the LARGEST plausible tripod leg: 50 mm across the opening,
    # 25 mm deep, apex at z=-11 so 11 mm stays solid above it. The MACTREM PT55's own
    # legs are only 20 mm tube, so this has a lot of headroom on purpose.
    SAD_T, SAD_W, VEE_D = 36.0, 66.0, 25.0
    cut = VEE_D / 0.70710678                      # square side whose half-diagonal = VEE_D
    sad = rr('leg saddle', OW, SAD_W, 8, -SAD_T, 0)
    v = box('vee', (OW + 4, cut, cut), (0, 0, -SAD_T)); v.rotation_euler = (math.radians(45), 0, 0)
    bake(v); boolean(sad, v)
    # strap slots go clean THROUGH the thickness (a blind pocket would seal a void)
    for sy in (-1, 1): boolean(sad, box('strap', (26, 6.0, SAD_T + 20), (0, sy*28, -10)))
    for sx in (-1, 1): bore(sad, 3.0, -9, 1, (sx*32, 0))

    # 4xAA bay - a real box with a cavity, not the solid placeholder it was
    aa = rr('4xAA bay', 68, 41, 5, 0, 21)
    boolean(aa, rr('aa cavity', 62, 35, 3, 3, 22))
    boolean(aa, box('aa lead', (12, 9, 9), (34, 0, 13)))
    for sx in (-1, 1): boolean(aa, box('aa strap', (5, 47, 7), (sx*26, 0, 11)))
    aa.location = (0, -62, 0)

    board = box('Uno R3', (68.6, 53.4, 1.6), (0, 0, 11.8)); material(board, blue)
    usb = box('usb', (12, 16, 11), (-31, 22, 18)); material(usb, black)
    part(tray, '08_uno_tray',   c, shell,  (0, 0, 0))
    part(lid,  '09_uno_lid',    c, accent, (0, 0, 0))
    part(sad,  '10_leg_saddle', c, shell,  (0, 0, 0))
    part(aa,   '11_aa_bay',     c, accent, (0, 0, 0))
    move(board, c); move(usb, c)
    return [tray, lid, sad, board, usb, aa]

def make_tripod(c, m, dark):
    plate = rr('MACTREM QR plate', 65, 45, 4, -9, -3); material(plate, m); move(plate, c)
    screw = cyl('1/4-20', 6.3, -4, 12); material(screw, m); move(screw, c)
    crown = cyl('crown', 46, -34, -9); material(crown, dark); move(crown, c)
    legs = []
    tilt = math.radians(24)
    for az in (90, 210, 330):
        a = math.radians(az)
        root = Vector((34*math.cos(a), 34*math.sin(a), -30.0))
        d = Vector((math.sin(tilt)*math.cos(a), math.sin(tilt)*math.sin(a), -math.cos(tilt)))
        leg = box('leg', (22, 13, 420), (0, 0, 0))
        leg.rotation_euler = d.to_track_quat('-Z', 'Y').to_euler()
        bake(leg)
        leg.location = root + d*210          # origin is the leg CENTRE -> push down half a length
        material(leg, dark); move(leg, c); legs.append(leg)
    return legs

metal_ref = [None]
LEG_CASE_AT = [Vector((0, 0, 0))]

# ---------------------------------------------------------------- STL export
def _tris(obj, mx):
    me = obj.data.copy()
    bm = bmesh.new(); bm.from_mesh(me)
    bmesh.ops.triangulate(bm, faces=bm.faces[:])
    bm.transform(mx)
    zmin = min(v.co.z for v in bm.verts)
    bm.transform(Matrix.Translation((0, 0, -zmin)))
    out = [tuple(tuple(v.co) for v in f.verts) for f in bm.faces]
    # integrity gate - this is what caught pod180's 3-piece carrier
    comps, seen = 0, set()
    for v in bm.verts:
        if v in seen: continue
        comps += 1; stack = [v]
        while stack:
            x = stack.pop()
            if x in seen: continue
            seen.add(x)
            for e in x.link_edges: stack.append(e.other_vert(x))
    bad = sum(1 for e in bm.edges if not e.is_manifold)
    vol = bm.calc_volume(signed=True)
    ext = [max(v.co[i] for v in bm.verts) - min(v.co[i] for v in bm.verts) for i in range(3)]
    bm.free(); bpy.data.meshes.remove(me)
    return out, comps, bad, vol, ext

def export_all():
    EXPORT.mkdir(parents=True, exist_ok=True)
    rows, ok = [], True
    for o, _ in PARTS:
        note, mx = ORIENT[o.name]
        tris, comps, bad, vol, ext = _tris(o, mx)
        good = comps == 1 and bad == 0 and vol > 0
        ok = ok and good
        if not good:
            print(f'  REFUSED {o.name}: components={comps} non-manifold={bad} vol={vol:.0f}')
            continue
        with (EXPORT / f'{o.name}.stl').open('wb') as f:
            f.write(b'FollowCam Pod v4'.ljust(80, b' '))
            f.write(struct.pack('<I', len(tris)))
            for t in tris:
                ux, uy, uz = (t[1][i] - t[0][i] for i in range(3))
                vx, vy, vz = (t[2][i] - t[0][i] for i in range(3))
                n = (uy*vz - uz*vy, uz*vx - ux*vz, ux*vy - uy*vx)
                L = math.sqrt(sum(c*c for c in n)) or 1.0
                f.write(struct.pack('<3f', *(c/L for c in n)))
                for vtx in t: f.write(struct.pack('<3f', *vtx))
                f.write(struct.pack('<H', 0))
        rows.append((o.name, len(tris), vol/1000, ext, note))
    print(f'\n{"part":32s} {"tris":>7s} {"cm3":>7s}  bed footprint (mm)      orientation')
    for n, t, v, e, note in rows:
        print(f'  {n:30s} {t:7d} {v:7.1f}  {e[0]:6.1f} x {e[1]:5.1f} x {e[2]:5.1f}   {note}')
    print(f'\n{len(rows)}/{len(PARTS)} parts exported to {EXPORT}')
    return ok

# ---------------------------------------------------------------- scene
def look_at(o, t):
    o.rotation_euler = (Vector(t) - o.location).to_track_quat('-Z', 'Y').to_euler()

def area(name, xyz, power, size, color):
    bpy.ops.object.light_add(type='AREA', location=xyz)
    L = bpy.context.object; L.name = name
    L.data.energy = power; L.data.size = size; L.data.color = color
    look_at(L, (0, 0, 0.09)); return L

def visible(objs, state):
    for o in objs:
        o.hide_render = not state; o.hide_viewport = not state

def build():
    bpy.ops.object.select_all(action='SELECT'); bpy.ops.object.delete(use_global=False)
    s = bpy.context.scene
    s.unit_settings.system = 'METRIC'; s.unit_settings.length_unit = 'MILLIMETERS'
    s.render.engine = 'BLENDER_EEVEE'
    try: s.eevee.taa_render_samples = 32
    except Exception: pass
    s.render.image_settings.file_format = 'PNG'
    s.world.use_nodes = True
    bg = s.world.node_tree.nodes['Background']
    bg.inputs[0].default_value = (.055, .065, .085, 1); bg.inputs[1].default_value = .25
    s.view_settings.view_transform = 'AgX'
    try: s.view_settings.look = 'AgX - Medium High Contrast'
    except Exception: pass
    s.view_settings.exposure = 0.6

    dark   = mat('PETG | midnight graphite', (.048, .058, .075), .10, .32)
    shellm = mat('PETG | slate',             (.115, .135, .165), .08, .34)
    accent = mat('PETG | vermilion',         (.88, .16, .08),    .05, .32)
    case_m = mat('PETG | porcelain',         (.62, .66, .68),    .03, .33)
    metal  = mat('Steel / aluminium',        (.52, .57, .62),    .85, .22)
    blue   = mat('Uno R3 PCB',               (.02, .24, .32),    .15, .45)
    black  = mat('Electronics polymer',      (.016, .020, .026), .00, .42)
    glass  = mat('Optical glass',            (.008, .018, .030), .30, .08)
    metal_ref[0] = metal

    c_stat = col('01 Stator (does not rotate)')
    c_rot  = col('02 Rotating assembly')
    c_port = col('03 Portrait attachment')
    c_land = col('04 Landscape attachment')
    c_hw   = col('05 Purchased hardware')
    c_ext  = col('06 Tripod + Uno leg case')

    make_shell(c_stat, shellm); make_keeper(c_stat, accent); make_carrier(c_stat, accent)
    rotor = make_rotor(c_rot, dark); dog = make_dog(c_rot, accent)
    dog_s = make_dog(c_rot, accent, short=True); visible([dog_s], False)
    make_cradle(c_port, case_m); make_cradle(c_land, case_m, True)
    pp, pg = dummy_phone(c_port, dark, glass)
    lp, lg = dummy_phone(c_land, dark, glass, True)
    body, spline = make_servo(c_hw, black, metal)
    legs = make_tripod(c_ext, metal, dark)
    leg_case = make_leg_case(c_ext, shellm, accent, blue, black)

    # ---- to metres FIRST: parenting after this point must not be re-scaled ----
    for o in list(s.objects):
        o.location *= .001; o.scale *= .001
    # matrix_world is cached: without this flush, the parenting below reads the
    # PRE-scale matrices and silently restores millimetre-sized objects.
    bpy.context.view_layer.update()

    # clip the Uno case onto the front-left leg (metres from here down)
    az, tilt, drop = math.radians(210), math.radians(24), .150
    d = Vector((math.sin(tilt)*math.cos(az), math.sin(tilt)*math.sin(az), -math.cos(tilt)))
    out = Vector((math.cos(az), math.sin(az), 0))
    holder = bpy.data.objects.new('leg case rig', None); s.collection.objects.link(holder)
    for o in leg_case:
        w = o.matrix_world.copy(); o.parent = holder; o.matrix_world = w
    xax = d.normalized()
    zax = (out - xax*out.dot(xax)).normalized()
    yax = zax.cross(xax)
    pos = Vector((.034*math.cos(az), .034*math.sin(az), -.030)) + d*drop + zax*.016
    holder.matrix_world = Matrix(((xax.x, yax.x, zax.x, pos.x),
                                  (xax.y, yax.y, zax.y, pos.y),
                                  (xax.z, yax.z, zax.z, pos.z),
                                  (0, 0, 0, 1)))
    LEG_CASE_AT[0] = pos.copy()

    ground = cyl('ground', 3000, -430, -428); ground.scale = (.001,)*3
    ground.location *= .001; material(ground, mat('Backdrop', (.035, .042, .055), .05, .55))

    area('key',  (-.34, -.40, .46), 14, .36, (1, .90, .82))
    area('fill', ( .42, -.24, .28),  7, .32, (.74, .86, 1))
    area('rim',  ( .06,  .40, .38), 11, .28, (1, .58, .36))

    # pan rig: everything that actually turns
    bpy.context.view_layer.update()
    pivot = bpy.data.objects.new('PAN', None); s.collection.objects.link(pivot)
    for o in [rotor, dog, *c_port.objects, *c_land.objects]:
        w = o.matrix_world.copy(); o.parent = pivot; o.matrix_world = w

    cam_d = bpy.data.cameras.new('cam'); cam = bpy.data.objects.new('cam', cam_d)
    s.collection.objects.link(cam); s.camera = cam; cam_d.lens = 62; cam_d.clip_end = 50
    return s, cam, pivot, rotor, c_port, c_land, c_stat, c_rot, c_hw, c_ext

# ---------------------------------------------------------------- shots
def render_shot(s, name, frames, res=(1280, 720)):
    d = IMAGES / name; d.mkdir(parents=True, exist_ok=True)
    s.render.resolution_x, s.render.resolution_y = res
    s.render.filepath = str(d) + '/f'
    s.frame_start, s.frame_end = 1, frames
    bpy.ops.render.render(animation=True)
    print(f'[shot] {name}: {frames} frames -> {d}')

def key_cam(cam, f, loc, target, lens=None):
    cam.location = Vector(loc); look_at(cam, target)
    cam.keyframe_insert('location', frame=f); cam.keyframe_insert('rotation_euler', frame=f)
    if lens is not None:
        cam.data.lens = lens; cam.data.keyframe_insert('lens', frame=f)

def orbit(cam, f0, f1, r, z, target, a0, a1, lens=62, steps=8):
    # Two keyframes would make Blender lerp a CHORD between them, halving the
    # camera distance mid-shot. Sample the arc instead.
    for i in range(steps + 1):
        f = round(f0 + (f1 - f0)*i/steps); a = a0 + (a1 - a0)*i/steps
        key_cam(cam, f, (r*math.cos(math.radians(a)), r*math.sin(math.radians(a)), z), target, lens)

def clear_anim(objs):
    for o in objs:
        o.animation_data_clear()
        if getattr(o, 'data', None) and getattr(o.data, 'animation_data', None):
            o.data.animation_data_clear()

def main():
    s, cam, pivot, rotor, c_port, c_land, c_stat, c_rot, c_hw, c_ext = build()
    if not export_all():
        raise SystemExit('export gate failed - not all parts are single watertight solids')
    if os.environ.get('FOLLOWCAM_SKIP_RENDER'):
        bpy.ops.wm.save_as_mainfile(filepath=str(OUT / 'followcam_pod_v4.blend'))
        print('[done] STLs only (render skipped)'); return
    s.render.fps = 30
    land = list(c_land.objects); port = list(c_port.objects); hw = list(c_hw.objects)
    visible(land, False)
    pivot.rotation_euler = (0, 0, 0)
    base = {o: o.location.copy() for o, _ in PARTS}
    # vertical half-FOV at 16:9 is atan(10.125/f) - frame from that, not from guesswork
    tripod = list(c_ext.objects)

    # 1 - hero orbit: pod, phone and the top of the tripod
    clear_anim([cam])
    orbit(cam, 1, 150, 1.02, .34, (0, 0, .030), -75, 65, 50)
    render_shot(s, '01_hero', 150)

    # 2 - the pan itself, camera locked off
    clear_anim([cam, pivot])
    key_cam(cam, 1, (.30, -.88, .30), (0, 0, .10), 50)
    for f, a in ((1, 0), (45, -90), (120, 90), (150, 0)):
        pivot.rotation_euler.z = math.radians(a); pivot.keyframe_insert('rotation_euler', frame=f)
    render_shot(s, '02_pan', 150)

    # 3 - cutaway: rotor hidden, MG996R hanging through the carrier inside the journal
    clear_anim([cam, pivot]); pivot.rotation_euler = (0, 0, 0)
    visible([rotor], False); visible(port, False); visible(tripod, False)
    orbit(cam, 1, 120, .33, .082, (0, 0, .036), -50, 45, 50)
    render_shot(s, '03_cutaway', 120)
    visible([rotor], True); visible(port, True)

    # 4 - exploded stack
    clear_anim([cam, pivot]); pivot.rotation_euler = (0, 0, 0)
    # the phone rides with its cradle; the servo rides with the carrier
    moving = list(PARTS)
    moving += [(o, Vector((0, 0, 110))) for o in port if not o.name.startswith('06_')]
    moving += [(o, Vector((0, 98, 18))) for o in hw]
    home = {o: o.location.copy() for o, _ in moving}
    for o, ex in moving:
        o.location = home[o]; o.keyframe_insert('location', frame=1)
        o.location = home[o] + Vector(ex)*.001; o.keyframe_insert('location', frame=95)
        o.keyframe_insert('location', frame=125)
    orbit(cam, 1, 125, .99, .44, (0, 0, .160), -38, 18, 50)
    render_shot(s, '04_exploded', 125)
    clear_anim([o for o, _ in moving])
    for o, _ in moving: o.location = home[o]
    visible(tripod, True)

    # 5 - landscape attachment
    clear_anim([cam, pivot]); pivot.rotation_euler = (0, 0, 0)
    visible(port, False); visible(land, True)
    orbit(cam, 1, 100, .54, .27, (0, 0, .080), -60, 15, 50)
    render_shot(s, '05_landscape', 100)
    visible(land, False); visible(port, True)

    # 6 - the Uno R3 case clipped to the tripod leg
    clear_anim([cam, pivot])
    tgt = LEG_CASE_AT[0] + Vector((0, 0, .012))
    for i in range(9):
        # 280-350 deg is roughly PERPENDICULAR to the case's outward face, so the
        # saddle straddling the leg reads instead of a flat-on view of the lid
        f = round(1 + 99*i/8); a = math.radians(278 + 74*i/8)
        cam.location = tgt + Vector((.38*math.cos(a), .38*math.sin(a), .085))
        look_at(cam, tgt); cam.data.lens = 50
        cam.keyframe_insert('location', frame=f); cam.keyframe_insert('rotation_euler', frame=f)
    render_shot(s, '06_leg_case', 100)

    bpy.ops.wm.save_as_mainfile(filepath=str(OUT / 'followcam_pod_v4.blend'))
    print('[done] all shots rendered')

if __name__ == '__main__':
    main()
