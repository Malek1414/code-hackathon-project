"""FollowCam 180: Blender-authored PETG prototype for a bare iPhone 16 Pro.
Run: blender --background --python cad/blender/pod180/build.py
Model construction/export coordinates are mm; the saved scene uses real metres.
"""
import bpy
import bmesh
import json
import math
import struct
from pathlib import Path
from mathutils import Vector, Matrix

OUT = Path(__file__).resolve().parent
PARTS = OUT / 'STL'
IMAGES = OUT / 'renders'
for directory in (PARTS, IMAGES):
    directory.mkdir(parents=True, exist_ok=True)

PHONE_W, PHONE_H, PHONE_T = 71.45, 149.61, 8.25
FIT = 0.45                         # clearance PER SIDE; thin foam shims take up play
R = 58.0
BOLT_XY = [(48*math.cos(math.radians(a)),48*math.sin(math.radians(a))) for a in (45,135,225,315)]
UNO_HOLES = [(-19.05,24.13),(-20.32,-24.13),(31.75,8.89),(31.75,-19.05)]
MOUNT_HOLES = [(x,y) for x in (-12,12) for y in (-7,23)]
PRINT = []
REPORT = []

def col(name):
    c=bpy.data.collections.new(name)
    bpy.context.scene.collection.children.link(c)
    return c

def move(o,c):
    for old in list(o.users_collection): old.objects.unlink(o)
    c.objects.link(o)
    return o

def mat(name,rgb,metal=0,rough=.35):
    m=bpy.data.materials.new(name); m.diffuse_color=(*rgb,1); m.use_nodes=True
    p=m.node_tree.nodes.get('Principled BSDF')
    p.inputs['Base Color'].default_value=(*rgb,1)
    p.inputs['Metallic'].default_value=metal; p.inputs['Roughness'].default_value=rough
    return m

def material(o,m):
    o.data.materials.clear(); o.data.materials.append(m); return o

def bake(o):
    bpy.context.view_layer.objects.active=o
    bpy.ops.object.select_all(action='DESELECT');o.select_set(True)
    bpy.ops.object.transform_apply(location=False,rotation=True,scale=True)
    return o

def boolean(a,b,operation='DIFFERENCE'):
    bpy.context.view_layer.objects.active=a
    m=a.modifiers.new(operation,'BOOLEAN');m.operation=operation;m.solver='EXACT';m.object=b
    bpy.ops.object.modifier_apply(modifier=m.name)
    bpy.data.objects.remove(b,do_unlink=True)
    return a

def cylinder(name,d,z0,z1,xy=(0,0),n=96):
    bpy.ops.mesh.primitive_cylinder_add(vertices=n,radius=d/2,depth=z1-z0,location=(*xy,(z0+z1)/2))
    o=bpy.context.object;o.name=name;return o

def box(name,size,xyz):
    bpy.ops.mesh.primitive_cube_add(size=1,location=xyz)
    o=bpy.context.object;o.name=name;o.dimensions=size;return bake(o)

def rr(name,w,h,r,z0,z1,xy=(0,0),steps=16):
    """Extruded round rectangle: planar faces, true XY corner radii, no tiny triangles."""
    points=[]
    for cx,cy,a0 in ((w/2-r,h/2-r,0),(-w/2+r,h/2-r,90),(-w/2+r,-h/2+r,180),(w/2-r,-h/2+r,270)):
        for i in range(steps+1):
            a=math.radians(a0+i*90/steps)
            points.append((cx+r*math.cos(a)+xy[0],cy+r*math.sin(a)+xy[1]))
    n=len(points);verts=[(x,y,z) for z in (z0,z1) for x,y in points]
    faces=[tuple(reversed(range(n))),tuple(range(n,2*n))]
    faces.extend((i,(i+1)%n,(i+1)%n+n,i+n) for i in range(n))
    mesh=bpy.data.meshes.new(name);mesh.from_pydata(verts,[],faces);mesh.update()
    o=bpy.data.objects.new(name,mesh);bpy.context.scene.collection.objects.link(o);return o

def bore(o,d,z0,z1,xy=(0,0),n=48):
    return boolean(o,cylinder('drill',d,z0,z1,xy,n))

def ring(name,od,id,z0,z1):
    return bore(cylinder(name,od,z0,z1),id,z0-1,z1+1)

def slot(o,w,h,z0,z1,xy=(0,0),r=1.5):
    return boolean(o,rr('opening',w,h,min(r,w/2-.01,h/2-.01),z0,z1,xy))

def part(o,name,c,m,orientation=None):
    o.name=name;move(o,c);material(o,m)
    PRINT.append((o,name,orientation or Matrix.Identity(4)))
    return o

def screw_points(o,points,d,z0,z1):
    for p in points:bore(o,d,z0,z1,p)
    return o

def make_floor(c,m):
    o=cylinder('floor',116,0,5)
    # A steel 1/4-20 nut is sandwiched against a real steel washer, not a printed thread.
    bore(o,6.8,-1,9)
    boolean(o,cylinder('tripod nut boss',24,4,9),'UNION')
    bore(o,6.8,-1,10)
    # Regular 7/16-inch AF nut: pocket circumdiameter = AF / cos(30).
    bore(o,11.45/math.cos(math.pi/6),4,9.5,n=6)
    bore(o,16.5,2.5,4.01)
    for x,y in UNO_HOLES:
        boolean(o,cylinder('board pedestal',6.4,4.5,14,(x,y)),'UNION')
        bore(o,2.5,7,15,(x,y))
    screw_points(o,BOLT_XY,3.4,-1,6)
    for p in BOLT_XY:bore(o,6.4,-.1,2.5,p)
    # Anti-rotation nut keeper screws and optional rubber foot recesses.
    screw_points(o,[(-9,0),(9,0)],2.5,4,10)
    return part(o,'01_tripod_floor',c,m)

def make_keeper(c,m):
    o=rr('nut lid',24,17,3,9,11)
    bore(o,6.8,8,12);screw_points(o,[(-9,0),(9,0)],3.4,8,12)
    return part(o,'02_tripod_nut_keeper',c,m)

def make_body(c,m):
    o=ring('body',116,109,5,82)
    # Four continuous bosses connect the floor, body and upper bearing deck.
    for x,y in BOLT_XY:
        boolean(o,cylinder('enclosure boss',16,5,82,(x,y)),'UNION')
        bore(o,4.2,4,12,(x,y));bore(o,4.2,75,83,(x,y))
    # Carrier shelves are at different azimuths from the enclosure screws.
    for xy in ((47,0),(-47,0),(0,47),(0,-47)):
        boolean(o,cylinder('servo shelf',18,49,56,xy),'UNION');bore(o,4.2,50,57,xy)
    # A wide USB opening aligns with the real Uno USB connector on its LEFT edge.
    boolean(o,box('USB-B access',(20,21,19),(-55,11,19)))
    boolean(o,box('DC lead and optional Uno jack access',(20,14,13),(-55,-20,18)))
    # Servo supply exits through an 8 mm strain-relieved cable opening at the rear.
    cable=cylinder('servo supply hole',8,-10,10);cable.rotation_euler.x=math.pi/2;cable.location=(18,55,20)
    bake(cable);boolean(o,cable)
    for x in (-28,-21,-14,0,7,14,28):
        boolean(o,box('vent',(3,20,3),(x,55,39)))
    # Internal tie points hold the stationary harness below the motor.
    for y in (-36,36):
        tab=rr('cable anchor',20,6,1.5,8,12,(0,y))
        # Connect to adjacent wall with a short rib.
        boolean(tab,box('anchor rib',(5,20,4),(0,y+math.copysign(9,y),10)),'UNION')
        slot(tab,12,2.6,7,13,(0,y),1)
        boolean(o,tab,'UNION')
    return part(o,'03_stationary_enclosure',c,m)

def make_carrier(c,m):
    o=cylinder('servo tray',107,56,60)
    screw_points(o,BOLT_XY,16.6,55,61)
    slot(o,41,21,55,61,(-10,0),1)
    screw_points(o,[(x,y) for x in (-34.75,14.75) for y in (-5,5)],3.4,55,61)
    for x,y in ((47,0),(-47,0),(0,47),(0,-47)):bore(o,3.4,55,61,(x,y))
    # Broad cable lanes and lightening pockets never overlap the flange or fasteners.
    for y in (-29,29):slot(o,52,18,55,61,(0,y),5)
    return part(o,'04_servo_carrier_DS3218',c,m)

def make_deck(c,m):
    o=cylinder('lid',116,82,88)
    boolean(o,cylinder('bearing cartridge',56,87,104),'UNION')
    bore(o,37,81,105)
    bore(o,42.30,84,105) # bottom shoulder is 2 mm high, outer race only
    screw_points(o,BOLT_XY,3.4,81,89)
    for p in BOLT_XY:bore(o,6.4,85,89,p)
    points=[(24*math.cos(math.radians(a)),24*math.sin(math.radians(a))) for a in (0,120,240)]
    screw_points(o,points,4.2,97,105)
    return part(o,'05_bearing_deck',c,m)

def make_bearing_parts(c,m):
    spacer=ring('outer race spacer',41.8,37,91,97)
    part(spacer,'06_outer_race_spacer',c,m)
    cap=ring('bearing lid',56,37,104,107)
    pts=[(24*math.cos(math.radians(a)),24*math.sin(math.radians(a))) for a in (0,120,240)]
    screw_points(cap,pts,3.4,103,108)
    part(cap,'07_bearing_retainer',c,m)
    return spacer,cap

def make_rotor(c,m):
    o=cylinder('rotor',90,110,116)
    boolean(o,cylinder('inner race shoulder',34,104,111),'UNION')
    boolean(o,cylinder('shaft',29.80,83.7,105),'UNION')
    # Sliding dog coupling gives axial float; the bearings carry the weight.
    slot(o,22.4,4.6,82,98,r=.3)
    bore(o,4.4,97,117)
    bolt_pts=[(12*math.cos(math.radians(a)),12*math.sin(math.radians(a))) for a in (90,210,330)]
    screw_points(o,bolt_pts,2.5,83,93)
    screw_points(o,MOUNT_HOLES,4.2,110,117)
    part(o,'08_rotor_and_spindle',c,m,Matrix.Rotation(math.pi,4,'X'))
    washer=ring('spindle retaining washer',36,4.4,80,83.7)
    slot(washer,22.4,4.6,79,85,r=.3)
    screw_points(washer,bolt_pts,3.4,79,85)
    part(washer,'09_spindle_retaining_washer',c,m)
    # DS3218 supplied single arm: use one outer bolt plus centre screw on metal horn.
    dog=rr('horn drive',30,10,4,76.5,79.5,(4,0))
    boolean(dog,rr('coupling blade',20,4.2,.3,79,94),'UNION')
    bore(dog,4.4,75,95)
    # Slot accepts measured outer horn holes from 12 to 16 mm from spline centre.
    slot(dog,7,3.4,75,81,(14,0),1.6)
    part(dog,'10_sliding_drive_dog',c,m)
    return o,washer,dog

def phone_transform(landscape=False):
    # Native phone: X width, Y height, Z back-to-screen. Screen faces -world Y.
    planar=Matrix.Rotation(math.pi/2 if landscape else 0,4,'Z')
    height=PHONE_W if landscape else PHONE_H
    return Matrix.Translation((0,7.425,134+height/2)) @ Matrix.Rotation(math.pi/2,4,'X') @ planar

def make_case(c,m,accent,landscape=False):
    w,h=PHONE_W,PHONE_H
    outerw,outerh=w+12,h+12
    o=rr('full perimeter case',outerw,outerh,12,0,12.5)
    slot(o,w+2*FIT,h+2*FIT,3,14,r=17.8)
    # Camera assembly and flash: >Apple plateau footprint, cut through back only.
    # Native coordinates viewed from FRONT, so camera appears at right.
    slot(o,46,49,-1,3.1,(w/2-23,h/2-24.5),10)
    # Buttons: continuous side openings preserve top and bottom rails.
    # Left action / volume centres 34.08,48.23,62.43 mm from top.
    boolean(o,box('action and volume',(14,43,7),(-w/2-2,h/2-48,7.8)))
    boolean(o,box('side button',(14,24,7),(w/2+2,h/2-55.33,7.8)))
    boolean(o,box('camera control',(14,25,7),(w/2+2,-h/2+30.5,7.8)))
    # A generous bottom opening clears speaker/mics and Apple's 14 mm connector keepout.
    boolean(o,box('USB-C and speaker opening',(55,16,8),(0,-h/2-2,7.8)))
    # Separate screw-on bezel captures edges; no PETG spring clips to fatigue.
    screws=[(sx*(w/2+1.5),sy*(h/2-5)) for sx in (-1,1) for sy in (-1,1)]
    for xy in screws:
        # Local post intersects the existing thick perimeter and stays outside phone cavity.
        bore(o,1.9,4,13,xy)
    bezel=rr('front retaining rim',outerw,outerh,12,12.5,14.9)
    slot(bezel,w-1.8,h-1.8,12,16,r=18.4)
    # Open earpiece slot and thin foam on inner rim maintain glass clearance.
    slot(bezel,19,6,12,16,(0,h/2),1)
    screw_points(bezel,screws,2.4,12,16)
    transform=phone_transform(landscape)
    o.matrix_world=transform;bezel.matrix_world=transform
    # Rear spine + gusseted base. It does not cross the USB plug path at Y=0.
    stem=box('rear spine',(30,14,48),(0,13,140))
    boolean(o,stem,'UNION')
    foot=rr('four-bolt shoe',38,44,5,116,122,(0,8))
    boolean(o,foot,'UNION')
    screw_points(o,MOUNT_HOLES,3.4,115,123)
    label='landscape' if landscape else 'portrait'
    # Orient tray rear-down for continuous layer paths through the neck; supports under tray.
    orient=Matrix.Rotation(-math.pi/2,4,'X')
    part(o,f'11_case_{label}_iphone16pro',c,m,orient)
    part(bezel,f'12_bezel_{label}_iphone16pro',c,accent,orient)
    return o,bezel

def visual_hardware(c,metal,blue,black,orange):
    items=[]
    def add(o,m):move(o,c);material(o,m);items.append(o);return o
    b=rr('Uno R3 PCB — correct asymmetric hole pattern',68.58,53.34,1.5,14,15.6)
    screw_points(b,UNO_HOLES,3.2,13,17);add(b,blue)
    add(box('USB-B socket',(17,13,12),(-33,11,21.6)),metal)
    add(box('Uno barrel jack',(14,9,11),(-30,-20,21.1)),black)
    add(box('ATmega328P — the microcontroller',(34,8,4),(4,-3,17.6)),black)
    for y in (-23,23):add(box('Uno female headers',(48,4,9),(5,y,20)),black)
    add(box('DS3218 servo body',(40,20,40.4),(-10,0,52.2)),black)
    add(box('DS3218 flange 49.5 x 10 hole pattern',(54.5,20,3),(-10,0,61.5)),black)
    add(cylinder('servo spline',6,72.4,75),metal)
    add(rr('supplied 25T aluminium horn',32,8,4,74,76.5,(9,0)),metal)
    for z in (84,97):
        add(ring(f'6806 bearing outer race z{z}',42,37,z,z+7),metal)
        add(ring(f'6806 bearing inner race z{z}',34,30,z,z+7),metal)
        add(ring('bearing shield',37,34,z+.4,z+6.6),black)
    add(cylinder('steel 1/4-20 nut',11.11/math.cos(math.pi/6),4,8.5,n=6),metal)
    add(ring('steel load washer',16,6.6,2.5,4),metal)
    # Stationary supply junction, capacitor and wires are kept below the servo tray.
    add(box('6V servo power terminal',(25,12,12),(14,-36,18)),orange)
    add(cylinder('1000 uF bulk capacitor',10,12,28,(31,-31)),black)
    return items

def wire(name,points,m,c):
    curve=bpy.data.curves.new(name,'CURVE');curve.dimensions='3D';curve.bevel_depth=.65;curve.bevel_resolution=3
    s=curve.splines.new('BEZIER');s.bezier_points.add(len(points)-1)
    for p,co in zip(s.bezier_points,points):p.co=co;p.handle_left_type='AUTO';p.handle_right_type='AUTO'
    o=bpy.data.objects.new(name,curve);c.objects.link(o);material(o,m);return o

def dummy_phone(c,m,glass,land=False):
    t=phone_transform(land)
    body=rr('iPhone 16 Pro 199 g — reference only',PHONE_W,PHONE_H,18.8,3.3,11.55)
    body.matrix_world=t;move(body,c);material(body,m)
    screen=rr('display',PHONE_W-3,PHONE_H-3,18,11.56,11.65)
    screen.matrix_world=t;move(screen,c);material(screen,glass)
    allparts=[body,screen]
    for x,y in ((PHONE_W/2-14.17,PHONE_H/2-14.17),(PHONE_W/2-14.17,PHONE_H/2-33.41),(PHONE_W/2-32.16,PHONE_H/2-23.79)):
        lens=cylinder('camera lens reference',16.2,-.98,3.31,(x,y));lens.matrix_world=t @ lens.matrix_world
        move(lens,c);material(lens,glass);allparts.append(lens)
    return allparts

def export_mesh(obj,name,rotation):
    """Write binary STL directly from evaluated triangles, independent of UI visibility."""
    mesh=obj.data.copy();matrix=rotation @ obj.matrix_world
    mesh.transform(matrix)
    zmin=min(v.co.z for v in mesh.vertices)
    for v in mesh.vertices:v.co.z-=zmin
    bm=bmesh.new();bm.from_mesh(mesh);bmesh.ops.remove_doubles(bm,verts=list(bm.verts),dist=0.00001)
    bmesh.ops.recalc_face_normals(bm,faces=list(bm.faces))
    bad=sum(not e.is_manifold for e in bm.edges)
    volume=bm.calc_volume(signed=True)
    seen=set();components=0
    for v in bm.verts:
        if v in seen:continue
        components+=1;stack=[v];seen.add(v)
        while stack:
            vertex=stack.pop()
            for e in vertex.link_edges:
                other=e.other_vert(vertex)
                if other not in seen:seen.add(other);stack.append(other)
    bm.to_mesh(mesh);bm.free();mesh.calc_loop_triangles()
    assert bad==0 and volume>0 and components==1, (name,bad,volume,components)
    minimum=[min(v.co[i] for v in mesh.vertices) for i in range(3)]
    maximum=[max(v.co[i] for v in mesh.vertices) for i in range(3)]
    dims=[round(b-a,3) for a,b in zip(minimum,maximum)]
    assert dims[0]<250 and dims[1]<210 and dims[2]<220,(name,dims)
    with (PARTS/(name+'.stl')).open('wb') as f:
        f.write(b'FollowCam 180 | millimetres | PETG prototype'.ljust(80,b' '))
        f.write(struct.pack('<I',len(mesh.loop_triangles)))
        for tri in mesh.loop_triangles:
            a,b,c=[mesh.vertices[i].co for i in tri.vertices]
            normal=(b-a).cross(c-a).normalized()
            f.write(struct.pack('<12fH',*normal,*a,*b,*c,0))
    REPORT.append(dict(part=name,dimensions_mm=dims,triangles=len(mesh.loop_triangles),nonmanifold_edges=bad,connected_components=components,solid_volume_cm3=round(volume/1000,3)))
    bpy.data.meshes.remove(mesh)

def look_at(o,target):o.rotation_euler=(Vector(target)-o.location).to_track_quat('-Z','Y').to_euler()

def camera(loc,target,scale):
    data=bpy.data.cameras.new('Product camera');data.type='ORTHO';data.ortho_scale=scale
    o=bpy.data.objects.new('Product camera',data);bpy.context.scene.collection.objects.link(o)
    o.location=loc;look_at(o,target);bpy.context.scene.camera=o;return o

def area(name,xyz,power,size,color):
    d=bpy.data.lights.new(name,'AREA');d.energy=power;d.shape='DISK';d.size=size;d.color=color
    o=bpy.data.objects.new(name,d);bpy.context.scene.collection.objects.link(o);o.location=xyz;look_at(o,(0,0,.12))

def visible(objs,state):
    for o in objs:o.hide_render=not state;o.hide_set(not state)

def render(name,cam,res=(1200,1200)):
    s=bpy.context.scene;s.camera=cam;s.render.resolution_x,s.render.resolution_y=res
    s.render.filepath=str(IMAGES/(name+'.png'));bpy.ops.render.render(write_still=True)

def main():
    bpy.ops.object.select_all(action='SELECT');bpy.ops.object.delete(use_global=False)
    s=bpy.context.scene;s.unit_settings.system='METRIC';s.unit_settings.length_unit='MILLIMETERS'
    s.render.engine='CYCLES';s.cycles.samples=40;s.cycles.use_denoising=True
    s.render.image_settings.file_format='PNG';s.render.resolution_percentage=100
    s.render.film_transparent=False
    s.world.use_nodes=True;s.world.node_tree.nodes['Background'].inputs[0].default_value=(.18,.22,.29,1)
    s.world.node_tree.nodes['Background'].inputs[1].default_value=.35
    dark=mat('PETG | midnight graphite',(.055,.075,.105),.08,.3)
    case_mat=mat('PETG | warm porcelain',(.64,.69,.70),.03,.32)
    accent=mat('PETG | vermilion',(.95,.15,.075),.05,.3)
    metal=mat('Purchased steel and aluminium',(.4,.47,.53),.8,.26)
    blue=mat('Uno R3 PCB',(.01,.22,.33),.15,.45)
    black=mat('Electronics polymer',(.018,.024,.031),0,.42)
    glass=mat('Optical glass',(.009,.021,.033),.3,.12)
    c=col('01 Stationary PETG housing');motion=col('02 Rotating drive');port=col('03 Portrait attachment');land=col('04 Landscape attachment');hw=col('05 Purchased hardware and stationary wiring')
    floor=make_floor(c,dark);keeper=make_keeper(c,accent);body=make_body(c,dark)
    carrier=make_carrier(c,accent);deck=make_deck(c,dark);spacer,cap=make_bearing_parts(c,dark)
    rotor,washer,dog=make_rotor(motion,dark)
    pcase,pbezel=make_case(port,case_mat,accent)
    lcase,lbezel=make_case(land,case_mat,accent,True)
    visual_hardware(hw,metal,blue,black,accent)
    pp=dummy_phone(port,metal,glass);lp=dummy_phone(land,metal,glass,True)
    red=mat('Servo power red',(.55,.015,.01));yellow=mat('D9 PWM amber',(.95,.45,.02))
    wire('6V to servo red',[(18,53,20),(14,-36,23),(-32,-30,29),(-39,0,38),(-30,0,39)],red,hw)
    wire('Common ground',[(17,-36,24),(5,-30,28),(-35,4,29),(-30,1,39)],black,hw)
    wire('Uno D9 signal',[(3,23,22),(-22,32,29),(-38,8,31),(-30,2,39)],yellow,hw)
    # Fit coupons: print before the full assembly.
    coupons=col('06 Fit coupons (hidden)')
    gauge=rr('bearing and spindle gauge',76,53,4,0,5)
    bore(gauge,42.3,-1,6,(-13,0));bore(gauge,29.8,-1,6,(24,0))
    part(gauge,'13_bearing_fit_coupon',coupons,case_mat)
    for obj,name,orientation in PRINT:export_mesh(obj,name,orientation)
    checks={
      'range_degrees':180,'servo':'DS3218 270-degree POSITIONAL variant',
      'pulse_nominal_us':[833,1500,2167], 'phone_mm':[PHONE_W,PHONE_H,PHONE_T],
      'phone_mass_g':199,'phone_clearance_per_side_mm':FIT,
      'bearing':'2 x 6806 / 61806, 30 x 42 x 7 mm',
      'bearing_seat_diametral_clearance_mm':.30,
      'spindle_diametral_clearance_mm':.20,
      'rotor_to_cap_axial_clearance_mm':3,
      'spindle_retention_axial_play_mm':.30,
      'drive_dog_end_clearance_mm':4,
      'drive_dog_flank_clearance_mm':.20,
      'phone_bottom_above_rotor_mm':18,
      'validation_scope':'Closed single-component positive-volume STL meshes; analytic assembly clearances. Physical fit, vibration, retention and torque require a prototype test.',
      'parts':REPORT}
    (OUT/'mesh_validation.json').write_text(json.dumps(checks,indent=2)+'\n')
    # Put the actual .blend into metres; lighting therefore has physically sensible units.
    for o in list(s.objects):
        o.location*=.001;o.scale*=.001
    visible(list(coupons.objects),False)
    visible(list(land.objects),False)
    # Animate complete moving payload; stationary wire harness never follows it.
    pivot=bpy.data.objects.new('PAN 180 degrees — scrub frames 1 to 181',None);s.collection.objects.link(pivot)
    for o in [rotor,washer,dog,*port.objects,*land.objects]:
        world=o.matrix_world.copy();o.parent=pivot;o.matrix_world=world
    for frame,angle in ((1,-90),(91,0),(181,90)):
        pivot.rotation_euler.z=math.radians(angle);pivot.keyframe_insert(data_path='rotation_euler',frame=frame)
    s.frame_start=1;s.frame_end=181;s.render.fps=30;s.frame_set(91)
    floor_mat=mat('Backdrop',(.075,.098,.13),.05,.5)
    stage=cylinder('Studio ground',2000,-5,-4);stage.scale=(.001,)*3;stage.location*=.001;material(stage,floor_mat)
    area('Warm softbox',(-.35,-.35,.50),45,.35,(1,.86,.74))
    area('Cool softbox',(.40,-.1,.40),32,.3,(.69,.84,1))
    area('Edge softbox',(0,.3,.40),55,.24,(1,.53,.31))
    cam=camera((.32,-.48,.32),(0,0,.148),.355)
    render('01_portrait',cam)
    visible(list(port.objects),False);visible(list(land.objects),True)
    cam.location=(.36,-.48,.28);cam.data.ortho_scale=.32;look_at(cam,(0,0,.111))
    render('02_landscape',cam,(1400,1050))
    # Back view exposes camera window, spine and connector clearance.
    cam.location=(-.30,.48,.30);look_at(cam,(0,0,.113))
    render('03_landscape_camera_side',cam,(1400,1050))
    visible(list(land.objects),False);visible(list(port.objects),True)
    visible([body],False)
    cam.location=(.36,-.48,.32);cam.data.ortho_scale=.36;look_at(cam,(0,0,.14))
    render('04_internal_packaging',cam)
    visible([body],True)
    # Separated layers; restore transforms before saving the editable assembly.
    saved={o:o.location.copy() for o in [body,deck,rotor,cap,spacer,washer,dog,*port.objects]}
    body.location.x+=.145
    deck.location.z+=.085;cap.location.z+=.085;spacer.location.z+=.085
    rotor.location.z+=.135;washer.location.z+=.135;dog.location.z+=.04
    for o in port.objects:o.location.z+=.185
    cam.location=(.55,-.7,.46);cam.data.ortho_scale=.58;look_at(cam,(.04,0,.24))
    render('05_exploded',cam,(1400,1400))
    for o,location in saved.items():o.location=location
    visible(list(port.objects),False);visible(list(land.objects),True)
    cam.location=(.34,-.48,.29);cam.data.ortho_scale=.33;look_at(cam,(0,0,.113))
    for frame,name in ((1,'06_pan_left'),(181,'07_pan_right')):
        s.frame_set(frame);render(name,cam,(1200,900))
    s.frame_set(91)
    # Save with portrait active and a usable material viewport.
    visible(list(land.objects),False);visible(list(port.objects),True)
    cam.location=(.32,-.48,.32);cam.data.ortho_scale=.355;look_at(cam,(0,0,.148))
    for screen in bpy.data.screens:
        for area_item in screen.areas:
            if area_item.type=='VIEW_3D':
                area_item.spaces.active.clip_end=10
                area_item.spaces.active.region_3d.view_distance=.5
                area_item.spaces.active.region_3d.view_location=(0,0,.14)
    bpy.ops.wm.save_as_mainfile(filepath=str(OUT/'FollowCam_180_iPhone16Pro.blend'))
    print('VERIFIED_STL_PARTS',len(REPORT))

if __name__=='__main__':main()
