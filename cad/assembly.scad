// FollowCam ASSEMBLY VISUAL — full rig as it looks printed + installed.
// Reuses the real printed-part modules from followcam-rig.scad and poses them
// on a modeled tripod (geometry from tripod-photos + MEASUREMENTS.md).
// pan_angle drives the head/handle/arm together (servo ~1:1 with pan axis).

use <followcam-rig.scad>

pan_angle = 90;          // 40..140, 90 = camera straight ahead (+X = court)
$fn = 80;

/* dims mirrored from followcam-rig.scad (use<> imports modules, not vars) */
column_d  = 15;
shaft_d   = 8.5;
arm_reach = 95;

/* colors */
C_TRIPOD = [0.10, 0.10, 0.11];
C_RUBBER = [0.16, 0.16, 0.17];
C_CHROME = [0.78, 0.79, 0.82];
C_PRINT  = [1.00, 0.45, 0.10];   // orange PLA
C_SERVO  = [0.15, 0.38, 0.90];   // SG90 blue
C_HORN   = [0.95, 0.95, 0.95];
C_PHONE  = [0.06, 0.06, 0.07];
C_SCREEN = [0.15, 0.22, 0.35];

/* key heights (z=0 at crown top / column exit) */
head_base_z = 160;       // measured free column height
clamp_z     = 112;       // printed clamp sits just below the head (h=40)
servo_x     = -22;       // servo output shaft offset from pan axis (shelf side)
arm_z       = 148;       // fork-arm bar bolted on the horn

/* ---------------- tripod ---------------- */
module tripod() {
  color(C_TRIPOD) {
    // crown + leg mounts
    translate([0, 0, -28]) cylinder(d=54, h=28);
    for (a = [90, 210, 330]) rotate([0, 0, a]) {
      translate([0, 0, -14]) rotate([0, 22, 0]) {
        translate([0, 0, -270]) cylinder(d=19, h=270);          // upper section
        translate([0, 0, -520]) cylinder(d=14, h=260);          // lower section
        translate([0, 0, -258]) cylinder(d=24, h=26);           // flip lock
        translate([0, 0, -532]) cylinder(d1=20, d2=15, h=14);   // foot
      }
    }
    // column crank housing
    translate([0, 24, -18]) rotate([-90, 0, 0]) cylinder(d=16, h=14);
  }
  color([0.35, 0.35, 0.37]) translate([0, 0, -28]) cylinder(d=column_d, h=28 + head_base_z); // center column
}

/* ---------------- head + phone + handle (pans together) ---------------- */
module head_assembly() {
  // 3-way head body
  color(C_TRIPOD) {
    translate([0, 0, head_base_z]) cylinder(d=56, h=12);
    translate([-24, -21, head_base_z + 12]) cube([48, 42, 44]);
    translate([0, 24, head_base_z + 34]) rotate([-90, 0, 0]) cylinder(d=26, h=8); // tilt drum
  }
  // phone clamp: bottom jaw, back plate, spring top jaw
  color(C_RUBBER) {
    translate([-36, -14, head_base_z + 56]) cube([72, 28, 8]);
    translate([-10, -14, head_base_z + 64]) cube([20, 6, 160]);
    translate([-30, -13, head_base_z + 208]) cube([60, 26, 12]);
  }
  // phone, portrait, rear camera facing +X (the court)
  color(C_PHONE) translate([-2, -37, head_base_z + 64]) cube([9, 74, 152]);
  color(C_SCREEN) translate([-3.2, -33, head_base_z + 70]) cube([1.5, 66, 140]);
  color([0.02, 0.02, 0.02]) translate([6.2, -32, head_base_z + 176]) cube([2, 22, 34]); // camera block
  // pan handle: chrome shaft out of the head, then rubber grip, ~8 deg down
  translate([-20, 0, head_base_z + 22]) rotate([0, -8, 0]) {
    color(C_CHROME) rotate([0, -90, 0]) cylinder(d=shaft_d, h=95);
    color(C_RUBBER) translate([-212, 0, 0]) rotate([0, 90, 0]) cylinder(d1=30, d2=22, h=120);
  }
}

/* ---------------- printed parts + servo ---------------- */
module servo_sg90() {
  color(C_SERVO) {
    translate([-servo_l_vis/2, -6.45, 0]) cube([servo_l_vis, 12.9, 24]);
    translate([-servo_l_vis/2 - 4, -6.45, 17]) cube([servo_l_vis + 8, 12.9, 2.6]); // flange
    translate([-servo_l_vis/2 + 5.9, 0, 24]) cylinder(d=11.8, h=4);               // gear cap
  }
  color(C_HORN) translate([-servo_l_vis/2 + 5.9, 0, 28]) cylinder(d=4.6, h=4);    // output shaft
}
servo_l_vis = 23.4;

module followcam_parts() {
  // printed column clamp (module's shelf points +X; handle side is -X)
  color(C_PRINT) translate([0, 0, clamp_z]) rotate([0, 0, 180]) column_clamp();
  // SG90 dropped into the shelf pocket, shaft up, shaft next to the pan axis
  translate([servo_x - 5.9 + servo_l_vis/2 - servo_l_vis, 0, clamp_z + 2])
    translate([servo_l_vis/2, 0, 0]) rotate([0, 0, 180]) servo_sg90();
  // printed fork arm on the horn, panning with the head
  color(C_PRINT) translate([servo_x, 0, arm_z])
    rotate([0, 0, 180 + (pan_angle - 90)]) fork_arm();
}

/* ---------------- scene ---------------- */
tripod();
rotate([0, 0, pan_angle - 90]) head_assembly();
followcam_parts();
