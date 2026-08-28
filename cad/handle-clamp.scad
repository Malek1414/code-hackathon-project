// FollowCam — tripod handle clamp + servo leg bracket
// Parametric OpenSCAD. PLACEHOLDER DIMENSIONS — replace from cad/MEASUREMENTS.md
// before slicing. Print: PLA/PETG, 0.2–0.28mm layers, 25–35% infill, no supports.
//
// Render one part at a time via PART, export STL:
//   PART = "clamp_top" | "clamp_bottom" | "servo_bracket"

PART = "clamp_bottom";

/* ---------- measured on the real tripod (EDIT THESE) ---------- */
handle_d      = 16;    // handle outer diameter at clamp point (mm)
leg_d         = 24;    // leg/column tube diameter where bracket straps on
/* ---------- servo (MG996R-class defaults; SG90 = 23 x 12.2 body) ---------- */
servo_body_l  = 40.7;  // along the long axis
servo_body_w  = 19.7;
servo_body_h  = 36;    // below the mounting flange
/* ---------- tunables ---------- */
clamp_len     = 40;    // length of clamp along the handle
wall          = 4;
tab_w         = 10;    // bolt/zip-tie flange width
bolt_d        = 3.4;   // M3 clearance
arm_len       = 35;    // pushrod arm sticking off the clamp
rod_hole_d    = 2.2;   // pushrod wire holes (multiple radii for tuning)
fit_tol       = 0.4;   // radial slop; increase if clamp won't close on rubber grip
ziptie_w      = 5.5;
ziptie_t      = 2.5;
$fn = 64;

bore_r  = handle_d/2 + fit_tol;
outer_r = bore_r + wall;

/* One clamshell half: half-pipe + two side flanges with bolt/zip-tie slots. */
module clamp_half(with_arm=false) {
  difference() {
    union() {
      // half cylinder shell
      difference() {
        cylinder(r=outer_r, h=clamp_len, center=false);
        translate([-outer_r-1, -2*outer_r, -1]) cube([2*outer_r+2, 2*outer_r, clamp_len+2]);
      }
      // flanges
      for (sx = [-1, 1])
        translate([sx*(outer_r + tab_w/2) - tab_w/2, 0, 0])
          cube([tab_w, wall, clamp_len]);
      // pushrod arm (on one half only), sticking radially up from the shell
      if (with_arm)
        translate([-wall/2, outer_r - 1, clamp_len/2 - wall/2])
          cube([wall, arm_len + 1, wall*2]);
    }
    // bore
    translate([0, 0, -1]) cylinder(r=bore_r, h=clamp_len+2);
    // two bolt holes + a zip-tie slot per flange
    for (sx = [-1, 1]) {
      for (z = [clamp_len*0.25, clamp_len*0.75])
        translate([sx*(outer_r + tab_w/2), -1, z])
          rotate([-90, 0, 0]) cylinder(d=bolt_d, h=wall+2);
      translate([sx*(outer_r + tab_w/2) - ziptie_w/2, -1, clamp_len/2 - ziptie_t/2])
        cube([ziptie_w, wall+2, ziptie_t]);
    }
    // pushrod holes along the arm at three radii (linkage tuning)
    if (with_arm)
      for (y = [outer_r + arm_len*0.5, outer_r + arm_len*0.75, outer_r + arm_len*0.95])
        translate([-wall/2 - 1, y, clamp_len/2 + wall/2])
          rotate([0, 90, 0]) cylinder(d=rod_hole_d, h=wall+2);
  }
}

/* Servo cradle with a V-notch base that zip-ties to the tripod leg. */
module servo_bracket() {
  cw = servo_body_w + 2*wall;
  cl = servo_body_l + 2*wall;
  ch = 16;                       // cradle side height — servo held by zip ties over top
  base_h = 14;
  difference() {
    union() {
      translate([0, 0, base_h]) cube([cl, cw, ch]);        // cradle walls block
      cube([cl, cw, base_h]);                              // base block
    }
    // servo pocket
    translate([wall, wall, base_h]) cube([servo_body_l, servo_body_w, ch+1]);
    // V-groove along the base underside to seat on the round leg
    translate([-1, cw/2, -leg_d*0.18])
      rotate([45, 0, 0]) rotate([0, 90, 0])
        cube([leg_d, leg_d, cl+2], center=false);
    // zip-tie tunnels: two through the base (strap to leg), two through walls (strap servo)
    for (x = [cl*0.2, cl*0.8]) {
      translate([x - ziptie_t/2, -1, base_h*0.45]) cube([ziptie_t, cw+2, ziptie_w]);
      translate([x - ziptie_t/2, -1, base_h + ch*0.5]) cube([ziptie_t, cw+2, ziptie_w]);
    }
  }
}

if (PART == "clamp_top")     clamp_half(with_arm=false);
if (PART == "clamp_bottom")  clamp_half(with_arm=true);
if (PART == "servo_bracket") servo_bracket();
