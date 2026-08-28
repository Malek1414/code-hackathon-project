// FollowCam rig v2 — column-mounted servo + fork arm (from tripod photos, Aug 27)
// Mechanism: servo clamps to the center column just below the head, output shaft
// UP and next to the pan axis -> servo angle ~ pan angle (near 1:1). A printed arm
// on the servo horn ends in a U-fork that captures the handle's metal shaft and
// pushes it left/right. The open fork absorbs axis offset + the handle's downward
// angle. PLACEHOLDER DIMS — fill from cad/MEASUREMENTS.md before slicing.
//
// PART = "column_clamp" | "fork_arm"   (print one of each; 25-35% infill, no supports)

PART = "fork_arm";

/* ---------- measured on the real tripod (MEASURED Aug 27 ~21:45) ---------- */
column_d      = 15;    // MEASURED: exactly 15mm
column_free_h = 160;   // MEASURED: ~160mm free after raising elevator column
shaft_d       = 8.5;   // MEASURED: 8-9mm chrome shaft; 8.5 + slot clearance
arm_reach     = 95;    // DERIVED: 230mm to grip tip - ~140 grip - 10 = chrome mid ~80; window covers 60-95
fork_drop     = 45;    // tall prongs: huge vertical catch window (M5 not needed)
/* ---------- servo: SG90 from the Elegoo Uno kit (confirmed Aug 27) ----------
   For an MG996R upgrade set servo_l=40.7, servo_w=20.2, horn_screw_d=4.2 */
servo_l       = 23.4;  // SG90 body length + tol
servo_w       = 12.9;  // SG90 body width + tol
horn_screw_d  = 3.0;   // SG90 horn center screw clearance (M2 screw, loose)
/* ---------- tunables ---------- */
wall   = 5;
bolt_d = 3.4;          // M3 clearance
ziptie_w = 5.5; ziptie_t = 2.5;
fit_tol = 0.5;
$fn = 64;

/* --- Part 1: column clamp -------------------------------------------------
Two mirrored halves in one print, joined by M3 bolts (or zip ties) around the
column; one half carries a shelf the servo drops into, shaft up. */
module column_clamp() {
  bore_r = column_d/2 + fit_tol;
  ring_r = bore_r + wall;
  h = min(column_free_h - 5, 40);          // clamp height on the column
  shelf_l = servo_l + 2*wall;
  shelf_w = servo_w + 2*wall + 12;         // +6mm tie ledge outside each rail
  difference() {
    union() {
      cylinder(r=ring_r, h=h);                                   // ring
      // servo shelf sticking out radially (servo drops in shaft-up, flange on plate)
      translate([ring_r - 2, -shelf_w/2, 0]) cube([shelf_l, shelf_w, wall]);
      // shelf side rails to box the servo body
      for (sy = [-1, 1])
        translate([ring_r - 2, sy*(servo_w/2 + wall/2) - wall/2, 0])
          cube([shelf_l, wall, 14]);
    }
    translate([0, 0, -1]) cylinder(r=bore_r, h=h+2);             // column bore
    // split the ring: subtract a slit so it can flex closed with bolts
    translate([-ring_r-1, -1.2, -1]) cube([ring_r+1, 2.4, h+2]);
    // bolt bosses through the slit side
    for (z = [h*0.25, h*0.75])
      translate([-ring_r+wall/2, 0, z]) rotate([90, 0, 0])
        cylinder(d=bolt_d, h=3*wall, center=true);
    // servo body drop-in pocket through the shelf
    translate([ring_r - 2 + wall, -servo_w/2, -1]) cube([servo_l, servo_w, wall+16]);
    // zip-tie slots through the tie ledges (2 per side): tie loops over the
    // servo body/flange and down through these, cinching it into the pocket
    for (fx = [0.28, 0.72]) for (sy = [-1, 1])
      translate([ring_r - 2 + wall + servo_l*fx - 3,
                 sy*(servo_w/2 + wall + 3) - 1.5, -1])
        cube([6, 3, wall + 2]);
    // zip-tie tunnels around ring (backup if bolts strip)
    for (a = [60, 180, 300]) rotate([0, 0, a])
      translate([ring_r - wall/2, -ziptie_w/2, h/2 - ziptie_t/2])
        cube([wall+2, ziptie_w, ziptie_t]);
  }
}

/* --- Part 2: fork arm -----------------------------------------------------
Flat bar; one end bolts to the servo horn (center + 2 outer holes), the other
end drops down and opens into a U-fork that captures the handle shaft. */
module fork_arm() {
  bar_w = 16; bar_t = 6;
  slot_w = shaft_d + 4.5;                 // 13mm: loose on chrome, swallows early grip taper
  fork_wall = 5;
  fork_ow  = slot_w + 2*fork_wall;   // fork width (y)
  fork_len = 35;                     // fork trough length (x): rod can engage
                                     // anywhere in radius window [60,95]mm
  difference() {
    union() {
      // horn plate + main bar
      hull() {
        cylinder(d=bar_w+6, h=bar_t);
        translate([arm_reach - 30, -bar_w/2, 0]) cube([1, bar_w, bar_t]);
      }
      // long fork trough at the far end
      translate([arm_reach - fork_len, -fork_ow/2, 0])
        cube([fork_len, fork_ow, bar_t + fork_drop]);
    }
    // horn screws: center hole + a row of small holes at several radii so the
    // servo's own plastic horn can bolt/tie on whichever holes line up
    // (fits SG90 single-arm and MG996R cross horns alike)
    translate([0, 0, -1]) cylinder(d=horn_screw_d, h=bar_t+2);
    for (r = [5, 8, 11, 14]) for (sx = [-1, 1])
      translate([sx*r, 0, -1]) cylinder(d=2.2, h=bar_t+2);
    // U-slot: opens upward, runs through both x faces; deep + long slot means
    // vertical AND radial alignment are both forgiving (8mm root above bar)
    translate([arm_reach - fork_len - 1, -slot_w/2, bar_t + 8])
      cube([fork_len + 2, slot_w, fork_drop + 2]);
  }
}

if (PART == "column_clamp") column_clamp();
if (PART == "fork_arm")     fork_arm();
