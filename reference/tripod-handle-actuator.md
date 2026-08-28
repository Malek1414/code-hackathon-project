# Basketball Tracking: 3D-Printed Tripod Handle Actuator

Iteration on the basketball tracking idea — same core concept, new mechanism for camera movement.

Instead of a fully custom rig, build a 3D-printed actuator (at the Code University 3D printer lab) that grips and moves the existing tripod handle. The handle steers the phone's field of view horizontally (left to right) to follow the ball on the court.

- Phone mounted on the tripod, connected to a laptop for processing
- 3D-printed "hand"/clamp fits over the tripod handle and pans it horizontally
- Ball location drives the panning

**Tracking approach**
- Chip inside the ball, or a stick-on tag/sticker anyone can apply to any ball
- Same idea for players: stickers as identifiable markers
- Similar in spirit to how FIFA / NBA 2K engineers capture movement data for their models

**Open questions**
- Which actuator/servo and how it mounts to the handle
- Whether tag detection or pure vision is more reliable
- Vertical (tilt) movement — horizontal only for now
