#pragma once
#include <math.h>
#include <stddef.h>

// DS3218 270-degree positional variant: its central 180 degrees at 50 Hz.
// Calibrate on the purchased servo; these nominal values follow its datasheet.
static const int POD_LEFT_US = 833;
static const int POD_RIGHT_US = 2167;

inline float podClamp(float value, float lo, float hi) {
  return value < lo ? lo : (value > hi ? hi : value);
}
inline int podPulse(float degrees) {
  return (int)lroundf(POD_LEFT_US + podClamp(degrees, 0, 180) *
                    (POD_RIGHT_US - POD_LEFT_US) / 180.0f);
}
inline bool podCommand(const char *line, size_t length, float &degrees) {
  if (length < 2 || length > 4 || (line[0] != 'A' && line[0] != 'P')) return false;
  unsigned value = 0;
  for (size_t i = 1; i < length; ++i) {
    if (line[i] < '0' || line[i] > '9') return false;
    value = value * 10 + (line[i] - '0');
  }
  if (value > 180) return false;
  // Existing app/trackers emit A40..A140. Their full span now covers physical 180°.
  // P0..P180 is the new explicit physical-angle bench/calibration command.
  degrees = line[0] == 'A' ? (podClamp(value, 40, 140) - 40) * 1.8f : value;
  return true;
}

struct PodMotion {
  float angle = 90;
  float velocity = 0;
  float target = 90;
  void step(float dt) {
    dt = podClamp(dt, 0, 0.03f);
    const float acceleration = 60.0f;
    const float speed = 30.0f;
    const float error = target - angle;
    const float wanted = (error < 0 ? -1 : 1) *
        fminf(speed, sqrtf(2 * acceleration * fabsf(error)));
    velocity += podClamp(wanted - velocity, -acceleration * dt, acceleration * dt);
    const float delta = velocity * dt;
    if ((error >= 0 && delta >= error) || (error < 0 && delta <= error)) {
      angle = target; velocity = 0;
    } else {
      angle = podClamp(angle + delta, 0, 180);
    }
  }
};
