// FollowCam pan servo — reads "A<angle>\n" (0-180) over serial at 115200.
// Slew-rate limited so the camera glides instead of snapping.
// Wiring (MG996R/SG90): signal -> pin 9, power -> 5V (MG996R prefers external
// 5-6V supply, common ground with the board!), GND -> GND.

#include <Servo.h>

const int SERVO_PIN = 9;
const int ANGLE_MIN = 40, ANGLE_MAX = 140;   // keep inside linkage range
const float MAX_DEG_PER_STEP = 2.0;          // per 15ms tick -> ~133 deg/s max
const unsigned long TICK_MS = 15;

Servo pan;
float current = 90, target = 90;
unsigned long lastTick = 0;
String buf;

void setup() {
  Serial.begin(115200);
  pan.attach(SERVO_PIN);
  pan.write((int)current);
}

void loop() {
  while (Serial.available()) {
    char c = Serial.read();
    if (c == '\n') {
      if (buf.length() > 1 && buf[0] == 'A') {
        long a = buf.substring(1).toInt();
        target = constrain((float)a, (float)ANGLE_MIN, (float)ANGLE_MAX);
      }
      buf = "";
    } else if (buf.length() < 8) {
      buf += c;
    }
  }

  unsigned long now = millis();
  if (now - lastTick >= TICK_MS) {
    lastTick = now;
    float d = target - current;
    if (d > MAX_DEG_PER_STEP) d = MAX_DEG_PER_STEP;
    if (d < -MAX_DEG_PER_STEP) d = -MAX_DEG_PER_STEP;
    current += d;
    pan.write((int)(current + 0.5f));
  }
}
