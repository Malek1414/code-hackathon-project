// FollowCam 180 pod ONLY. Do not flash onto the old tripod-handle linkage.
// Uno R3: pin 9 -> DS3218 signal; GND -> servo supply GND.
// External regulated 6 V / 3 A -> servo power. Uno powered by laptop USB.
// DS3218 270-degree POSITIONAL version; middle 180° travel, no continuous rotation.
#include <Servo.h>
#include "pod_motion.h"

Servo servo;
PodMotion motion;
char command[5];
unsigned char commandLength = 0;
bool overflow = false;
unsigned long lastTick = 0;

void setup() {
  Serial.begin(115200);
  servo.writeMicroseconds(podPulse(90));
  servo.attach(9, 500, 2500);
  servo.writeMicroseconds(podPulse(90));
  lastTick = millis();
}

void loop() {
  while (Serial.available()) {
    const char ch = (char)Serial.read();
    if (ch == '\r') continue;
    if (ch == '\n') {
      float target;
      if (!overflow && podCommand(command, commandLength, target)) motion.target = target;
      commandLength = 0; overflow = false;
    } else if (commandLength < sizeof(command)) {
      command[commandLength++] = ch;
    } else {
      overflow = true;
    }
  }
  const unsigned long now = millis();
  if (now - lastTick >= 20) {
    motion.step((now - lastTick) / 1000.0f);
    lastTick = now;
    servo.writeMicroseconds(podPulse(motion.angle));
  }
}
