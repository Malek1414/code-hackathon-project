#include "pod_motion.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main() {
  assert(podPulse(0) == 833 && podPulse(90) == 1500 && podPulse(180) == 2167);
  assert(podPulse(-1) == 833 && podPulse(181) == 2167);
  float result = -1;
  assert(podCommand("A40", 3, result) && result == 0);
  assert(podCommand("A90", 3, result) && fabsf(result - 90) < .001);
  assert(podCommand("A140", 4, result) && fabsf(result - 180) < .001);
  assert(podCommand("P180", 4, result) && result == 180);
  const char *bad[] = {"A", "A1x", "P-1", "P181", "A900", "P9999", "nan", "Z90", "A90junk"};
  for (const char *text : bad) assert(!podCommand(text, strlen(text), result));
  PodMotion motion;
  for (const float target : {0.0f, 180.0f, 90.0f}) {
    motion.target = target;
    for (int i=0; i<600; ++i) {
      const float before = motion.angle;
      motion.step(.02f);
      assert(motion.angle >= 0 && motion.angle <= 180);
      assert(fabsf(motion.velocity) <= 30.001);
      assert(fabsf(motion.angle - before) <= .601);
    }
    assert(fabsf(motion.angle - target) < .001);
  }
  puts("PASS: protocol validation, nominal pulse mapping, full 180-degree sweep and speed bound");
}
