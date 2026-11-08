// Compile as C++ with this directory on the include path; run once per case name.
#include <cassert>
#include <iostream>
#include "../voltometer.ino"

void tick(unsigned long elapsed = 0) { clockMs += elapsed; loop(); }
void buttons(bool forward, bool backward, unsigned long stableMs = 30) {
  pins[PIN_FORWARD] = forward;
  pins[PIN_BACKWARD] = backward;
  tick();
  tick(stableMs);
}
void home(long position = -2000) {
  setup();
  pins[PIN_LIMIT] = HIGH;
  tick();
  assert(state == READY);
  pins[PIN_LIMIT] = LOW;
  stepper.setCurrentPosition(position);
}
void finish() { stepper.finish(); tick(); }

void distance() {
  home();
  const long increment = static_cast<long>(7.0 * STEPS_PER_MM);
  assert(increment == 1400);
  buttons(true, false, 29);
  assert(stepper.target == -2000);
  tick(1);
  assert(stepper.target == -2000 + increment);
  buttons(false, false);
  assert(stepper.target == -600);  // Release must not stop the move.
  finish();
  buttons(false, true);
  assert(stepper.target == -2000);
  assert(stepper.maxSpeed == MAX_SPEED && stepper.acceleration == ACCELERATION);
}

void debounce() {
  home();
  buttons(true, false, 10);
  buttons(false, false, 10);
  buttons(true, false, 29);
  assert(stepper.moves == 0);
  tick(1);
  assert(stepper.moves == 1);
  finish();
  tick(1000);
  assert(stepper.moves == 1);  // Holding cannot repeat.
  buttons(false, false, 10);
  buttons(true, false, 30);
  assert(stepper.moves == 1);  // Release bounce cannot re-arm.
  buttons(false, false);
  buttons(false, true);
  assert(stepper.moves == 2);
}

void busy() {
  home();
  buttons(false, true);
  buttons(false, false);
  buttons(true, false, 10);  // A new press queues after debounce, even when busy.
  tick(20);
  assert(queueCount == 1 && stepper.target == -3400);
  buttons(false, false);
  buttons(false, true);
  assert(queueCount == 2);
  finish();
  assert(stepper.moves == 2 && stepper.target == -2000 && queueCount == 1);
  finish();
  assert(stepper.moves == 3 && stepper.target == -3400 && queueCount == 0);
  finish(); tick(1000);
  assert(stepper.moves == 3);  // Holding the last button never adds moves.
}

void boundaries() {
  home(0);
  buttons(true, false);
  assert(stepper.moves == 0);
  buttons(false, false);
  buttons(false, true);
  finish();
  buttons(false, false);
  buttons(true, false);
  assert(stepper.target == 0);  // Exact boundary is accepted.
  finish();
  buttons(false, false);
  stepper.setCurrentPosition(-MAX_TRAVEL_STEPS + 1400);
  buttons(false, true);
  assert(stepper.target == -MAX_TRAVEL_STEPS);
  finish();
  int moves = stepper.moves;
  buttons(false, false);
  buttons(false, true);
  assert(stepper.moves == moves);
  buttons(false, false);
  stepper.setCurrentPosition(-MAX_TRAVEL_STEPS + 1000);
  buttons(false, true);
  assert(stepper.moves == moves);  // No shortened final move.
  buttons(false, false);
  stepper.setCurrentPosition(-1000);
  buttons(true, false);
  assert(stepper.moves == moves);
}

void simultaneous() {
  home();
  buttons(true, true);
  buttons(true, false);
  assert(stepper.moves == 0);
  buttons(false, false);
  buttons(false, true);
  assert(stepper.moves == 1 && stepper.target == -3400);
}

void keyboard() {
  home();
  Serial.input = "BFBH"; tick();  // Every press in one serial batch must survive.
  assert(stepper.target == -3400 && queueCount == 2);
  Serial.input = "H"; tick(100);
  assert(queueCount == 2);  // Heartbeats do not add or cancel moves.
  finish();
  assert(stepper.target == -2000 && queueCount == 1);
  finish();
  assert(stepper.target == -3400 && queueCount == 0);
  finish();
  buttons(false, true);
  Serial.input = "F"; tick();
  assert(stepper.target == -4800 && queueCount == 1);  // Both controls share the FIFO.
  finish();
  assert(stepper.target == -3400);
}

void stop() {
  home();
  Serial.input = "BFB"; tick();
  assert(queueCount == 2);
  stepper.position = -2500;  // Stop part-way through a move.
  Serial.input = "S"; tick();
  assert(stepper.target == -2500 && queueCount == 0);
  Serial.input = "H"; tick(100);
  assert(stepper.target == -2500 && queueCount == 0);
  Serial.input = "B"; tick();
  assert(stepper.target == -3900);  // New moves start at the stopped position.
  buttons(false, true);
  assert(queueCount == 1);
  tick(301);
  assert(!serialConnected && queueCount == 0 && stepper.distanceToGo() == 0);
}

void projected_limits() {
  home();
  Serial.input = "FFF"; tick();
  assert(stepper.target == -600 && queueCount == 0);
  assert(Serial.output.find("Move rejected - travel limit") != std::string::npos);
  Serial.input = "BF"; tick();
  assert(queueCount == 2);  // A reversal is checked from the projected endpoint.
  finish();
  assert(stepper.target == -2000);
  finish();
  assert(stepper.target == -600);
}

void capacity() {
  home();
  for (int i = 0; i < MOVE_QUEUE_SIZE + 1; ++i) Serial.input += i % 2 ? 'F' : 'B';
  tick();
  assert(queueCount == MOVE_QUEUE_SIZE - 1);
  assert(Serial.output.find("Move rejected - queue full") != std::string::npos);
  for (int i = 0; i < MOVE_QUEUE_SIZE; ++i) finish();
  assert(stepper.moves == MOVE_QUEUE_SIZE && queueCount == 0);
  Serial.input = "BFB"; tick();  // Exercise ring-buffer wraparound after draining it.
  assert(stepper.target == -3400 && queueCount == 2);
}

void safety() {
  home();
  buttons(true, false);
  buttons(false, false);
  buttons(false, true);
  assert(queueCount == 1);
  pins[PIN_LIMIT] = HIGH;
  tick();
  assert(stepper.distanceToGo() == 0 && queueCount == 0);
  pins[PIN_LIMIT] = LOW;
  tick(100);
  assert(stepper.moves == 1);
  assert(Serial.output.find("Voltage:") != std::string::npos);
  assert(Serial.output.find(",Distance_mm:") != std::string::npos);
}

void homing() {
  setup();
  pins[PIN_FORWARD] = HIGH;
  tick();
  assert(state == FAULT);  // Emergency stop must bypass debounce.
  pins[PIN_FORWARD] = LOW;
  pins[PIN_BACKWARD] = HIGH;
  setup(); tick();
  assert(state == FAULT);
  pins[PIN_BACKWARD] = LOW;
  setup(); Serial.input = "F"; tick();
  assert(state == FAULT);
  setup(); Serial.input = "S"; tick();
  assert(state == FAULT);  // Space also stops homing.
}

int main(int argc, char** argv) {
#ifdef _MSC_VER
  _set_error_mode(_OUT_TO_STDERR);
  _set_abort_behavior(0, _WRITE_ABORT_MSG | _CALL_REPORTFAULT);
#endif
  assert(argc == 2);
  const std::string name = argv[1];
  if (name == "distance") distance();
  else if (name == "debounce") debounce();
  else if (name == "busy") busy();
  else if (name == "boundaries") boundaries();
  else if (name == "simultaneous") simultaneous();
  else if (name == "keyboard") keyboard();
  else if (name == "stop") stop();
  else if (name == "projected_limits") projected_limits();
  else if (name == "capacity") capacity();
  else if (name == "safety") safety();
  else if (name == "homing") homing();
  else { std::cerr << "Unknown case: " << name << '\n'; return 1; }
  std::cout << "PASS " << name << '\n';
}
