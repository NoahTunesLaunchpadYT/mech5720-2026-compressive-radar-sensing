/*
  Stepper step control (A4988 + NEMA11) with homing and distance reporting

  On power-up / reset:
    The motor travels in the pin 13 (FORWARD) direction at HOMING_SPEED
    until the limit switch on pin 9 goes HIGH. That point becomes 0 mm.
    During this calibration pass, pin 12 or pin 13 going HIGH is an
    EMERGENCY STOP: the motor halts immediately and stays halted until
    the Arduino is reset.

  After homing:
    The slider moves START_OFFSET_MM (3.9 mm) off the switch to the
    starting position (position 1). Buttons and arrow keys are ignored
    during this move; 'S' still stops it. The 7 mm steps start from there.
    There are 24 positions: 3.9, 10.9, ... 164.9 mm from the switch.

  After reaching the start position:
    Pin 13 -> FORWARD  (press = 7 mm towards the limit switch)
    Pin 12 -> BACKWARD (press = 7 mm away from the switch)
    Buttons debounce for 30 ms. Release does not cancel; holding never repeats.
    Individual presses queue 7 mm moves in press order, including during motion.
    Both HIGH starts nothing; release both buttons before trying again.
    Moves that cannot fit entirely within positions 1..24 after queued moves
    are rejected. Up to 32 moves may wait in the queue.
    The limit switch still blocks forward motion if it is hit again.
    Backward travel stops at position 24, MAX_TRAVEL_MM (164.9 mm) from the
    switch (soft limit). Forward travel cannot go past position 1.

  Outputs to A4988:
    Pin 11 -> STEP
    Pin 10 -> DIR

  Keyboard stepping:
    Run main.py on the PC. Each arrow press sends one 'F' / 'B' command.
    Holding never repeats; release does not cancel. 'H' is a 100 ms heartbeat.
    Space sends 'S': stop immediately and clear all queued moves.
    Focus loss and closing the app also send 'S'. After 300 ms without a
    heartbeat or command, an attached client's moves stop and the queue clears.
    During homing, an arrow key acts as an emergency stop, like pins 12/13.
    'a'..'x' go straight to position 1..24 in one move: the queue is cleared and the
    carriage redirects immediately (rejected unless READY).
    'C' re-runs the homing/calibration pass (also recovers from FAULT).
    'T' plays a short celebration tune on the motor (step rate = note pitch), only
    when READY and stationary. Each note reverses direction so the carriage just
    wiggles, then it returns to where it started. 'S' stops the tune where it is.
    'G<mm x 100>,<mm/s x 100>' + newline: absolute move to any distance from the switch
    (clamped to 0.5 .. MAX_TRAVEL_MM + 10 mm) at that top speed, with SWEEP_ACCEL_MM so
    the constant-speed part is long; speed 0 = the jog speed. Rejected unless READY.
    Until it stops, "P<micros>,<distance in um>" lines (each 8 hex digits, the distance
    as 32-bit two's complement; hex because decimal printing stalls the steps) report
    the position every 10 ms,
    sampled together, plus one final line once stationary (rail_sync_test.py).

  Serial output at 115200 baud: "Distance_mm:<mm>" every 50 ms (distance from the
  limit switch, 0 until homed), except during 'G' moves, which send P lines instead.
  Everything else printed is an occasional event message.

  Requires the AccelStepper library (Tools > Manage Libraries > "AccelStepper").
*/

#include <AccelStepper.h>

// ---- Pins ----
const uint8_t PIN_FORWARD  = 13;
const uint8_t PIN_BACKWARD = 12;
const uint8_t PIN_STEP     = 11;
const uint8_t PIN_DIR      = 10;
const uint8_t PIN_LIMIT    = 9;   // HIGH when the limit switch is pressed

// ---- Actuator: T6x1 lead screw, 1.8 deg motor, 200 mm stroke ----
// Steps per mm = (200 steps/rev x microsteps) / 1 mm per rev
const float STEPS_PER_REV = 200.0;  // 1.8 degree step angle
const float MICROSTEPS    = 1;    // 1, 2, 4, 8 or 16 to match MS1-MS3 on the A4988
const float MM_PER_REV    = 1.0;    // T6x1 screw: 1 mm per revolution
const float STEPS_PER_MM  = STEPS_PER_REV * MICROSTEPS / MM_PER_REV;  // 200 at full step
const long  UM_PER_STEP   = (long)(1000.0 / STEPS_PER_MM + 0.5);  // 5; P lines need whole um

const long  BUTTON_MOVE_STEPS = (long)(7.0 * STEPS_PER_MM + 0.5);
const unsigned long BUTTON_DEBOUNCE_MS = 30;

// Starting position (position 1): how far off the switch to go after homing
const float START_OFFSET_MM    = 3.9;
const long  START_OFFSET_STEPS = (long)(START_OFFSET_MM * STEPS_PER_MM + 0.5);  // 780 at full step

// Soft limit: 24 positions, 7 mm apart, starting at the offset.
// Position 24 = 3.9 + 7 x 23 = 164.9 mm from the switch. Built from whole
// steps so the last position is never lost to float rounding.
const uint8_t NUM_POSITIONS    = 24;
const long    MAX_TRAVEL_STEPS = START_OFFSET_STEPS + (long)(NUM_POSITIONS - 1) * BUTTON_MOVE_STEPS;  // 32980
const float   MAX_TRAVEL_MM    = MAX_TRAVEL_STEPS / STEPS_PER_MM;  // 164.9 mm
const long    MAX_TARGET_STEPS = MAX_TRAVEL_STEPS + (long)(10.0 * STEPS_PER_MM);  // 'G' run-up room

// ---- Motion settings (tune these, in mm so they survive microstep changes) ----
// An Uno running AccelStepper tops out around 4000 steps/s, so keep
// MAX_SPEED_MM * STEPS_PER_MM below that.
const float MAX_SPEED_MM    = 5.0;  // mm/s while jogging
const float ACCEL_MM        = 10.0;  // mm/s^2
const float HOMING_SPEED_MM = 5.0;   // mm/s; full 200 mm worst case takes 40 s
const bool  INVERT_DIRECTION = true;  // true if "forward" moves away from the switch

const float MAX_SPEED    = MAX_SPEED_MM    * STEPS_PER_MM;  // steps per second
const float ACCELERATION = ACCEL_MM        * STEPS_PER_MM;  // steps per second^2
const float HOMING_SPEED = HOMING_SPEED_MM * STEPS_PER_MM;  // steps per second

const unsigned long PLOT_INTERVAL_MS = 50;  // distance reports, ~20 per second

// ---- Celebration tune ('T'): step rate in Hz is the pitch ----
// Rhythm in semiquavers: ti = 2 (a 0.15 s quaver), tika = 1 + 1, ta = 4.
const unsigned long SEMIQUAVER_MS = 75;
const unsigned long NOTE_GAP_MS   = 20;  // silence at the end of each note, so repeats sound separate
// NOTE_ prefix: A4 etc. are already analog pin names
const float NOTE_C4 = 261.63, NOTE_D4 = 293.66, NOTE_E4 = 329.63, NOTE_F4 = 349.23,
            NOTE_G4 = 392.00, NOTE_A4 = 440.00, NOTE_C5 = 523.25;
// ti tika ti tika ti ti ta, twice
const float TUNE_HZ[] = {
  NOTE_C5, NOTE_C5, NOTE_C5, NOTE_C5, NOTE_C5, NOTE_C5, NOTE_A4, NOTE_G4, NOTE_F4,
  NOTE_G4, NOTE_G4, NOTE_G4, NOTE_G4, NOTE_G4, NOTE_G4, NOTE_E4, NOTE_D4, NOTE_C4};
const uint8_t TUNE_LEN[] = {2, 1, 1, 2, 1, 1, 2, 2, 4,
                            2, 1, 1, 2, 1, 1, 2, 2, 4};
const uint8_t TUNE_NOTES = sizeof(TUNE_LEN);

// ---- 'G' moves (constant-speed sweeps) ----
const float SWEEP_ACCEL_MM     = 50.0;  // mm/s^2: short ramps so most of a sweep is at speed
const float MAX_SWEEP_SPEED_MM = 10.0;  // 1600 steps/s at full step
const long  MIN_TARGET_STEPS   = (long)(0.5 * STEPS_PER_MM);
const unsigned long FAST_REPORT_US = 10000;  // P lines every 10 ms while moving
const long MAX_CATCH_UP_US = 5000;           // a step later than this restarts the schedule

// AccelStepper times each step from when the previous one actually fired, so every
// bit of loop() latency adds up and the motor runs slower than commanded (up to 20%
// at 8 mm/s). This times each step from when it was due instead, so late steps
// catch up and the average rate matches the command.
class CatchUpStepper : public AccelStepper {
public:
  using AccelStepper::AccelStepper;

protected:
  void step(long s) override {
    stepUs = micros();
    stepped = true;
    AccelStepper::step(s);
  }

  unsigned long computeNewSpeed() override {
    unsigned long result = AccelStepper::computeNewSpeed();
    if (!stepped || !_stepInterval) return result;
    stepped = false;
    long late = (long)(stepUs - dueUs);
    if (late < -(long)_stepInterval || late > MAX_CATCH_UP_US) dueUs = stepUs;  // new move or stall
    dueUs += _stepInterval;
    long wait = (long)(dueUs - stepUs);
    _stepInterval = wait > 1 ? wait : 1;
    return result;
  }

private:
  unsigned long stepUs = 0, dueUs = 0;
  bool stepped = false;
};

CatchUpStepper stepper(AccelStepper::DRIVER, PIN_STEP, PIN_DIR);

enum State { HOMING, TO_START, READY, TUNE, FAULT };
State state = HOMING;

uint8_t lastButtons = 0;
unsigned long buttonChangedMs = 0;
bool buttonsArmed = true;

// Absolute targets preserve every queued direction change and bound total travel.
// shortcut: 32 pending moves fit small boards; increase only if more buffering is needed.
const uint8_t MOVE_QUEUE_SIZE = 32;
long moveQueue[MOVE_QUEUE_SIZE];
uint8_t queueHead = 0, queueCount = 0;

// ---- Serial step commands (from the arrow-key app) ----
// 'F' / 'B' enqueue one move; 'H' keeps the connection alive; 'S' stops and clears.
const unsigned long SERIAL_TIMEOUT_MS = 300;
bool serialConnected = false;
unsigned long lastSerialMs = 0;

// Stop instantly, no deceleration ramp
void hardStop() {
  stepper.setCurrentPosition(stepper.currentPosition());  // zeroes speed and target
}

void stopMoves() {
  hardStop();
  queueHead = queueCount = 0;
  buttonsArmed = false;
  buttonChangedMs = millis();
}

void queueMove(int direction) {
  if (state == HOMING) {
    stopMoves();
    state = FAULT;
    Serial.println(F("EMERGENCY STOP during homing - reset to retry"));
    return;
  }
  if (state == TO_START) {
    Serial.println(F("Move rejected - moving to start position"));
    return;
  }
  if (state != READY) return;
  if (queueCount == MOVE_QUEUE_SIZE) {
    Serial.println(F("Move rejected - queue full"));
    return;
  }
  long origin = queueCount > 0
              ? moveQueue[(queueHead + queueCount - 1) % MOVE_QUEUE_SIZE]
              : stepper.currentPosition() + stepper.distanceToGo();
  long target = origin + direction * BUTTON_MOVE_STEPS;
  // Keep every move between position 1 (offset) and position 24 (soft limit)
  if (target > -START_OFFSET_STEPS || target < -MAX_TRAVEL_STEPS) {
    Serial.println(F("Move rejected - travel limit"));
    return;
  }
  moveQueue[(queueHead + queueCount) % MOVE_QUEUE_SIZE] = target;
  queueCount++;
}

void useJogSpeed() {
  stepper.setMaxSpeed(MAX_SPEED);
  stepper.setAcceleration(ACCELERATION);
}

// Absolute move to position n (1..24), replacing anything queued
void goToPosition(uint8_t n) {
  if (state != READY) {
    Serial.println(F("Move rejected - not ready"));
    return;
  }
  queueHead = queueCount = 0;
  useJogSpeed();
  stepper.moveTo(-(START_OFFSET_STEPS + (long)(n - 1) * BUTTON_MOVE_STEPS));
}

bool fastReport = false;
unsigned long lastFastUs = 0;

// 'G': absolute move in hundredths of a mm at a speed in hundredths of a mm/s
void startSweep(long hundredthsMm, long hundredthsMmPerS) {
  if (state != READY) {
    Serial.println(F("Move rejected - not ready"));
    return;
  }
  long target = constrain((long)(hundredthsMm * STEPS_PER_MM / 100.0 + 0.5),
                          MIN_TARGET_STEPS, MAX_TARGET_STEPS);
  queueHead = queueCount = 0;
  if (hundredthsMmPerS > 0) {
    stepper.setMaxSpeed(min(hundredthsMmPerS / 100.0, MAX_SWEEP_SPEED_MM) * STEPS_PER_MM);
    stepper.setAcceleration(SWEEP_ACCEL_MM * STEPS_PER_MM);
  } else {
    useJogSpeed();
  }
  stepper.moveTo(-target);
  fastReport = true;
}

// 'G' argument parsing: digits and ',' are only meaningful inside a G command, so
// other commands (e.g. a heartbeat) may still arrive in the middle of one.
bool parsingG = false;
long gArgs[2];
uint8_t gArg = 0;

uint8_t tuneNote = 0;
unsigned long noteStartMs = 0;
long tuneStartPos = 0;

void startNote() {
  // Alternate direction every note so the net travel stays tiny
  stepper.setSpeed(tuneNote % 2 ? TUNE_HZ[tuneNote] : -TUNE_HZ[tuneNote]);
}

void startTune() {
  if (state != READY || stepper.distanceToGo() != 0 || queueCount > 0) {
    Serial.println(F("Tune rejected - wait until stationary"));
    return;
  }
  tuneStartPos = stepper.currentPosition();
  tuneNote = 0;
  noteStartMs = millis();
  startNote();
  state = TUNE;
  Serial.println(F("Celebrating!"));
}

void endTune() {
  hardStop();
  stepper.moveTo(tuneStartPos);  // undo the small net drift
  state = READY;
  Serial.println(F("Tune done"));
}

void runTune() {
  unsigned long length = TUNE_LEN[tuneNote] * SEMIQUAVER_MS;
  if (millis() - noteStartMs >= length) {
    noteStartMs += length;
    if (++tuneNote == TUNE_NOTES) {
      endTune();
      return;
    }
    startNote();
    length = TUNE_LEN[tuneNote] * SEMIQUAVER_MS;
  }
  // Positive speed is towards the switch: never drive into it
  if (digitalRead(PIN_LIMIT) == HIGH && stepper.speed() > 0) {
    endTune();
    return;
  }
  if (millis() - noteStartMs < length - NOTE_GAP_MS) stepper.runSpeed();
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c == 'G') {
      parsingG = true;
      gArgs[0] = gArgs[1] = 0;
      gArg = 0;
      continue;
    }
    if (parsingG) {
      if (c >= '0' && c <= '9') { gArgs[gArg] = gArgs[gArg] * 10 + (c - '0'); continue; }
      if (c == ',') { gArg = 1; continue; }
      if (c == '\n') {
        parsingG = false;
        serialConnected = true;
        lastSerialMs = millis();
        startSweep(gArgs[0], gArgs[1]);
        continue;
      }
    }
    if (c >= 'a' && c < 'a' + NUM_POSITIONS) {
      serialConnected = true;
      lastSerialMs = millis();
      goToPosition(c - 'a' + 1);
      continue;
    }
    if (c != 'F' && c != 'B' && c != 'S' && c != 'H' && c != 'C' && c != 'T') continue;
    serialConnected = true;
    lastSerialMs = millis();
    if (c == 'C') {
      stopMoves();
      stepper.setSpeed(HOMING_SPEED);
      state = HOMING;
      Serial.println(F("HOMING"));
    }
    else if (c == 'T') startTune();
    else if (c == 'F') queueMove(+1);
    else if (c == 'B') queueMove(-1);
    else if (c == 'S') {
      stopMoves();
      if (state == HOMING) {
        state = FAULT;
        Serial.println(F("EMERGENCY STOP during homing - reset to retry"));
      } else {
        if (state == TUNE) state = READY;  // stop where it is; the position is still known
        Serial.println(F("STOPPED - queue cleared"));
      }
    }
  }
  if (serialConnected && millis() - lastSerialMs > SERIAL_TIMEOUT_MS) {
    stopMoves();
    serialConnected = false;
    if (state == HOMING) state = FAULT;
    Serial.println(F("Serial timeout - stopped and queue cleared"));
  }
}

unsigned long lastPlotMs = 0;

// 8 hex digits by shifting: decimal printing divides per digit (~0.5 ms a line),
// long enough to delay steps every 10 ms and make the motor run rough
void printHex32(unsigned long v) {
  char buf[8];
  for (int8_t i = 7; i >= 0; i--) {
    buf[i] = "0123456789ABCDEF"[v & 15];
    v >>= 4;
  }
  Serial.write(buf, 8);
}

// "<mm>.<2 digits>" with 16-bit maths (up to 655 mm); float printing is slow like decimal
void printHundredths(long h) {
  if (h < 0) {
    Serial.write('-');
    h = -h;
  }
  unsigned int v = h;
  char buf[8];
  int8_t i = 8;
  buf[--i] = '\n';
  buf[--i] = '0' + v % 10;  v /= 10;
  buf[--i] = '0' + v % 10;  v /= 10;
  buf[--i] = '.';
  do { buf[--i] = '0' + v % 10;  v /= 10; } while (v);
  Serial.write(buf + i, 8 - i);
}

// Step if one is due; called again after each slow part of loop()
void runMotor() {
  if (state == HOMING) stepper.runSpeed();
  else if (state == TO_START || state == READY) stepper.run();
}

void setup() {
  Serial.begin(115200);

  // Plain INPUT because the signals are driven by an external 5 V source.
  // Add 10k pull-downs to GND on any pin whose source can float (e.g. a switch).
  pinMode(PIN_FORWARD, INPUT);
  pinMode(PIN_BACKWARD, INPUT);
  pinMode(PIN_LIMIT, INPUT);

  stepper.setPinsInverted(INVERT_DIRECTION, false, false);
  stepper.setMinPulseWidth(2);  // A4988 needs >= 1 us
  stepper.setMaxSpeed(MAX_SPEED);
  stepper.setAcceleration(ACCELERATION);

  // Start the calibration pass: constant slow speed in the FORWARD direction
  stepper.setSpeed(HOMING_SPEED);
  state = HOMING;
  Serial.println(F("HOMING"));
}

void runHoming() {
  bool fwd   = digitalRead(PIN_FORWARD)  == HIGH;
  bool back  = digitalRead(PIN_BACKWARD) == HIGH;
  bool limit = digitalRead(PIN_LIMIT)    == HIGH;

  // Emergency stop: either physical button during calibration halts everything.
  // Arrow presses halt homing immediately in queueMove(), before debounce.
  if (fwd || back) {
    hardStop();
    state = FAULT;
    Serial.println(F("EMERGENCY STOP during homing - reset to retry"));
    return;
  }

  if (limit) {
    hardStop();
    stepper.setCurrentPosition(0);  // this point is 0 mm
    // Backward (away from the switch) is negative, so the start position is -offset
    stepper.moveTo(-START_OFFSET_STEPS);
    state = TO_START;
    Serial.println(F("HOMED - moving to start position"));
    return;
  }

  stepper.runSpeed();  // constant speed, no ramp
}

void runToStart() {
  stepper.run();  // accelerated move off the switch
  if (stepper.distanceToGo() == 0) {
    state = READY;
    // Require every button to be released before the first 7 mm press counts
    buttonsArmed = false;
    buttonChangedMs = millis();
    Serial.println(F("READY at start position"));
  }
}

void runJog() {
  uint8_t buttons = (digitalRead(PIN_FORWARD) == HIGH ? 1 : 0)
                  | (digitalRead(PIN_BACKWARD) == HIGH ? 2 : 0);
  bool limit = digitalRead(PIN_LIMIT)    == HIGH;
  unsigned long now = millis();

  // Never drive into the limit switch or resume queued moves after hitting it.
  if (limit && stepper.distanceToGo() > 0) {
    stopMoves();
    Serial.println(F("Limit switch - stopped and queue cleared"));
  }

  if (buttons != lastButtons) {
    lastButtons = buttons;
    buttonChangedMs = now;
  }
  if (now - buttonChangedMs >= BUTTON_DEBOUNCE_MS) {
    if (buttons == 0) {
      buttonsArmed = true;
    } else if (buttonsArmed) {
      buttonsArmed = false;
      if (buttons == 1 || buttons == 2) {
        queueMove(buttons == 1 ? +1 : -1);
      }
    }
  }

  if (stepper.distanceToGo() == 0 && queueCount > 0) {
    long target = moveQueue[queueHead];
    if (limit && target > stepper.currentPosition()) {
      stopMoves();
      Serial.println(F("Limit switch - stopped and queue cleared"));
    } else {
      queueHead = (queueHead + 1) % MOVE_QUEUE_SIZE;
      queueCount--;
      useJogSpeed();
      stepper.moveTo(target);
    }
  }

  // AccelStepper decelerates at each 7 mm target before starting the next move.
  stepper.run();
}

void loop() {
  readSerialCommands();

  if (state == HOMING)        runHoming();
  else if (state == TO_START) runToStart();
  else if (state == READY)    runJog();
  else if (state == TUNE)     runTune();
  // FAULT: do nothing with the motor until reset

  runMotor();
  if (state == TUNE) runTune();

  // Fast position reports for 'G' moves: time and position sampled together
  unsigned long us = micros();
  if (fastReport && us - lastFastUs >= FAST_REPORT_US) {
    lastFastUs = us;
    Serial.write('P');
    printHex32(us);
    Serial.write(',');
    printHex32(-stepper.currentPosition() * UM_PER_STEP);
    Serial.write('\n');
    if (!stepper.isRunning()) fastReport = false;  // that was the final, stationary report
    runMotor();
  }

  unsigned long now = millis();
  if (now - lastPlotMs >= PLOT_INTERVAL_MS && !fastReport) {   // P lines cover 'G' moves
    lastPlotMs = now;

    // Forward is towards the switch, so moving backward makes position negative.
    // Report distance from the switch as a positive number.
    long hundredths = 0;
    if (state == TO_START || state == READY || state == TUNE) {
      hundredths = -stepper.currentPosition() * 100 / (long)STEPS_PER_MM;
    }

    Serial.print(F("Distance_mm:"));
    printHundredths(hundredths);
    runMotor();
  }
}