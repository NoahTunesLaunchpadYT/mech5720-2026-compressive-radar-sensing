/*
  Stepper step control (A4988 + NEMA11) with homing, distance reporting
  and a voltmeter on A5 for the Serial Plotter

  On power-up / reset:
    The motor travels in the pin 13 (FORWARD) direction at HOMING_SPEED
    until the limit switch on pin 9 goes HIGH. That point becomes 0 mm.
    During this calibration pass, pin 12 or pin 13 going HIGH is an
    EMERGENCY STOP: the motor halts immediately and stays halted until
    the Arduino is reset.

  After homing:
    Pin 13 -> FORWARD  (press = 7 mm towards the limit switch)
    Pin 12 -> BACKWARD (press = 7 mm away from the switch)
    Buttons debounce for 30 ms. Release does not cancel; holding never repeats.
    Individual presses queue 7 mm moves in press order, including during motion.
    Both HIGH starts nothing; release both buttons before trying again.
    Moves that cannot fit entirely within 0..MAX_TRAVEL_MM after queued moves
    are rejected. Up to 32 moves may wait in the queue.
    The limit switch still blocks forward motion if it is hit again.
    Backward travel stops at MAX_TRAVEL_MM from the switch (soft limit).

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

  Serial output at 115200 baud (Serial Plotter or main.py) shows:
    Voltage      - voltage on A5 through the divider
    Distance_mm  - distance from the limit switch (0 until homed)

  Requires the AccelStepper library (Tools > Manage Libraries > "AccelStepper").
*/

#include <AccelStepper.h>

// ---- Pins ----
const uint8_t PIN_FORWARD  = 13;
const uint8_t PIN_BACKWARD = 12;
const uint8_t PIN_STEP     = 11;
const uint8_t PIN_DIR      = 10;
const uint8_t PIN_LIMIT    = 9;   // HIGH when the limit switch is pressed
const uint8_t PIN_SENSE    = A5;

// ---- Actuator: T6x1 lead screw, 1.8 deg motor, 200 mm stroke ----
// Steps per mm = (200 steps/rev x microsteps) / 1 mm per rev
const float STEPS_PER_REV = 200.0;  // 1.8 degree step angle
const float MICROSTEPS    = 1.0;    // 1, 2, 4, 8 or 16 to match MS1-MS3 on the A4988
const float MM_PER_REV    = 1.0;    // T6x1 screw: 1 mm per revolution
const float STEPS_PER_MM  = STEPS_PER_REV * MICROSTEPS / MM_PER_REV;  // 200 at full step

// Soft limit: how far from the switch the slider may travel
const float MAX_TRAVEL_MM = 195.0;  // 200 mm stroke minus a 5 mm safety margin
const long  MAX_TRAVEL_STEPS = (long)(MAX_TRAVEL_MM * STEPS_PER_MM);
const long  BUTTON_MOVE_STEPS = (long)(7.0 * STEPS_PER_MM + 0.5);
const unsigned long BUTTON_DEBOUNCE_MS = 30;

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

// ---- Voltmeter settings ----
// VREF: measure the Arduino's 5V pin with a multimeter and put the real value here.
const float VREF = 5.0;
// Divider: 10k on the input side, 1k on the ground side. Measures up to about 55 V.
const float R1 = 0.0;  // ohms, input -> A5
const float R2 = 1000.0;   // ohms, A5 -> GND
const unsigned long PLOT_INTERVAL_MS = 50;  // ~20 points per second

AccelStepper stepper(AccelStepper::DRIVER, PIN_STEP, PIN_DIR);

enum State { HOMING, READY, FAULT };
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
  if (state != READY) return;
  if (queueCount == MOVE_QUEUE_SIZE) {
    Serial.println(F("Move rejected - queue full"));
    return;
  }
  long origin = queueCount > 0
              ? moveQueue[(queueHead + queueCount - 1) % MOVE_QUEUE_SIZE]
              : stepper.currentPosition() + stepper.distanceToGo();
  long target = origin + direction * BUTTON_MOVE_STEPS;
  if (target > 0 || target < -MAX_TRAVEL_STEPS) {
    Serial.println(F("Move rejected - travel limit"));
    return;
  }
  moveQueue[(queueHead + queueCount) % MOVE_QUEUE_SIZE] = target;
  queueCount++;
}

void readSerialCommands() {
  while (Serial.available() > 0) {
    char c = Serial.read();
    if (c != 'F' && c != 'B' && c != 'S' && c != 'H') continue;
    serialConnected = true;
    lastSerialMs = millis();
    if (c == 'F') queueMove(+1);
    else if (c == 'B') queueMove(-1);
    else if (c == 'S') {
      stopMoves();
      if (state == HOMING) {
        state = FAULT;
        Serial.println(F("EMERGENCY STOP during homing - reset to retry"));
      } else {
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

// Voltmeter accumulators: one sample per loop() so the stepper is never starved
long          adcTotal   = 0;
unsigned int  adcCount   = 0;
unsigned long lastPlotMs = 0;

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
    state = READY;
    Serial.println(F("HOMED"));
    return;
  }

  stepper.runSpeed();  // constant speed, no ramp
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
      stepper.moveTo(target);
    }
  }

  // AccelStepper decelerates at each 7 mm target before starting the next move.
  stepper.run();
}

void loop() {
  readSerialCommands();

  if (state == HOMING)     runHoming();
  else if (state == READY) runJog();
  // FAULT: do nothing with the motor until reset

  // ---- Voltmeter + distance ----
  adcTotal += analogRead(PIN_SENSE);
  adcCount++;

  if (state == HOMING)     stepper.runSpeed();
  else if (state == READY) stepper.run();

  unsigned long now = millis();
  if (now - lastPlotMs >= PLOT_INTERVAL_MS && adcCount > 0) {
    lastPlotMs = now;

    float raw        = adcTotal / (float)adcCount;
    float pinVolts   = raw * VREF / 1023.0;
    float inputVolts = pinVolts * (R1 + R2) / R2;
    adcTotal = 0;
    adcCount = 0;

    // Forward is towards the switch, so moving backward makes position negative.
    // Report distance from the switch as a positive number.
    float distanceMm = 0.0;
    if (state == READY) {
      distanceMm = -stepper.currentPosition() / STEPS_PER_MM;
    }

    // "Label:value" pairs become named traces in the Serial Plotter
    Serial.print(F("Voltage:"));
    Serial.print(inputVolts, 2);
    Serial.print(F(",Distance_mm:"));
    Serial.println(distanceMm, 2);
  }
}
