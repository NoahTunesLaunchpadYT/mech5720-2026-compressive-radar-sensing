#pragma once
#include <cstdint>
#include <sstream>
#include <string>

constexpr int LOW = 0, HIGH = 1, INPUT = 0, A5 = 19;
#define F(text) text
int pins[20] = {};
unsigned long clockMs = 0;
unsigned long millis() { return clockMs; }
int digitalRead(int pin) { return pins[pin]; }
int analogRead(int) { return 512; }
void pinMode(int, int) {}

struct MockSerial {
  std::string input, output;
  void begin(int) {}
  int available() { return static_cast<int>(input.size()); }
  char read() { char c = input.front(); input.erase(0, 1); return c; }
  template<class T> void print(T value, int = 0) {
    std::ostringstream text;
    text << value;
    output += text.str();
  }
  template<class T> void println(T value, int digits = 0) {
    print(value, digits);
    output += '\n';
  }
} Serial;

// Model targets and commands only; tests explicitly finish each simulated move.
class AccelStepper {
 public:
  enum { DRIVER = 1 };
  long position = 0, target = 0;
  int moves = 0, stops = 0;
  float maxSpeed = 0, acceleration = 0;
  AccelStepper(int, int, int) {}
  void setPinsInverted(bool, bool, bool) {}
  void setMinPulseWidth(int) {}
  void setMaxSpeed(float value) { maxSpeed = value; }
  void setAcceleration(float value) { acceleration = value; }
  void setSpeed(float) {}
  long currentPosition() { return position; }
  long distanceToGo() { return target - position; }
  void setCurrentPosition(long value) { position = target = value; }
  void moveTo(long value) { target = value; ++moves; }
  void stop() { ++stops; }
  bool run() { return distanceToGo() != 0; }
  bool runSpeed() { return true; }
  void finish() { position = target; }
};
