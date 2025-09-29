#include <AccelStepper.h>
#include <HardwareSerial.h> // Needed for string reading on ESP32

const int dirPin = 12;
const int stepPin = 14;

#define motorInterfaceType 1

AccelStepper myStepper(motorInterfaceType, stepPin, dirPin);

// Variables for step-to-degree conversion
const int STEPS_PER_REV = 200; // Typical for a standard stepper motor
const int MICROSTEPS = 16;     // Check your driver's setting (e.g., 1, 2, 4, 8, 16)
const float GEAR_RATIO = 1.0;  // Change this if you have a gearbox, e.g., 10 for a 10:1 ratio

// Calculate total steps per full revolution of the camera
const float TOTAL_STEPS_PER_REV = STEPS_PER_REV * MICROSTEPS * GEAR_RATIO;
// Calculate steps per degree
const float STEPS_PER_DEGREE = TOTAL_STEPS_PER_REV / 360.0;

void setup() {
    Serial.begin(115200);
    myStepper.setMaxSpeed(400);
    myStepper.setAcceleration(300);
    Serial.println("ESP32 ready. Waiting for commands from computer.");
}

void loop() {
    if (Serial.available() > 0) {
        String command = Serial.readStringUntil('\n'); // Read until a newline character
        command.trim(); // Remove any leading/trailing whitespace
       
        // Define the number of steps to move per command
        const long stepsToMove = static_cast<long>(1.0 * STEPS_PER_DEGREE); // 1 degree of movement
       
        if (command.equals("Right")) {
            myStepper.move(-stepsToMove); // Negative moves it to the right
            Serial.println("Moving Right");
        } else if (command.equals("Left")) {
            myStepper.move(stepsToMove); // Positive moves it to the left
            Serial.println("Moving Left");
        } else if (command.equals("Stop")) {
            myStepper.stop();
            Serial.println("Stopping");
        }
    }
    myStepper.run();
}