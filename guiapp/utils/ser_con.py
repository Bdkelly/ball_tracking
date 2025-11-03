import serial
import serial.tools.list_ports
import time

_command_signal_ref = None # Internal reference to hold the signal

def set_command_signal(signal):
    """Sets the global signal reference used for logging."""
    global _command_signal_ref
    _command_signal_ref = signal

def find_esp32():
    # ... (function remains the same)
    return "/dev/cu.usbserial-10"

def move_left(ser):
    if ser:
        if _command_signal_ref: _command_signal_ref.emit("Left") # Use internal reference
        ser.write(b"Left\n") 
        print("Sent command: Left")
    else:
        print("Serial not connected: Left")

def move_right(ser):
    if ser:
        if _command_signal_ref: _command_signal_ref.emit("Right") # Use internal reference
        ser.write(b"Right\n") 
        print("Sent command: Right")
    else:
        print("Serial not connected: Right")