# This script requires the pybluez library.
# You can install it using: pip install pybluez

import bluetooth

def find_esp32(target_name="ESP32_BT"):
    """Searches for a Bluetooth device with the given name."""
    print("Searching for devices...")
    nearby_devices = bluetooth.discover_devices(duration=8, lookup_names=True,
                                               flush_cache=True, lookup_class=False)
    for addr, name in nearby_devices:
        if target_name == name:
            print(f"Found {name} at {addr}")
            return addr
    print(f"Could not find a device named {target_name}")
    return None

def connect_and_send(device_address):
    """Connects to the ESP32 and sends commands."""
    port = 1  # Standard serial port profile
    sock = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
    try:
        sock.connect((device_address, port))
        print("Connected to ESP32.")
        while True:
            command = input("Enter command (Right, Left, Stop) or 'exit' to quit: ")
            if command.lower() == 'exit':
                break
            sock.send(command + "\n") # Send command with a newline
    except Exception as e:
        print(f"Error: {e}")
    finally:
        print("Closing connection.")
        sock.close()

if __name__ == "__main__":
    esp32_address = find_esp32()
    if esp32_address:
        connect_and_send(esp32_address)
