import time
import albumentations as A
from albumentations.pytorch import ToTensorV2
import os
import numpy as np
import cv2  # Import OpenCV
import torch
import serial
import serial.tools.list_ports
from models import get_fasterrcnn_model_single_class as fmodel

# Define class names globally or pass them. Must match training order.
GLOBAL_CLASS_NAMES = ['__background__', 'Ball']

# Agent's actions helper functions
# Now sends a command to the ESP32
def move_left(ser):
    """Sends command to ESP32 to move camera left."""
    try:
        ser.write(b"Left") # Changed to a different command for clarity
        print("Sent command: Left")
    except serial.SerialException as e:
        print(f"Serial communication error: {e}")

def move_right(ser):
    """Sends command to ESP32 to move camera right."""
    try:
        ser.write(b"Right") # Changed to a different command for clarity
        print("Sent command: Right")
    except serial.SerialException as e:
        print(f"Serial communication error: {e}")

# New function to handle ball detection and drawing
def get_ball_detection(model, frame, transform, device, confidence_threshold=0.98):
    """
    Performs ball detection on a single frame and draws the bounding box.
    Returns the bounding box coordinates and the frame with drawn boxes.
    """
    # Convert BGR frame to RGB and apply transformations
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    transformed = transform(image=frame_rgb)
    image_tensor = transformed['image'].to(device).unsqueeze(0)

    # Perform inference
    with torch.no_grad():
        predictions = model(image_tensor)

    # Process predictions
    boxes_raw = predictions[0]['boxes']
    labels_raw = predictions[0]['labels']
    scores_raw = predictions[0]['scores']

    keep_indices = torch.where(scores_raw >= confidence_threshold)[0]
    filtered_boxes = boxes_raw[keep_indices].cpu().numpy()
    filtered_labels = labels_raw[keep_indices].cpu().numpy()
    filtered_scores = scores_raw[keep_indices].cpu().numpy()

    # Get original image dimensions from the frame
    original_size = (frame.shape[1], frame.shape[0])
    x_scale = original_size[0] / 640
    y_scale = original_size[1] / 640

    detected_boxes = []
    for i in range(len(filtered_boxes)):
        box = filtered_boxes[i]
        label_idx = filtered_labels[i]
        score = filtered_scores[i]

        x_min_orig = int(box[0] * x_scale)
        y_min_orig = int(box[1] * y_scale)
        x_max_orig = int(box[2] * x_scale)
        y_max_orig = int(box[3] * y_scale)
        
        detected_boxes.append({
            'box': (x_min_orig, y_min_orig, x_max_orig, y_max_orig),
            'label': GLOBAL_CLASS_NAMES[label_idx],
            'score': score
        })

        # Draw the bounding box on the original frame
        cv2.rectangle(frame, (x_min_orig, y_min_orig), (x_max_orig, y_max_orig), (0, 0, 255), 2)
        text = f"{GLOBAL_CLASS_NAMES[label_idx]} {score:.2f}"
        cv2.putText(frame, text, (x_min_orig, y_min_orig - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
        break

    return detected_boxes, frame
def find_esp32():
    ports = list(serial.tools.list_ports.comports())
    for p in ports:
        if "USB" in p.description.upper() or "SERIAL" in p.description.upper() or "ESP" in p.description.upper():
            print(f"Found ESP32: {p.device}")
            return p.device
    print("No ESP32 found. Ensure it's connected and drivers are installed.")
    return None

# New main function that replaces the original run_inference_on_video
def main():
    # Model and device setup
    model_path = 'fin_comb.pth'
    num_classes = len(GLOBAL_CLASS_NAMES)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = fmodel(num_classes).to(device)
    if not os.path.exists(model_path):
        print(f"Error: Model file not found at {model_path}")
        return
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    # Transformations for inference
    transform = A.Compose([
        A.Resize(640, 640),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2()
    ])
    
    ser = None
    esp32_port = find_esp32()
    if esp32_port:
        try:
            ser = serial.Serial(esp32_port, 115200, timeout=1)
            print("Successfully connected to ESP32.")
        except serial.SerialException as e:
            print(f"Failed to connect to ESP32: {e}")
            ser = None
    
    # Video stream setup
    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error: Could not open video stream.")
        return

    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Add a variable to track the last command time
    last_command_time = time.time()
    command_interval = 1.0 # 1 second
    
    # Main loop for video processing
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Get ball detection
        detected_boxes, frame_with_detections = get_ball_detection(model, frame.copy(), transform, device)

        # Simple logic for camera movement
        if detected_boxes and ser: # Only move if a ball is detected and serial is connected
            # Check if one second has passed since the last command
            current_time = time.time()
            if current_time - last_command_time >= command_interval:
                # Get the ball's center X-coordinate
                ball_box = detected_boxes[0]['box']
                ball_center_x = (ball_box[0] + ball_box[2]) / 2

                # Get the screen's center X-coordinate
                screen_center_x = W / 2
                
                # Move based on ball's position relative to the screen center
                if ball_center_x < screen_center_x - 50: # Adding a deadzone
                    move_left(ser)
                elif ball_center_x > screen_center_x + 50: # Adding a deadzone
                    move_right(ser)
                else:
                    # If the ball is exactly in the center, stay put.
                    ser.write(b"Stop")
                    pass
                
                # Update the last command time
                last_command_time = current_time

        # Display the result
        cv2.imshow('Video Inference and Agent View', frame_with_detections)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    if ser:
        ser.close()

if __name__ == "__main__":
    main()