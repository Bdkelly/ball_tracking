import cv2

def find_available_cameras(max_tests=10):
    """
    Tests camera indices starting from 0 up to max_tests-1.
    """
    print("--- Testing Camera Indices (0 to 9) ---")
    available_indices = []
    
    for index in range(max_tests):
        print(f"Attempting to open camera index: {index}...", end="")
        
        # Try to open the video capture stream
        cap = cv2.VideoCapture(index)
        
        # Check if the camera was opened successfully
        if cap.isOpened():
            # Check a second time by trying to read a frame (more robust check)
            ret, frame = cap.read()
            
            if ret:
                available_indices.append(index)
                print("✅ Found.")
            else:
                print("⚠️ Opened but failed to read frame.")
            
            # Release the camera immediately
            cap.release()
        else:
            print("❌ Not found.")
            
    if available_indices:
        print("\n--- Summary ---")
        print(f"Available Camera Indices: {available_indices}")
        print(f"Use index {available_indices[0]} as your self.camera_index = {available_indices[0]}.")
    else:
        print("\n--- Summary ---")
        print("No cameras found. Ensure your webcam is connected and drivers are installed.")
        
if __name__ == "__main__":
    find_available_cameras()