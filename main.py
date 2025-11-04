import cv2
import numpy as np
import pyautogui

# Try to import mediapipe, if not available show error message
MP_AVAILABLE = False
mp = None

try:
    import mediapipe as mp_module
    mp = mp_module
    MP_AVAILABLE = True
except ImportError as e:
    print(f"MediaPipe not available: {e}")
    print("Please install mediapipe using: pip install mediapipe")

def main():
    if not MP_AVAILABLE:
        print("Application cannot run without MediaPipe. Please install the required dependencies.")
        return

    # Initialize Mediapipe Hand module
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(max_num_hands=1)
    mp_draw = mp.solutions.drawing_utils

    # Get screen size
    screen_w, screen_h = pyautogui.size()

    # Start webcam
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        h, w, c = frame.shape
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Process frame with Mediapipe
        result = hands.process(rgb_frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                # Get coordinates of index finger tip (id 8) and thumb tip (id 4)
                landmarks = hand_landmarks.landmark
                index_finger_tip = landmarks[8]
                thumb_tip = landmarks[4]

                # Convert normalized coordinates (0–1) to screen coordinates
                x = int(index_finger_tip.x * w)
                y = int(index_finger_tip.y * h)
                screen_x = int(index_finger_tip.x * screen_w)
                screen_y = int(index_finger_tip.y * screen_h)

                # Move mouse
                pyautogui.moveTo(screen_x, screen_y, duration=0.05)

                # Check distance between thumb and index finger for click
                dist = np.linalg.norm(
                    np.array([thumb_tip.x, thumb_tip.y]) - np.array([index_finger_tip.x, index_finger_tip.y])
                )

                if dist < 0.03:
                    cv2.putText(frame, "Click!", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    pyautogui.click()
                    pyautogui.sleep(0.5)

        cv2.imshow("Hand Controlled Mouse", frame)

        if cv2.waitKey(1) == 27:  # ESC to exit
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()