import cv2
# Use the standard import for MediaPipe
import mediapipe as mp

# Access the modules via the mp alias
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

def run_hand_tracking_on_webcam():
    # Initialize the camera object. index=0 usually refers to the built-in webcam.
    cam = cv2.VideoCapture(index=0)

    # Check if the camera was opened successfully
    if not cam.isOpened():
        print("Error: Could not open webcam.")
        return

    # Initialize the MediaPipe Hands object
    with mp_hands.Hands(
        model_complexity=0,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5,
    ) as hands:
        while True: # Loop infinitely until the 'q' key is pressed
            success, frame = cam.read()
            if not success:
                print("Empty frame! Skipping.")
                continue

            # 1. Convert the BGR image from OpenCV to RGB for MediaPipe processing.
            frame.flags.writeable = False 
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = hands.process(frame_rgb)

            # 2. Convert the image back to BGR for drawing and displaying.
            frame.flags.writeable = True
            frame = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)

            # 3. Draw the hand landmarks if found.
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=hand_landmarks,
                        connections=mp_hands.HAND_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_hand_landmarks_style(),
                        connection_drawing_spec=mp_drawing_styles.get_default_hand_connections_style(),
                    )
            # 

            # 4. Display the resulting image. cv2.flip(frame, 1) creates a mirror image (selfie view).
            cv2.imshow("Hand Tracking (Press 'q' to Quit)", cv2.flip(frame, 1))

            # 5. Break the loop if the 'q' key is pressed.
            # cv2.waitKey(5) ensures the window updates every 5 milliseconds
            if cv2.waitKey(5) & 0xFF == ord("q"):
                break

    # Release the webcam and close all OpenCV windows
    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    run_hand_tracking_on_webcam()