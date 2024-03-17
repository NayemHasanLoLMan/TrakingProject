import cv2
import os
import mediapipe as mp
import numpy as np

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Webcam dimensions
wCam, hCam = 640, 480

# Set up webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Load overlay images for the right hand
folderPathRight = "FingerImage_Right"
overlayListRight = []
for imPath in os.listdir(folderPathRight):
    image = cv2.imread(f'{folderPathRight}/{imPath}')
    # Resize the overlay image to 150x150 pixels
    image = cv2.resize(image, (150, 150))
    overlayListRight.append(image)

# Load overlay images for the left hand
folderPathLeft = "FingerImage_Left"
overlayListLeft = []
for imPath in os.listdir(folderPathLeft):
    image = cv2.imread(f'{folderPathLeft}/{imPath}')
    # Resize the overlay image to 150x150 pixels
    image = cv2.resize(image, (150, 150))
    overlayListLeft.append(image)

# Initialize MediaPipe Hands model
with mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue

        # Flip the image horizontally for a selfie-view
        image = cv2.flip(image, 1)

        # Convert the image to RGB and process it with MediaPipe Hands
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = hands.process(image_rgb)

        fingerCountRight = 0  # Initialize finger count for the right hand
        fingerCountLeft = 0   # Initialize finger count for the left hand

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand label (left or right)
                handLabel = results.multi_handedness[results.multi_hand_landmarks.index(hand_landmarks)].classification[0].label

                # Get landmarks positions (x and y)
                handLandmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                # Finger counting logic for the right hand
                if handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                    fingerCountRight += 1
                for i in range(8, 21, 4):
                    if handLabel == "Right" and handLandmarks[i][1] < handLandmarks[i - 2][1]:  # Check if finger is raised for right hand
                        fingerCountRight += 1

                # Finger counting logic for the left hand
                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                    fingerCountLeft += 1
                for i in range(8, 21, 4):
                    if handLabel == "Left" and handLandmarks[i][1] < handLandmarks[i - 2][1]:  # Check if finger is raised for left hand
                        fingerCountLeft += 1

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Overlay image for the right hand
        if fingerCountRight > 0 and fingerCountRight <= len(overlayListRight):
            overlay_right = overlayListRight[min(fingerCountRight - 1, len(overlayListRight) - 1)]
        elif fingerCountRight == 0:
            # Display the last overlay image when fingerCount is 0
            overlay_right = overlayListRight[-1]
        else:
            # If fingerCount is out of range, display the first overlay image
            overlay_right = overlayListRight[0]

        # Overlay image for the left hand
        if fingerCountLeft > 0 and fingerCountLeft <= len(overlayListLeft):
            overlay_left = overlayListLeft[min(fingerCountLeft - 1, len(overlayListLeft) - 1)]
        elif fingerCountLeft == 0:
            # Display the last overlay image when fingerCount is 0
            overlay_left = overlayListLeft[-1]
        else:
            # If fingerCount is out of range, display the first overlay image
            overlay_left = overlayListLeft[0]

        # Place the overlay images in the corners
        h, w, _ = overlay_left.shape
        image[0:h, 0:w] = overlay_left

        h, w, _ = overlay_right.shape
        image[0:h, image.shape[1] - w:] = overlay_right

        # Display total finger count
        total_finger_count = fingerCountLeft + fingerCountRight
        cv2.putText(image, f'Total Finger Count: {total_finger_count}', (50, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 4)

        # Show the image
        cv2.imshow('MediaPipe Hands', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()

