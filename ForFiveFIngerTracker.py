import cv2
import os
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands

# Webcam dimensions
wCam, hCam = 640, 480

# Set up webcam
cap = cv2.VideoCapture(0)
cap.set(3, wCam)
cap.set(4, hCam)

# Load overlay images
folderPath = "FingerImage_Right"
overlayList = []
for imPath in os.listdir(folderPath):
    image = cv2.imread(f'{folderPath}/{imPath}')
    # Resize the overlay image to 40x40 pixels
    image = cv2.resize(image, (150, 150))
    overlayList.append(image)

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

        fingerCount = 0  # Initialize finger count

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Get hand index to check label (left or right)
                handIndex = results.multi_hand_landmarks.index(hand_landmarks)
                handLabel = results.multi_handedness[handIndex].classification[0].label

                # Get landmarks positions (x and y)
                handLandmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]

                # Finger counting logic
                if handLabel == "Left" and handLandmarks[4][0] > handLandmarks[3][0]:
                    fingerCount += 1
                elif handLabel == "Right" and handLandmarks[4][0] < handLandmarks[3][0]:
                    fingerCount += 1

                for i in range(8, 21, 4):
                    if handLandmarks[i][1] < handLandmarks[i - 2][1]:  # Check if finger is raised
                        fingerCount += 1

                # Draw hand landmarks
                mp_drawing.draw_landmarks(
                    image,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS,
                    mp_drawing_styles.get_default_hand_landmarks_style(),
                    mp_drawing_styles.get_default_hand_connections_style())

        # Overlay image based on finger count
        if fingerCount > 0 and fingerCount <= len(overlayList):
            overlay = overlayList[min(fingerCount - 1, len(overlayList) - 1)]
        elif fingerCount == 0:
            # Display the last overlay image when fingerCount is 0
            overlay = overlayList[-1]
        else:
            # If fingerCount is out of range, display the first overlay image
            overlay = overlayList[0]

        h, w, _ = overlay.shape
        image[0:h, 0:w] = overlay

        # Display finger count
        cv2.putText(image, f'No of Finger: {str(fingerCount)}', (50, image.shape[0] - 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (255, 0, 0), 4)

        # Show the image
        cv2.imshow('MediaPipe Hands', image)

        # Break the loop if 'q' is pressed
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# Release the webcam and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()