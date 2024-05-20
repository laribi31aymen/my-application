
# views.py
from django.shortcuts import render
import cv2
import mediapipe as mp

def hand_detection_view(request):
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()
    
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(image)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                cv2.circle(frame, (int(hand_landmarks.landmark[0].x * image.shape[1]), int(hand_landmarks.landmark[0].y * image.shape[0]), 10, (0, 255, 0), -1))

        ret, jpeg = cv2.imencode('.jpg', frame)
        frame_data = jpeg.tobytes()

        return render(request, 'hand_detection.html', {'frame_data': frame_data})

    cap.release()
    cv2.destroyAllWindows()
