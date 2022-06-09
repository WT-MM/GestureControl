import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2
import uuid
from utility import *

labels = []

def main():
    sessionName = str(uuid.uuid4())
    print("Session name is " + sessionName)
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Data Collection')
    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
        saved = []
        while cap.isOpened():
            success, image = cap.read()
            if not success:
                print("Ignoring empty camera frame.")
                # If loading a video, use 'break' instead of 'continue'.
                continue
            
            # Flip the image horizontally for a later selfie-view display, and convert
            # the BGR image to RGB.
            image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
            # To improve performance, optionally mark the image as not writeable to
            # pass by reference.
            image.flags.writeable = False
            results = hands.process(image)
            # Draw the hand annotations on the image.
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    keypoints = [hand_landmarks.landmark[0]]
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    calced= getRelPositions(hand_landmarks)
                    #32 is space bar
                    if(cv2.waitKey(5) & 0xFF == 32):
                        #Convert to string and remove square brackets
                        pass
            if cv2.waitKey(5) & 0xFF == 27:
                with open("data/motion" + sessionName+".txt", "w") as f:
                    for i in saved:
                        f.write(i+"\n")               
                break
            cv2.imshow('Data Collection', image)
    cap.release()
    
if __name__ == "__main__":
    main()