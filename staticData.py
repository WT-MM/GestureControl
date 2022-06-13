import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2
import uuid
import os
from utility import *

labels = ['fist', 'face', 'corner', 'cup', 'circle', 'side', 'forward', 'checkmark']

def main():
    sessionName = str(uuid.uuid4())
    print("Session name is " + sessionName)
    print("Change the slider to change the recorded position")
    print("Use spacebar to save positions")
    print("Use esc to write to file and quit the program")
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Data Collection')
    cv2.createTrackbar('Position', 'Data Collection', 0,len(labels)-1, lambda _ :None)
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
            label = cv2.getTrackbarPos('Position', 'Data Collection')
            image = cv2.putText(image, labels[label], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    keypoints = [hand_landmarks.landmark[0]]
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    calced= getRelPositions(hand_landmarks)
                    #32 is space bar
                    if(cv2.waitKey(5) & 0xFF == 32):
                        print("Taking snapshot")
                        #Convert to string and remove square brackets
                        saved.append(str(calced)[1:-1] + ", " + labels[label])

            if cv2.waitKey(5) & 0xFF == 27:
                with open("data/static/" + sessionName+".txt", "w") as f:
                    for i in saved:
                        f.write(i+"\n")               
                break
            cv2.imshow('Data Collection', image)
    cap.release()
    
if __name__ == "__main__":
    os.makedirs("data/static/", exist_ok=True)
    main()