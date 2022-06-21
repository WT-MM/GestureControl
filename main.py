import mediapipe as mp
import cv2
import pickle
from utility import *

def test():
    try:
        with open('models/static.pkl', 'rb') as f:
                clf = pickle.load(f)
    except FileNotFoundError:
        print("Missing SVM pkl file")
        exit()
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
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
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    calced= getRelPositions(hand_landmarks)
                    inDat = calced
                    inference=clf.predict([inDat])
                    image = cv2.putText(image, inference[0], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            if cv2.waitKey(5) & 0xFF == 27:            
                break
            cv2.imshow('MediaPipe Hands', image)
    cap.release()


def main():
    try:
        with open('models/motion.pkl', 'rb') as f:
            motionCLF = pickle.load(f)
        with open('models/static.pkl', 'rb') as f:
            staticCLF = pickle.load(f)
    except FileNotFoundError:
        print("Missing SVM pkl file(s)")
        exit()
    mp_hands = mp.solutions.hands
    
    cap = cv2.VideoCapture(0)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    motionData = ["" for i in range(fps)]
    with mp_hands.Hands(
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5) as hands:
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
            currentInf = []
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    calced= getRelPositions(hand_landmarks)
                    inDat = calced
                    staticInf=staticCLF.predict([inDat])
                    staticProba = staticCLF.predict_proba([inDat])
                    
                    motionData.append(hand_landmarks.landmark)
                    motionData.pop(0)
                    motIn = getInitDiff(processArray(motionData))
                    motionInf = motionCLF.predict([motIn])
                    motionProba = motionCLF.predict_proba([motIn])
                    #Testing purposes
                    print("Static Inference: " + staticInf)
                    print("Motion Inference: " + motionInf) 
                    print("Static probability: " + staticProba)
                    print("Motion probability" + motionProba)
                    image.flags.writeable = True
                    image = cv2.putText(image, "Static Inference: " + staticInf[0], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
                    image = cv2.putText(image, "Motion Inference: " + motionInf[0], (50,100), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            cv2.imshow("mmmmm", image) #Delete when done testing

    cap.release()


tB = True
if __name__ == '__main__':
    if(tB):
        test()
    else:
        main()