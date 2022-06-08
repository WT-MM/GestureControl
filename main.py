import mediapipe as mp
import cv2
import collectData
import pickle

def test():
    try:
        with open('model.pkl', 'rb') as f:
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
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    calced= collectData.getRelPositions(hand_landmarks)
                    inDat = calced
                    inference=clf.predict([inDat])
                    image = cv2.putText(image, inference[0], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

                cv2.imshow('MediaPipe Hands', image)
    cap.release()


def main():
    #things
    pass


test = True
if __name__ == '__main__':
    if(test):
        test()
    else:
        main()