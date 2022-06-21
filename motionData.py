import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2
import uuid
from utility import *
import time


'''
Largest problem: inaccurate time logging... is it an important issue? 
Test to see if this has a significant impact

Capture points continuously

Two methods:

1.
Break down into per millisecond segments
Perform relative distances from starting frame
Take set timeframe and feed into svm

2.
Somehow feed continuously???????

3.
Get FPS
Adjust length of array based on FPS -> standard should be 30? (important for formatting data into svm)
Continuously append and pop into array
Perform relative distances from starting frame
Save as chunk and use to train or classify

Data format:
Will do 1 second chunks, = 30 typically.
If fps above 30, take number of extra frames and remove randomly
If fps below 30, take number of missing frames and duplicate positions
Can also interpolate if needed

Practically, should set length of array at beginning based on fps
Then, it should be processed before saving/feeding

Issue: differentiating gestures from non gestures

'''


def main():
    labels = ["swiperight", "swipeleft", "swipeup", "swipedown", "closefist", "openfist", "scrubindex"]
    sessionName = str(uuid.uuid4())
    timeTolerance = 0.01 #0.01 seconds
    
    print("Session name is " + sessionName)
    print("Press the spacebar to start recording")
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    cap = cv2.VideoCapture(0)
    cv2.namedWindow('Data Collection')
    cv2.createTrackbar('Movement', 'Data Collection', 0,len(labels)-1, lambda _ :None)
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    counter = fps
    prep = 0
    recording = False
    dataArr = ["" for i in range(fps)]
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
            label = cv2.getTrackbarPos('Movement', 'Data Collection')
            image = cv2.putText(image, labels[label], (0,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    keypoints = [hand_landmarks.landmark[0]]
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    dataArr.append(hand_landmarks.landmark)
                    dataArr.pop(0)
                    k = cv2.waitKey(12)
                    if(k == ord(' ')):
                        prepTime = time.time()
                        recording = True

            if(recording and time.time() - prepTime < 5):
                image = cv2.putText(image, str(5-round(time.time()-prepTime)), (int(width/2),int(height/2)), cv2.FONT_HERSHEY_COMPLEX, 5, (255,0,0), 2, cv2.LINE_AA)
                counter = -1
            elif(counter < fps):
                image = cv2.ellipse(image, (width-50,50), (50,50), 0, 0, 360, (0,0,255), -1)
            elif(recording and counter == fps):
                fixed = processArray(fps, dataArr)
                data = getInitDiff(fixed)
                saved.append(str(data)[1:-1] + ", " + labels[label])
            if cv2.waitKey(5) & 0xFF == 27:
                with open("data/motion/" + sessionName+".txt", "w") as f:
                    for i in saved:
                        f.write(i+"\n")
                break
            counter+=1
            prep-=1
            cv2.imshow('Data Collection', image)
    cap.release()
    
if __name__ == "__main__":
    os.makedirs("data/motion/", exist_ok=True)
    main()