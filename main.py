import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2
import sys
import pickle


collectData = False

def main():
    with open('model.pkl', 'rb') as f:
        clf = pickle.load(f)
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
                    keypoints = [hand_landmarks.landmark[0]]
                    '''
                    for data_point in hand_landmarks.landmark:
                        keypoints.append({
                                            'x': data_point.x,
                                            'y': data_point.y,
                                            'z': data_point.z,
                                            'visibility': data_point.visibility,
                                            })'''
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    calced= getRelPositions(hand_landmarks)
                    inDat = calced
                    inference=clf.predict([inDat])
                    image = cv2.putText(image, inference[0], (50,50), cv2.FONT_HERSHEY_COMPLEX, 1, (255,0,0), 2, cv2.LINE_AA)

                cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                with open("data/fist.txt", "w") as f:
                    for i in saved:
                        f.write(i+"\n")               
                break
    cap.release()

label="corner"

def data():
    iter = 0
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    print(list(mp_hands.HandLandmark))

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
            second = image.copy()
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    keypoints = [hand_landmarks.landmark[0]]
                    '''
                    for data_point in hand_landmarks.landmark:
                        keypoints.append({
                                            'x': data_point.x,
                                            'y': data_point.y,
                                            'z': data_point.z,
                                            'visibility': data_point.visibility,
                                            })'''
                    updatedLandmarks= landmark_pb2.NormalizedLandmarkList(landmark=keypoints)
                    mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                    calced= getRelPositions(hand_landmarks)
                    '''
                    for i in range(21):
                        lm = hand_landmarks.landmark[i]
                        org = (int(lm.x * image.shape[0]), int(lm.y * image.shape[1]))
                        #org = (lm.x * image.shape[0], lm.y * image.shape[1])
                        cv2.putText(image, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
                        '''
                    #mp_drawing.draw_landmarks(second, updatedLandmarks)
                    #32 is space bar
                    if(cv2.waitKey(5) & 0xFF == 32):
                        print("Taking snapshot")
                        saved.append(str(calced)[1:-1] + ", " + label)

                cv2.imshow('MediaPipe Hands', image)
                iter+= 1
            if cv2.waitKey(5) & 0xFF == 27:
                with open("data/corner.txt", "w") as f:
                    for i in saved:
                        f.write(i+"\n")               
                break
    cap.release()
            
def getRelPositions(landmarks):
    landmarks = landmarks.landmark
    dists = []
    dists.append(calcDist(1,0, landmarks))
    dists.append(calcDist(2,1, landmarks))
    dists.append(calcDist(3,2, landmarks))
    dists.append(calcDist(4,3, landmarks))
    dists.append(calcDist(5,0, landmarks))
    dists.append(calcDist(6,5, landmarks))
    dists.append(calcDist(7,6, landmarks))
    dists.append(calcDist(8,7, landmarks))
    dists.append(calcDist(9,5, landmarks))
    dists.append(calcDist(10,9, landmarks))
    dists.append(calcDist(11,10, landmarks))
    dists.append(calcDist(12,11, landmarks))
    dists.append(calcDist(13,9, landmarks))
    dists.append(calcDist(14,13, landmarks))
    dists.append(calcDist(15,14, landmarks))
    dists.append(calcDist(16,15, landmarks))
    dists.append(calcDist(17,13, landmarks))
    dists.append(calcDist(18,17, landmarks))
    dists.append(calcDist(19,18, landmarks))
    dists.append(calcDist(20,19, landmarks))
    dists.append(calcDist(17,0, landmarks))
    return [item for sublist in dists for item in sublist]

    
        
def calcDist(first, second, landmarks):
    return (landmarks[first].x-landmarks[second].x, landmarks[first].y-landmarks[second].y, landmarks[first].z-landmarks[second].z)








if __name__ == '__main__':
    if(collectData):
        data()
    else:
        main()