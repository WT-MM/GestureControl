import mediapipe as mp
from mediapipe.framework.formats import landmark_pb2
import cv2

def main():
    mp_drawing = mp.solutions.drawing_utils
    mp_hands = mp.solutions.hands
    
    print(list(mp_hands.HandLandmark))

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
                    for i in range(21):
                        lm = hand_landmarks.landmark[i]
                        org = (int(lm.x * image.shape[0]), int(lm.y * image.shape[1]))
                        #org = (lm.x * image.shape[0], lm.y * image.shape[1])
                        cv2.putText(image, str(i), org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,0,0), 2, cv2.LINE_AA)
                    mp_drawing.draw_landmarks(second, updatedLandmarks)
                    cv2.imwrite("Yep.png", image)
                cv2.imshow('MediaPipe Hands', image)
            if cv2.waitKey(5) & 0xFF == 27:
                break
    cap.release()








if __name__ == '__main__':
    main()