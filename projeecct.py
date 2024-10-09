from imutils import face_utils
from scipy.spatial import distance as dist
import cv2
import dlib
import os
import pygame  
import time 

cwd = os.path.dirname(__file__)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(cwd + '/shape_predictor_68_face_landmarks.dat')
pygame.mixer.init()

pygame.mixer.music.load(cwd + '/wake.wav') 
rest_recommendation_sound = cwd + '/rest.wav' 

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])
    C = dist.euclidean(eye[0], eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

def mouth_aspect_ratio(mouth):
    A = dist.euclidean(mouth[3], mouth[9])  
    B = dist.euclidean(mouth[2], mouth[10])  
    C = dist.euclidean(mouth[4], mouth[8])  
    D = dist.euclidean(mouth[0], mouth[6])  
    mar = (A + B + C) / (3.0 * D)
    return mar

EYE_AR_THRESH = 0.25
MOUTH_AR_THRESH = 0.7
EYE_AR_CONSEC_FRAMES = 15
CLOSED_EYE_DURATION_THRESH = 3
blink_counter = 0
eye_closure_instances = 0  
yawn_counter = 0  
alarm_on = False
start_time = 0
MAX_EYE_CLOSURES = 5  
MAX_YAWNS = 3

def play_alarm():
    global alarm_on
    if not alarm_on:  
        pygame.mixer.music.play(-1) 
        pygame.time.delay(3000)  
        alarm_on = True

def stop_alarm():
    global alarm_on
    pygame.mixer.music.stop()
    alarm_on = False

def suggest_rest():
    pygame.mixer.music.load(rest_recommendation_sound)
    pygame.mixer.music.play()
    


cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    rects = detector(gray, 0)
    
    for rect in rects:
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        leftEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["left_eye"][1]]
        rightEye = shape[face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][0]:face_utils.FACIAL_LANDMARKS_IDXS["right_eye"][1]]
        
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)
        ear = (leftEAR + rightEAR) / 2.0
        
        mouth = shape[face_utils.FACIAL_LANDMARKS_IDXS["mouth"][0]:face_utils.FACIAL_LANDMARKS_IDXS["mouth"][1]]
        
        mar = mouth_aspect_ratio(mouth)
        
        if ear < EYE_AR_THRESH:
            if blink_counter == 0:  
                start_time = time.time()
            blink_counter += 1

            if time.time() - start_time >= CLOSED_EYE_DURATION_THRESH:
                play_alarm()
                cv2.putText(frame, "Eyes Closed!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        else:
            if blink_counter > 0:  
                eye_closure_instances += 1
            blink_counter = 0
            stop_alarm()

        if mar > MOUTH_AR_THRESH:
            cv2.putText(frame, "Yawning", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            yawn_counter += 1

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        mouthHull = cv2.convexHull(mouth)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [mouthHull], -1, (255, 0, 0), 1)

        if eye_closure_instances >= MAX_EYE_CLOSURES or yawn_counter >= MAX_YAWNS:
            suggest_rest()

    cv2.imshow("Frame", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


