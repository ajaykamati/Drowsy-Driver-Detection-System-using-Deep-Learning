# Use following command to download the necessary modules
#pip3 install mediapipe numpy tensorflow opencv-contrib-python

import mediapipe as mp #pip3 install mediapipe
import time
import math
import numpy as np #pip3 install numpy
from playsound import playsound
import pygame
pygame.init()
# import systemcheck

import cv2 #pip3 install opencv-contrib-python


# variables 
close_eye_count =0
open_mouth_count = 0
Blink_counts =0

# constants
Close_frames = 10 # try to increase the value

FONTS = cv2.FONT_HERSHEY_SIMPLEX
TEXT_COLOR = (0,250, 0) # BGR 0 - 255
TEXT_COLOR_ALARM = (0,0,250) # BGR 0 - 255
alarm_sound_path = "alarm.mp3"


face_outline=[ 10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103,67, 109]

LIPS=[ 61, 146, 91, 181, 84, 17, 314, 405, 321, 375,291, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95,185, 40, 39, 37,0 ,267 ,269 ,270 ,409, 415, 310, 311, 312, 13, 82, 81, 42, 183, 78 ]

LEFT_EYE =[ 362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385,384, 398 ]

RIGHT_EYE=[ 33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161 , 246 ]  

LEFT_EYEBROW = [336, 296, 334, 293, 300, 276, 283, 282, 295, 285]

RIGHT_EYEBROW = [70, 63, 105, 66, 107, 55, 65, 52, 53, 46]

NOSE = [8,240,460]

face_detection = mp.solutions.face_detection.FaceDetection(model_selection=0, min_detection_confidence=0.5)#conf=True)
face_mesh = mp.solutions.face_mesh.FaceMesh(min_detection_confidence =0.5, min_tracking_confidence=0.5)



def detect_eye_mouth_status(face_img, raw_img):
    global close_eye_count, Blink_counts, open_mouth_count

    rgb_frame = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB) #Convert image to RGB
    results  = face_mesh.process(rgb_frame)# Get landmarks of the Face
    
    if results.multi_face_landmarks:
        img_h, img_w= face_img.shape[:2]
        mesh_coords = [(int(p.x * img_w), int(p.y * img_h)) for p in results.multi_face_landmarks[0].landmark]
        reRatio, leRatio, mRatio = Open_Close_Ratios(mesh_coords) # Get Eyes and Mouth Ratio
        ratio = round((reRatio+ leRatio)/2, 2) # Average Eye Ratio of both

        eye_threshold = 3.5
        mouth_threshold = 1.8
        
        if mRatio > mouth_threshold: 
            cv2.putText(raw_img, 'Mouth Closed', (10, 30), FONTS, 1, TEXT_COLOR, 1)
        else:
            cv2.putText(raw_img, 'Mouth Open', (10, 30), FONTS, 1, TEXT_COLOR, 1)
            open_mouth_count += 1
            if open_mouth_count>Close_frames:
                    cv2.putText(raw_img, 'Yawning, do not sleep', (10, 150), FONTS, 1, TEXT_COLOR_ALARM, 3)
                    print("*************Yawning!!!! Don't sleep*************")
                    # playsound(alarm_sound_path)
                    pygame.mixer.music.load(alarm_sound_path)
                    pygame.mixer.music.play()
                    open_mouth_count =0


        if ratio >= eye_threshold:
            cv2.putText(raw_img, 'Eyes Closed', (10, 70), FONTS, 1, TEXT_COLOR, 1)
            close_eye_count += 1
            if close_eye_count>Close_frames:
                    cv2.putText(raw_img, 'Do not sleep', (10, 180), FONTS, 1, TEXT_COLOR_ALARM, 3)
                    print("*************Don't sleep*************")
                    # playsound(alarm_sound_path)
                    pygame.mixer.music.load(alarm_sound_path)
                    pygame.mixer.music.play()
                    Blink_counts +=1
                    close_eye_count =0
        else:
            cv2.putText(raw_img, 'Eyes Open', (10, 70), FONTS, 1, TEXT_COLOR, 1)

        cv2.putText(raw_img, f'Total Blinks: {Blink_counts}', (10, 110), FONTS, 1, TEXT_COLOR, 1)
    

        #Draw curved shapes for Eye and Mouth and Face Outline
        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in LEFT_EYE ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)

        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in RIGHT_EYE ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)

        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in LIPS ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)

        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in face_outline ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)

        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in LEFT_EYEBROW ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)

        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in RIGHT_EYEBROW ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)

        cv2.polylines(face_img,  [np.array([mesh_coords[p] for p in NOSE ], dtype=np.int32)], True, (0,255,0), 1, cv2.LINE_AA)
    
    return face_img, raw_img



# Blinking Ratio
def Open_Close_Ratios(landmarks):
    # Right eyes 
    rh_right = landmarks[246]  # Eye Points from Mediapipe Landmark Points
    rh_left = landmarks[133] 
    rv_top = landmarks[160]
    rv_bottom = landmarks[145]


    # LEFT_EYE 
    lh_right = landmarks[362]
    lh_left = landmarks[387]
    lv_top = landmarks[386]
    lv_bottom = landmarks[374]


    rhDistance = math.dist(rh_right, rh_left)
    rvDistance = math.dist(rv_top, rv_bottom)

    lvDistance = math.dist(lv_top, lv_bottom)
    lhDistance = math.dist(lh_right, lh_left)

    try:
        reRatio = rhDistance/rvDistance
        leRatio = lhDistance/lvDistance
    except:
        reRatio = 0
        leRatio = 0


    # Mouth Ratio
    mouth_right = landmarks[409] # Mouth Points from Mediapipe Landmark Points
    mouth_left = landmarks[185]
    mouth_top = landmarks[0]
    mouth_bottom = landmarks[17]

    mhDistance = math.dist(mouth_right, mouth_left)
    mvDistance = math.dist(mouth_top, mouth_bottom)

    try:
        mRatio = mhDistance/mvDistance
    except:
        mRatio = 10

    return reRatio,  leRatio, mRatio

##########################################

cam = cv2.VideoCapture(0) #Get access to Camera

while True:
    _, raw_img = cam.read() # Read Image from Camera
    if _:
        start_time = time.time() #Start time, helpful in estimating FPS

        ########## FACE DETECTION ##########
        x,y,w,h = 0,0,0,0
        img_w = raw_img.shape[1]
        img_h = raw_img.shape[0]

        face_detection_results = face_detection.process(raw_img[:,:,::-1])# Detect Faces in Images

        if face_detection_results.detections:
            for face in face_detection_results.detections:

                # print(f'FACE CONFIDENCE: {round(face.score[0], 2)}')
                if face.score[0] < 0.8: # Check confidence of next Face if confidence of current face is less
                    continue

                face_data = face.location_data

                x,y,w,h = int(img_w*face_data.relative_bounding_box.xmin), \
                            int(img_h*face_data.relative_bounding_box.ymin), \
                            int(img_w*face_data.relative_bounding_box.width), \
                            int(img_h*face_data.relative_bounding_box.height)
                break #Break if found Face as we are using only 1 Face


      
        if x+y+w+h > 0:
            # print("Detected Face Points:", x,y,w,h)
            x = x - 10 # Extend X axis to cover whole Face
            y = y - 40 # Extend Y axis to cover forehead
            w = w + 20 # Extend width as we moved X axis bit left
            h = h + 40 # Extend height as we moved Y axis bit up

            if x<0:
                x = 0
            if y<0:
                y = 0

            face_img = raw_img[y:y+h, x:x+w]
            cv2.rectangle(raw_img, (x,y), (x+w, y+h), (255,0,0), 2) # Draw Rectangle around Face

            ########## EYE and MOUTH STATUS DETECTION ##########
            face_processed, raw_img = detect_eye_mouth_status(face_img, raw_img) 
            raw_img[y:y+h, x:x+w] = face_processed #Replace Processed Face Area

        cv2.imshow('frame', raw_img)
        cv2.waitKey(1)

        # calculating  frame per seconds FPS
        time_taken = time.time()-start_time
        fps = 1/time_taken

