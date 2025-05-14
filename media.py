import cv2
import mediapipe as mp 

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)
selection_body=[]
selection_hand_left = []
selection_hand_right = []
selection_legright = []
selection_legleft = []
selection_nose = []
selection_earleft = []
selection_earright = []

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break    
    selection_body.clear()
    selection_hand_left.clear()
    selection_hand_right.clear()
    selection_legright.clear()
    selection_legleft.clear()
    selection_nose.clear()
    selection_earleft.clear()
    selection_earright.clear()
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)   
    results = pose.process(image_rgb)
   
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            frame,results.pose_landmarks,
            [(11,12),(12,24),(24,23),(23,11),
             (11,13),(13,15),
             (12,14),(14,16),
             (24,26),(26,28),
             (23,25),(25,27),
             (0,8),(0,7)]
        )
        
        index_body = [11,12,24,23]
        for ind in index_body:
            landmark_body = results.pose_landmarks.landmark[ind]
            selection_body.append({
                'index_body' : ind,
                'xbody' : landmark_body.x,
                'ybody' : landmark_body.y,
                'zbody' : landmark_body.z
            })

        
        index_hand_left = [12,14,16]
        for ind in index_hand_left:
            landmark_lefthand = results.pose_landmarks.landmark[ind]
            selection_hand_left.append({
                'index_lefthand' : ind,
                'xlefthand' : landmark_lefthand.x,
                'ylefthand' : landmark_lefthand.y,
                'zlefthand' : landmark_lefthand.z
            })

        index_hand_right = [11,13,15]
        for ind in index_hand_right:
            landmark_righthand = results.pose_landmarks.landmark[ind]
            selection_hand_right.append({
                'index_righthand' : ind,
                'xrighthand' : landmark_righthand.x,
                'yrighthand' : landmark_righthand.y,
                'zrighthand' : landmark_righthand.z
            })

        index_rightleg = [23,25,27]
        for ind in index_rightleg:
            landmark_rightleg = results.pose_landmarks.landmark[ind]
            selection_legright.append({
                'index_rightleg' : ind,
                'xrightleg' : landmark_rightleg.x,
                'yrightleg' : landmark_rightleg.y,
                'zrightleg' : landmark_rightleg.z
            })

        index_leftleg = [24,26,28]
        for ind in index_leftleg:
            landmark_leftleg = results.pose_landmarks.landmark[ind]
            selection_legleft.append({
                'index_leftleg' : ind,
                'xleftleg' : landmark_leftleg.x,
                'yleftleg' : landmark_leftleg.y,
                'zleftleg' : landmark_leftleg.z
            })
        
        index_nose = 0
        landmark_nose = results.pose_landmarks.landmark[index_nose]
        selection_nose.append({
            'index_nose' : index_nose,
            'xnose' : landmark_nose.x,
            'ynose' : landmark_nose.y,
            'znose' : landmark_nose.z
            })

        index_leftear = 8
        landmark_earleft = results.pose_landmarks.landmark[index_leftear]
        selection_earleft.append({
            'index_earleft' : index_leftear,
            'xleftear' : landmark_earleft.x,
            'yleftear' : landmark_earleft.y,
            'zleftear' : landmark_earleft.z
        })

        index_rightear = 7
        landmark_earright = results.pose_landmarks.landmark[index_rightear]
        selection_earright.append({
            'index_earright' : index_rightear,
            'xrightear' : landmark_earright.x,
            'yrightear' : landmark_earright.y,
            'zrightear' : landmark_earright.z
        })
        
    flipped = cv2.flip(frame,1)
    resize = cv2.resize(flipped,(1920,1080))
    cv2.imshow('MediaPipe Pose', resize)
    
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()



