import cv2
import mediapipe as mp
import math
video = cv2.VideoCapture("Treino3.mp4")
pose = mp.solutions.pose
Pose = pose.Pose(min_tracking_confidence=0.5, min_detection_confidence=0.5)
draw = mp.solutions.drawing_utils

contador = 0
check = True
while True:
    success, img = video.read()
    videoRGB = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    results = Pose.process(videoRGB)
    points = results.pose_landmarks
    draw.draw_landmarks(img, points, pose.POSE_CONNECTIONS)

    h, w, _ = img.shape

    if points:
        #print(points)
        peDy = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].y*h)
        peDx = int(points.landmark[pose.PoseLandmark.RIGHT_FOOT_INDEX].x*w)

        peEy = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].y*h)
        peEx = int(points.landmark[pose.PoseLandmark.LEFT_FOOT_INDEX].x*w)

        moDy = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].y*h)
        moDx = int(points.landmark[pose.PoseLandmark.RIGHT_INDEX].x*w)

        moEy = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].y*h)
        moEx = int(points.landmark[pose.PoseLandmark.LEFT_INDEX].x*w)

        distMO = math.hypot(moDx-moEx, moDx-moEy)
        distPE = math.hypot(peDx-peEx, peDx-peDy)

        if check == True and distMO >= 700:
            contador +=1
            check = False
        if distMO <700:
            check = True

        texto = f"Numero: {contador}"
        cv2.rectangle(img,(20,240),(288,120),(255,0,0),-1)
        cv2.putText(img, texto, (40,200), cv2.FONT_HERSHEY_SIMPLEX, 2, (255,255,255), 5)

        print(f"Maos {distMO} Pes {distPE}")

    cv2.imshow("Resultado", img)
    cv2.waitKey(40)
