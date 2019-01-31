from math import fmod

import cv2 as cv
import numpy as np

from utils import TripodHeads, arc, read_from_yaml, get_objpoints

dhpara = np.array([[arc(90), 13, 10, arc(90)],
                   [arc(90), 8.6, 0, arc(90)],
                   [0, 52, 0, 0]])

servopara = [2, [7, 8], 50, [90, 90]]

tripodheads = TripodHeads(dhpara, servopara)

hand2cam = read_from_yaml('hand_eye_output.yaml', ['hand2cam'])[0]

tripodheads.set_hand2cam(hand2cam)

# 生成aruco二维码的字典
aruco_dict = cv.aruco.getPredefinedDictionary(1)

mtx, dist = read_from_yaml('camera_paraments.yaml', ['mtx', 'dist'])
objp, axisp = get_objpoints(2, 45.4)

cap = cv.VideoCapture(2)

while True:
    # 读摄像头
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    # 因为做了畸变纠正，以后要用新的相机内参矩阵
    newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
    # 畸变纠正
    frame = cv.undistort(frame, mtx, dist, None, newcameramtx)
    # 灰度处理快
    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    # 检测aruco二维码
    corners, ids, rejectedImgPoints = cv.aruco.detectMarkers(gray, aruco_dict, cameraMatrix=newcameramtx)

    # 如果检测到
    if corners:
        # 画出来
        cv.aruco.drawDetectedMarkers(frame, corners, ids)
        # 按每个二维码分开
        corner = np.array(corners).reshape(4, 2)
        corner = np.squeeze(np.array(corner))
        # 检测的点顺序是左上 右上 右下 左下 所以调换一下
        corner_pnp = np.array([corner[0], corner[1], corner[3], corner[2]])
        # solvePNP获取r,t矩阵
        retval, rvec, tvec = cv.solvePnP(objp, corner_pnp, newcameramtx, None)
        cv.aruco.drawAxis(frame, newcameramtx, np.zeros((1, 5)), rvec, tvec, 45.4)
        rm, _ = cv.Rodrigues(rvec)
        hvec = np.concatenate((np.concatenate((rm, tvec), axis=1), [[0, 0, 0, 1]]), axis=0)

    cv.imshow('img', frame)

    key = cv.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('a'):
        out = tripodheads.get_aimming_arc(hvec)
        deltatheta = [fmod(out[0], 2 * np.pi) / np.pi * 180, fmod(out[1], 2 * np.pi) / np.pi * 180]
        tripodheads.servo_run([0, 1], deltatheta)
        print('hvec:', hvec)
        print('result', deltatheta)

# 清理设备
cap.release()
cv.destroyAllWindows()
