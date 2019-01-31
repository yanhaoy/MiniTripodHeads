import cv2 as cv
import numpy as np

from utils import get_objpoints, read_from_yaml

# 生成aruco二维码的字典
aruco_dict = cv.aruco.getPredefinedDictionary(1)

mtx, dist = read_from_yaml('camera_paraments.yaml', ['mtx', 'dist'])
objp, axisp = get_objpoints(2, 45.4)

cap = cv.VideoCapture(2)
fourcc = cv.VideoWriter_fourcc(*'XVID')

hvec = []

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

    cv.imshow('img', frame)

    key = cv.waitKey(1)
    if key & 0xFF == ord('q'):
        break
    elif key & 0xFF == ord('a'):
        print('rvec:', rvec)
        print('tvec:', tvec)
        rm, _ = cv.Rodrigues(rvec)
        print('rm:', rm)
        hm = np.concatenate((np.concatenate((rm, tvec), axis=1), [[0, 0, 0, 1]]), axis=0)
        print('hm:', hm)
        hvec.append(hm)
        print(len(hvec))
    elif key & 0xFF == ord('e'):
        fs = cv.FileStorage('hand_eye_paraments.yaml', cv.FileStorage_WRITE)
        fs.write('hvec', np.array(hvec))

# 清理设备
cap.release()
cv.destroyAllWindows()
