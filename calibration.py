# 标定相机和输出内参和畸变矩阵

import cv2 as cv
import numpy as np

# 迭代条件
criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.001)
# chessboard真实坐标点
objp = np.zeros((9 * 6, 3), np.float32)
objp[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2) * 8.6
# 初始化存储空间
objpoints = []  # 3D点
imgpoints = []  # 2D点

cap = cv.VideoCapture(2)

distort = False  # 畸变纠正
record = False  # 记录数据

while True:
    ret, frame = cap.read()

    if distort:
        h, w = frame.shape[:2]
        newcameramtx, roi = cv.getOptimalNewCameraMatrix(mtx, dist, (w, h), 0, (w, h))
        # 畸变纠正
        frame = cv.undistort(frame, mtx, dist, None, newcameramtx)

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    # 找chessboard
    ret, corners = cv.findChessboardCorners(gray, (9, 6), None)
    # 亚像素迭代
    if ret:
        corners2 = cv.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)

        # 记录
        if record:
            objpoints.append(objp)
            imgpoints.append(corners2)
            print(len(imgpoints))
            record = False

        cv.drawChessboardCorners(frame, (9, 6), corners2, ret)

    cv.imshow('img', frame)

    key = cv.waitKey(1)
    if key & 0xFF == ord('a'):
        record = True
    elif key & 0xFF == ord('e'):
        # 保存到文件
        ret, mtx, dist, rvecs, tvecs = cv.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

        distort = True

        fs = cv.FileStorage('camera_paraments.yaml', cv.FileStorage_WRITE)
        fs.write('mtx', mtx)
        fs.write('dist', dist)
        fs.release()
    elif key & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
