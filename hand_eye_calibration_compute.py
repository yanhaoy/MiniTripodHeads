import cv2 as cv
import numpy as np

from utils import TripodHeads, arc, tf_get_cam_matrix, read_from_yaml

hvec = read_from_yaml('hand_eye_paraments.yaml', ['hvec'])[0]
dhpara = np.array([[arc(90), 13, 10, arc(90)],
                   [arc(90), 8.6, 0, arc(90)],
                   [0, 52, 0, 0]])

servopara = [2, [8, 7], 50, [0, 0]]

tripodheads = TripodHeads(dhpara, servopara)

handvec = []
tripodheads.servo_run([0, 1], [-15, -5])
handvec.append(tripodheads.dh.transformation(0, 3).astype(np.float32))
tripodheads.servo_run([0, 1], [15, 5])

tripodheads.servo_run([0, 1], [0, 10])
handvec.append(tripodheads.dh.transformation(0, 3).astype(np.float32))
tripodheads.servo_run([0, 1], [0, -10])

tripodheads.servo_run([0, 1], [10, -5])
handvec.append(tripodheads.dh.transformation(0, 3).astype(np.float32))
tripodheads.servo_run([0, 1], [-10, 5])

tripodheads.servo_run([0, 1], [-15, 0])
handvec.append(tripodheads.dh.transformation(0, 3).astype(np.float32))
tripodheads.servo_run([0, 1], [15, 0])

tripodheads.servo_run([0, 1], [10, 10])
handvec.append(tripodheads.dh.transformation(0, 3).astype(np.float32))
tripodheads.servo_run([0, 1], [-10, -10])

tripodheads.servo_run([0, 1], [0, -5])
handvec.append(tripodheads.dh.transformation(0, 3).astype(np.float32))
tripodheads.servo_run([0, 1], [0, 5])

tripodheads.servo_run([0, 1], [-15, 10])
handvec.append(tripodheads.dh.transformation(0, 3).astype(np.float32))
tripodheads.servo_run([0, 1], [15, -10])

tripodheads.servo_run([0, 1], [10, 0])
handvec.append(tripodheads.dh.transformation(0, 3).astype(np.float32))
tripodheads.servo_run([0, 1], [-10, 0])

x_predict = tf_get_cam_matrix(hvec, handvec)
print(x_predict)

fs = cv.FileStorage('hand_eye_output.yaml', cv.FileStorage_WRITE)
fs.write('hand2cam', np.array(x_predict))
