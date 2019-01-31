from math import fmod

import Adafruit_PCA9685
import cv2 as cv
import numpy as np
import tensorflow as tf
from scipy.optimize import fsolve


class DH:
    def __init__(self, dh_parameters, hand2cam=None):
        # The units of dh_parameters are rad, millimetre, millimetre, rad.
        self.pairs = dh_parameters.shape[0]
        self.frames = self.pairs
        self.hand2cam = hand2cam
        self.transfomation_matrix = []
        self.dh_parameters = dh_parameters
        for pair in range(self.pairs):
            theta, d, a, alpha = dh_parameters[pair]
            matrix = np.array(
                (np.cos(theta), -np.sin(theta) * np.cos(alpha), np.sin(theta) * np.sin(alpha), a * np.cos(theta),
                 np.sin(theta), np.cos(theta) * np.cos(alpha), -np.cos(theta) * np.sin(alpha), a * np.sin(theta),
                 0, np.sin(alpha), np.cos(alpha), d,
                 0, 0, 0, 1)).reshape(4, 4)
            self.transfomation_matrix.append(matrix)

    def _transformation(self, begin, end):
        # reverse matrix
        if begin > end:
            pairs = begin - end
            transfomation = np.linalg.inv(self.transfomation_matrix[end - 1])
            pairs -= 1
            for pair in range(1, pairs + 1):
                transfomation = transfomation @ np.linalg.inv(self.transfomation_matrix[end - 1 - pair])
            return transfomation
        # forward matrix
        elif begin < end:
            pairs = end - begin
            transfomation = self.transfomation_matrix[begin]
            pairs -= 1
            for pair in range(1, pairs + 1):
                transfomation = transfomation @ self.transfomation_matrix[begin + pair]
            return transfomation
        else:
            print('It is the same frame')
            return np.array((1, 0, 0, 0,
                             0, 1, 0, 0,
                             0, 0, 1, 0,
                             0, 0, 0, 0))

    def new_theta(self, new_thetas):
        for idx, new_theta in enumerate(new_thetas):
            if new_theta is not None:
                theta, d, a, alpha = self.dh_parameters[idx]
                new_theta += theta
                self.transfomation_matrix[idx] = \
                    np.array((np.cos(new_theta), -np.sin(new_theta) * np.cos(alpha), np.sin(new_theta) * np.sin(alpha),
                              a * np.cos(new_theta),
                              np.sin(new_theta), np.cos(new_theta) * np.cos(alpha), -np.cos(new_theta) * np.sin(alpha),
                              a * np.sin(new_theta),
                              0, np.sin(alpha), np.cos(alpha), d,
                              0, 0, 0, 1)).reshape(4, 4)
                self.dh_parameters[idx][0] = new_theta

    def transformation(self, begin, end):
        return self._transformation(begin, end)

    def rotation(self, begin, end):
        return self._transformation(begin, end)[0:3, 0:3]

    def transition(self, begin, end):
        return self._transformation(begin, end)[0:3, 3]

    def get_newvec(self, delta_theta):
        _transfomation_matrix = self.transfomation_matrix.copy()

        theta, d, a, alpha = self.dh_parameters[0]
        _transfomation_matrix[0] = np.array(
            (np.cos(theta + delta_theta[0]), -np.sin(theta + delta_theta[0]) * np.cos(alpha),
             np.sin(theta + delta_theta[0]) * np.sin(alpha), a * np.cos(theta + delta_theta[0]),
             np.sin(theta + delta_theta[0]), np.cos(theta + delta_theta[0]) * np.cos(alpha),
             -np.cos(theta + delta_theta[0]) * np.sin(alpha), a * np.sin(theta + delta_theta[0]), 0,
             np.sin(alpha), np.cos(alpha), d, 0, 0, 0, 1)).reshape(4, 4)

        theta, d, a, alpha = self.dh_parameters[1]
        _transfomation_matrix[1] = np.array(
            (np.cos(theta + delta_theta[1]), -np.sin(theta + delta_theta[1]) * np.cos(alpha),
             np.sin(theta + delta_theta[1]) * np.sin(alpha), a * np.cos(theta + delta_theta[1]),
             np.sin(theta + delta_theta[1]), np.cos(theta + delta_theta[1]) * np.cos(alpha),
             -np.cos(theta + delta_theta[1]) * np.sin(alpha), a * np.sin(theta + delta_theta[1]), 0,
             np.sin(alpha), np.cos(alpha), d, 0, 0, 0, 1)).reshape(4, 4)

        transfomation = _transfomation_matrix[0] @ _transfomation_matrix[1] @ _transfomation_matrix[2]
        return transfomation


def arc(angle):
    return fmod(angle, 360) / 180. * np.pi


class Servo:
    def __init__(self, servo_para):
        # generate the ids from num and pins respectively, set the pwm as frequency, and turn the servos to the init_angle
        num, pins, frequency, init_angle = servo_para
        self.num = num
        self.pins = pins
        self.frequency = frequency
        self.pwm = Adafruit_PCA9685.PCA9685()
        self.pwm.set_pwm_freq(frequency)
        self.pwm.set_pwm(15, 0, 4095)
        self.angle = []
        for i in range(num):
            self.angle.append(init_angle[i])
            self.set_servo_angle(self.pins[i], self.angle[i])

    def set_servo_angle(self, channel, angle):
        pulse_length = 1000000
        pulse_length //= self.frequency
        pulse_length //= 4096

        pulse = 0.5 + angle / 180 * 2
        pulse *= 1000
        pulse //= pulse_length
        self.pwm.set_pwm(channel, 0, int(pulse))

    def run(self, ids, angle):
        # set the servos to the angle respectively
        for i, id in enumerate(ids):
            self.angle[id] += angle[i]
            self.set_servo_angle(self.pins[id], self.angle[id])


class TripodHeads:
    def __init__(self, dh_para, servo_para):
        self.dh = DH(dh_para)
        self.servo = Servo(servo_para)

    def servo_run(self, ids, angle):
        self.servo.run(ids, angle)
        new_theta = []
        for i in range(self.servo.num):
            if i in ids:
                new_theta.append(arc(angle[ids.index(i)]))
            else:
                new_theta.append(None)
        self.dh.new_theta(new_theta)

    def equlization(self, x, hvec):
        qvec = self.dh.transformation(0, 3) @ self.dh.hand2cam @ hvec
        qr_loc = qvec[0:3, 3]

        _x = x.reshape(2)

        newvec = self.dh.get_newvec(_x) @ self.dh.hand2cam
        new_loc = newvec[0:3, 3]

        loc_array = qr_loc - new_loc
        loc_array = loc_array / np.sqrt(np.sum(np.square(loc_array)))

        new_rot = newvec[0:3, 0:3]
        zero_equlization = (new_rot[0:3, 2] - loc_array).reshape(3)

        return [zero_equlization[0], zero_equlization[2]]

    def get_aimming_arc(self, hvec):
        if self.dh.hand2cam is not None:
            x_out = fsolve(self.equlization, np.array((0, 0)), args=hvec)
            return x_out
        else:
            print('no hand2cam matrix')
            return None

    def set_hand2cam(self, hand2cam):
        self.dh.hand2cam = hand2cam


def tf_get_cam_matrix(hvec, handvec):
    rotation_weight = tf.Variable([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=tf.float32)
    transition_weight = tf.Variable([[37.5], [0], [15]], dtype=tf.float32)
    weight = tf.concat([rotation_weight, transition_weight], 1)
    last = tf.constant([[0, 0, 0, 1]], dtype=tf.float32)
    weight_add = tf.concat([weight, last], 0)

    qvec = []
    for i in range(len(hvec)):
        qvec.append(tf.matmul(tf.matmul(handvec[i], weight_add), hvec[i]))

    loss = 0
    for i in range(len(hvec) - 1):
        loss += tf.reduce_mean(tf.slice(tf.square(qvec[i] - qvec[i + 1]), [0, 0], [3, 3]))

    cut_weight = tf.slice(weight_add, [0, 0], [3, 3])
    loss += tf.reduce_mean(tf.square(tf.reduce_sum(tf.square(cut_weight), 1) - 1))
    loss += tf.reduce_mean(tf.square(tf.reduce_sum(tf.square(tf.transpose(cut_weight)), 1) - 1))

    opmizer = tf.train.AdadeltaOptimizer()
    train = opmizer.minimize(loss)

    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    train = opmizer.minimize(loss, var_list=rotation_weight)

    for i in range(100000):
        sess.run(train)
        if i % 1000 == 0:
            print(i, sess.run(weight_add), 'loss:', sess.run(loss))

    loss = 0
    for i in range(len(hvec) - 1):
        loss += tf.reduce_mean(tf.slice(tf.square(qvec[i] - qvec[i + 1]), [0, 3], [3, 1]))

    train = opmizer.minimize(loss, var_list=transition_weight)

    for i in range(100000):
        sess.run(train)
        if i % 1000 == 0:
            print(i, sess.run(weight_add), 'loss:', sess.run(loss))

    x_predict = sess.run(weight_add)

    return x_predict


def read_from_yaml(loc, names):
    _fs = cv.FileStorage(loc, cv.FileStorage_READ)

    _data = []
    for name in names:
        _data.append(np.array(_fs.getNode(name).mat()).astype(np.float32))
    _fs.release()
    return _data


# 画目标的三个坐标轴
def draw(_img, _corners, _imgpts):
    _corners = tuple(_corners[0].ravel())
    _img = cv.line(_img, _corners, tuple(_imgpts[0].ravel()), (255, 0, 0), 5)
    _img = cv.line(_img, _corners, tuple(_imgpts[1].ravel()), (0, 255, 0), 5)
    _img = cv.line(_img, _corners, tuple(_imgpts[2].ravel()), (0, 0, 255), 5)
    return _img


# 获取真实尺寸点
def get_objpoints(_num, _lenth):
    _objp = np.zeros((_num * _num, 3), np.float32)
    _objp[:, :2] = np.mgrid[0:2, 0:2].T.reshape(-1, 2) * _lenth

    _axisp = np.array([[0, 0, 0],
                       [_lenth, 0, 0],
                       [0, _lenth, 0],
                       [0, 0, _lenth]], dtype=np.float32)

    return _objp, _axisp


if __name__ == "__main__":
    dhpara = np.array([[arc(90), 13, 10, arc(90)],
                       [arc(90), 8.6, 0, arc(90)],
                       [0, 52, 0, 0]])

    servopara = [2, [7, 8], 50, [90, 90]]

    tripodheads = TripodHeads(dhpara, servopara)

    print('pins:', tripodheads.servo.pins)
    print('angle:', tripodheads.servo.angle)
    print('theta:', tripodheads.dh.dh_parameters)

    tripodheads.servo_run([1], [5])

    print('angle:', tripodheads.servo.angle)
    print('theta:', tripodheads.dh.dh_parameters)

    tripodheads.servo_run([1, 0], [-5, 10])

    print('angle:', tripodheads.servo.angle)
    print('theta:', tripodheads.dh.dh_parameters)
