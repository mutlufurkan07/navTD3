import copy
import os
import ctypes
import time
import collections
import math

import cv2
import airsim
from airsim import Pose
import numpy as np
from airsim import to_eularian_angles
import matplotlib.pyplot as plt


def __dummyProjector__(xx, yy):
    # x_max, x_min = 243.27, -5.23
    # y_max, y_min = 140, -108.38
    relative_x = int((256 * xx / (243.27 + 5.23)) + 10)
    relative_y = int((256 * yy / (140 + 108.38)) + 111)
    return relative_x + 256, relative_y + 256


def cross_p(v1, v2):
    return v1[0] * v2[1] - v1[1] * v2[0]


def dot(v1, v2):
    return v1[0] * v2[0] + v1[1] * v2[1]


def getAgnle(v1, v2):
    return math.atan2(cross_p(v1, v2), dot(v1, v2))


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


def smallWait():
    stt = time.perf_counter()
    end = stt + 0.005
    while stt < end:
        stt = time.perf_counter()


def __compute_relative_pos__(xx, yy):
    # x_max, x_min = 243.27, -5.23
    # y_max, y_min = 140, -108.38
    relative_x = int((256 * xx / (243.27 + 5.23)) + 10)
    relative_y = int((256 * yy / (140 + 108.38)) + 111)
    return relative_x, relative_y


def __euclidianNorm1__(pos_x, pos_y, pos_instance):
    return math.sqrt((pos_x - pos_instance.x_val) ** 2 + (pos_y - pos_instance.y_val) ** 2)


def __euclidianNorm2__(pos_instance1, pos_instance2):
    d1 = pos_instance1.kinematics_estimated.position.x_val - pos_instance2.x_val
    d2 = pos_instance1.kinematics_estimated.position.y_val - pos_instance2.y_val
    return math.sqrt(d1 ** 2 + d2 ** 2)


def __checkSuccess__(car_state, pos_instance2):
    d1 = car_state.kinematics_estimated.position.x_val - pos_instance2.x_val
    d2 = car_state.kinematics_estimated.position.y_val - pos_instance2.y_val
    return math.sqrt(d1 ** 2 + d2 ** 2)


GODVIEW = False


class Simulation:
    def __init__(self, debugMode, training_params):

        self.training_params = training_params
        self.DEBUG_MODE = debugMode

        if self.DEBUG_MODE:
            # create a window to display the cam output if it is debug mode...
            if GODVIEW:
                cv2.namedWindow(f"GodView", cv2.WINDOW_FREERATIO)

            cv2.namedWindow(f"DepthPlanar1", cv2.WINDOW_FREERATIO)
            cv2.namedWindow(f"TargetMap", cv2.WINDOW_FREERATIO)

        self.sketch_np = cv2.imread(os.path.join(os.path.expanduser("~"), training_params["map_kroki_path_abs"]),
                                    cv2.IMREAD_GRAYSCALE)
        self.sketch9x9 = np.zeros((256 * 3, 256 * 3), dtype=np.uint8)
        self.sketch9x9[256:512, 256:512] = self.sketch_np

        # init some class variables
        self.control_interval_secs = 1 / (training_params["car_control_frequency"] * training_params["sim_clock_speed"])
        self.steering_queue = collections.deque(maxlen=training_params["last_steering_queue_len"])
        [self.steering_queue.append(0.0) for _ in range(training_params["last_steering_queue_len"])]
        self.episode_step_counter = 0
        self.old_pose = None
        self.reset_pos = airsim.utils.Vector3r(0, 0, 0)
        self.reset_orientation = airsim.utils.Quaternionr(1, 0, 0, 0)
        self.target_pos = airsim.utils.Vector3r(0, 0, 0)

        # connect to the map and give some logs
        self.req_im_lib = ctypes.cdll.LoadLibrary("req_im.dll")
        self.req_im_lib.mCl.restype = ctypes.c_void_p
        self.req_im_lib.check_connection.restype = ctypes.c_bool
        self.req_im_lib.check_connection.argtypes = [ctypes.c_void_p]
        self.req_im_lib.getImages.argtypes = [ctypes.c_void_p]

        self.req_im_lib_obj = self.req_im_lib.mCl()
        conn_error_flag = self.req_im_lib.check_connection(self.req_im_lib_obj)
        print(f"connection error from c++ side: {conn_error_flag}")

        self.client = airsim.CarClient(port=training_params["sim_apiServerPort"], timeout_value=30)
        self.client.confirmConnection()
        self.client.simSetCameraFov(camera_name="front", fov_degrees=training_params["camera_fov"],
                                    vehicle_name="CAR_0")
        self.client.simSetCameraFov(camera_name="right", fov_degrees=training_params["camera_fov"],
                                    vehicle_name="CAR_0")
        self.client.simSetCameraFov(camera_name="left", fov_degrees=training_params["camera_fov"], vehicle_name="CAR_0")
        self.client.enableApiControl(True)
        dd = training_params["sim_apiServerPort"]
        print(f"Successfully connected to port: {dd}")
        print(f"Control interval is {self.control_interval_secs} secs")

        self.car_controls = airsim.CarControls()
        self.last_camera_frames = collections.deque(maxlen=training_params["framestack_len"])
        self.last_sketches = collections.deque(maxlen=training_params["framestack_len"])
        self.aBlank_frame = np.zeros((72, 128 * 3), dtype=np.uint8)

        self.dummy_map = np.zeros((300, 300, 3), dtype=np.uint8)

    @timeit
    def reset(self, reset_pos, reset_orientation, target_pos):
        self.episode_step_counter = 0
        self.reset_pos, self.reset_orientation = reset_pos, reset_orientation
        self.target_pos = target_pos

        # reset some instance variables
        [self.steering_queue.append(0.0) for _ in range(self.training_params["last_steering_queue_len"])]

        self.__reset_car_controls__()
        self.client.simPause(False)
        self.client.reset()
        self.client.enableApiControl(True)
        self.__sendData2Sim__(position_data=True, car_control_data=True)
        smallWait()

        sketchs_out, camera_out, pos_state = self.__getResetState__()

        if self.DEBUG_MODE:
            cv2.imshow(f"DepthPlanar1", np.asarray(camera_out[-1], dtype=np.uint8))
            cv2.imshow(f"TargetMap", np.asarray(sketchs_out[-1], dtype=np.uint8))
            cv2.pollKey()

        return sketchs_out, camera_out, pos_state, np.asarray(self.steering_queue, dtype=np.float32)

    def step(self, action_gas, action_steering):
        # update the step counter for max_time_step alert
        self.episode_step_counter += 1

        # start sim, run the car for control interval seconds and stop the simulation
        self.client.simPause(False)
        stt = time.perf_counter()
        if -0.4 > action_gas > -0.5:
            self.car_controls.throttle = 0
            self.car_controls.brake = 0
        elif action_gas > -0.4:
            self.car_controls.throttle = (action_gas + 0.4) / 1.4
            self.car_controls.brake = 0
        elif action_gas < -0.5:
            self.car_controls.throttle = 0
            self.car_controls.brake = -(action_gas + 0.5)

        self.car_controls.steering = action_steering
        # publish calculated throttle and steering to simulation
        self.__sendData2Sim__(position_data=False, car_control_data=True)
        end = stt + self.control_interval_secs
        while stt < end:
            stt = time.perf_counter()
        self.client.simPause(True)

        self.steering_queue.append(action_steering)
        sketch_out, camera_out, pos_state, collision_state, current_car_state = self.__getStepState__()

        # compare the current distance to target point with success_radiues for this step
        ds = __checkSuccess__(current_car_state, self.target_pos)
        is_success = ds < self.training_params["success_radius"]

        # calculate the reward for the current step
        episode_max_step = self.episode_step_counter > 1000
        if self.episode_step_counter == 1:
            self.old_pose = [self.reset_pos.x_val, self.reset_pos.y_val]

        pos_x = current_car_state.kinematics_estimated.position.x_val
        pos_y = current_car_state.kinematics_estimated.position.y_val

        # current distance to target
        distance_now = __euclidianNorm1__(pos_x, pos_y, self.target_pos)
        # distance to target one step before
        distance_before = __euclidianNorm1__(self.old_pose[0], self.old_pose[1], self.target_pos)
        # update old pose for next step of the episode
        self.old_pose = [pos_x, pos_y]

        # velocity rewards
        linear_v_x = current_car_state.kinematics_estimated.linear_velocity.x_val
        linear_v_y = current_car_state.kinematics_estimated.linear_velocity.y_val
        linear_v = math.sqrt(linear_v_x ** 2 + linear_v_y ** 2)
        stopping_brake_flag = not (action_gas > -0.4) and linear_v < 0.05 and self.episode_step_counter > 10
        distance_reward = (distance_before - distance_now)

        # reward = -0.01 * 100
        reward = 0
        # print(str(linear_v) + "\n")
        """
        if action_gas < -0.5:
            reward += 50 * action_gas
        """
        if linear_v < 0.99:
            reward -= 50  # bunun mındıstancerewarddan daha buyuk olması lazım, durmaktansa uzaklaşmayı tercıh etsınç
        # if distance_reward > 0:

        reward += distance_reward * 35  # 10du

        reward += is_success * 12000
        reward += collision_state * -3000  # 3000
        reward += episode_max_step * -4000  # 3000  # HATALI OLABILIR
        reward += stopping_brake_flag * -4000

        reward /= 100

        # reward -= 3  # step reward
        ###

        done = is_success or collision_state or episode_max_step or stopping_brake_flag
        if self.DEBUG_MODE:
            cv2.imshow(f"DepthPlanar1", np.asarray(camera_out[-1], dtype=np.uint8))
            cv2.imshow(f"TargetMap", np.asarray(sketch_out[-1], dtype=np.uint8))
            if GODVIEW:
                self.dummy_getGodViewImg()
            # cv2.waitKey(1)  # apprx 15msecs of delay on debug mode............!!!!!!!!!!!
            cv2.pollKey()  # apprx 15msecs of delay on debug mode............!!!!!!!!!!!

        return sketch_out, camera_out, pos_state, np.asarray(self.steering_queue, dtype=np.float32), \
               reward, done, is_success, episode_max_step, stopping_brake_flag

    # @timeit
    def __getResetState__(self):
        camera_state, kinematic_state, collision_state = self.__request_simData__(True, True)

        x_diff = self.target_pos.x_val - kinematic_state.kinematics_estimated.position.x_val
        y_diff = self.target_pos.y_val - kinematic_state.kinematics_estimated.position.y_val

        v1 = [x_diff, y_diff]
        v2 = [kinematic_state.kinematics_estimated.linear_velocity.x_val,
              kinematic_state.kinematics_estimated.linear_velocity.y_val]
        theta = getAgnle(v1, v2) / math.pi
        current_linear_speed = math.sqrt(v2[0] ** 2 + v2[1] ** 2) / 15
        distance2target = math.sqrt(x_diff ** 2 + y_diff ** 2) / self.training_params["max_distance_to_target"]

        linear_acceleration = math.sqrt(
            kinematic_state.kinematics_estimated.linear_acceleration.x_val ** 2 + kinematic_state.kinematics_estimated.linear_acceleration.y_val ** 2) / 9.8

        pos_state = np.asarray(
            [current_linear_speed,
             distance2target,
             math.cos(theta),
             math.sin(theta),
             linear_acceleration,
             kinematic_state.kinematics_estimated.angular_velocity.z_val / 15,
             kinematic_state.kinematics_estimated.angular_acceleration.z_val / 100],
            dtype=np.float32)

        centered2carMap = self.warpSketchAffine(self.sketch9x9, kinematic_state, self.target_pos)
        centered2carMap = cv2.resize(centered2carMap, (80, 80))
        # sketch_empty_player = cv2.resize(sketch_empty_player, (80, 80))
        # sketch_empty_goal = cv2.resize(sketch_empty_goal, (80, 80))

        [self.last_camera_frames.append(camera_state) for _ in range(self.training_params["framestack_len"])]
        [self.last_sketches.append(centered2carMap) for _ in range(self.training_params["framestack_len"])]

        camera_out = np.asarray(self.last_camera_frames)
        sketchs_out = np.asarray(self.last_sketches)
        print(f"Reseted and distance to target: {math.sqrt(x_diff ** 2 + y_diff ** 2):5.3f} meters")
        return sketchs_out, camera_out, pos_state

    # @timeit
    def __getStepState__(self):
        camera_state, kinematic_state, collision_state = self.__request_simData__(True, True)

        x_diff = self.target_pos.x_val - kinematic_state.kinematics_estimated.position.x_val
        y_diff = self.target_pos.y_val - kinematic_state.kinematics_estimated.position.y_val

        v1 = [x_diff, y_diff]
        v2 = [kinematic_state.kinematics_estimated.linear_velocity.x_val,
              kinematic_state.kinematics_estimated.linear_velocity.y_val]
        theta = getAgnle(v1, v2) / math.pi
        current_linear_speed = math.sqrt(v2[0] ** 2 + v2[1] ** 2) / 15
        distance2target = math.sqrt(x_diff ** 2 + y_diff ** 2) / self.training_params["max_distance_to_target"]

        linear_acceleration = math.sqrt(
            kinematic_state.kinematics_estimated.linear_acceleration.x_val ** 2 + kinematic_state.kinematics_estimated.linear_acceleration.y_val ** 2) / 9.8

        pos_state = np.asarray(
            [current_linear_speed,
             distance2target,
             math.cos(theta),
             math.sin(theta),
             linear_acceleration,
             kinematic_state.kinematics_estimated.angular_velocity.z_val / 15,
             kinematic_state.kinematics_estimated.angular_acceleration.z_val / 100],
            dtype=np.float32)

        # print(f"angular v: {kinematic_state.kinematics_estimated.angular_velocity}")
        centered2carMap = self.warpSketchAffine(self.sketch9x9, kinematic_state, self.target_pos)
        centered2carMap = cv2.resize(centered2carMap, (80, 80))

        self.last_camera_frames.append(camera_state)
        self.last_sketches.append(centered2carMap)
        camera_out = np.asarray(self.last_camera_frames)
        sketchs_out = np.asarray(self.last_sketches)
        return sketchs_out, camera_out, pos_state, collision_state, kinematic_state

    # @timeit
    def __request_simData__(self, is_depth, is_kinematics):
        camera_output, car_kinematics, collision_flag = None, None, None
        if is_depth:
            try:
                self.req_im_lib.getImages.restype = ctypes.POINTER(ctypes.c_uint8 * 27648)
                mPtr = self.req_im_lib.getImages(self.req_im_lib_obj)
                camera_output = np.asarray([i for i in mPtr.contents], dtype=np.uint8).reshape(72, 384)
                self.c1 = camera_output
                # camera_output = np.ctypeslib.as_array(mPtr, shape=(1,)).reshape(72, 384)
            except Exception as e:
                camera_output = self.c1
                print(e)
        if is_kinematics:
            try:
                car_kinematics = self.client.getCarState()
                self.c2 = car_kinematics
            except Exception as e:
                car_kinematics = self.c2
                print(e)
        try:
            collision_flag = self.client.simGetCollisionInfo().has_collided
        except Exception as e:
            collision_flag = False
            print(e)
        return camera_output, car_kinematics, collision_flag

    def dummy_getGodViewImg(self):
        resp = self.client.simGetImages([airsim.ImageRequest("FixedCamera1", airsim.ImageType.Scene, False, False)],
                                        external=True)[0]
        img1d = np.fromstring(resp.image_data_uint8, dtype=np.uint8)
        img1d = img1d.reshape((resp.height, resp.width, 3))
        cv2.imshow("GodView", img1d)

    # @timeit
    def warpSketchAffine(self, sketch_raw, car_state, expected_end_pos):
        sketch_raw_ = copy.deepcopy(sketch_raw)
        yaw_deg = to_eularian_angles(car_state.kinematics_estimated.orientation)[2] * 180 / math.pi
        car_relative2map_coordinates = __dummyProjector__(car_state.kinematics_estimated.position.x_val,
                                                          car_state.kinematics_estimated.position.y_val)
        xx1, xx2 = __dummyProjector__(expected_end_pos.x_val, expected_end_pos.y_val)

        cv2.rectangle(sketch_raw_, (xx1 - 4, xx2 - 4), (xx1 + 4, xx2 + 4), 255, -1)

        im_c = tuple(np.array(sketch_raw_.shape[1::-1]) / 2)
        mRotMat = cv2.getRotationMatrix2D(im_c, 90 + yaw_deg, 1.0)
        rotated_sketchBig = cv2.warpAffine(sketch_raw_, mRotMat, sketch_raw_.shape[1::-1], flags=cv2.INTER_LINEAR)

        car_relative2map_coordinates_np = np.asarray(
            [car_relative2map_coordinates[0], car_relative2map_coordinates[1], 1])
        car_relative2map_coordinates_np = mRotMat @ car_relative2map_coordinates_np
        car_relative2map_coordinates = (
            int(car_relative2map_coordinates_np[0]), int(car_relative2map_coordinates_np[1]))
        cv2.circle(rotated_sketchBig, car_relative2map_coordinates, 3, 255, -1)
        x1 = car_relative2map_coordinates[0]
        y1 = car_relative2map_coordinates[1]
        centered2carMap = rotated_sketchBig[y1 - 128:y1 + 128, x1 - 128:x1 + 128]
        return centered2carMap

    def dummy_sketch_plot(self):
        x_max, x_min = 243.27, -5.23
        y_max, y_min = 140, -108.38
        _, klk, _ = self.__request_simData__(False, True)
        relative_x = int((256 * klk.kinematics_estimated.position.x_val / (x_max - x_min)) + 10)
        relative_y = int((256 * klk.kinematics_estimated.position.y_val / (y_max - y_min)) + 111)
        self.sketch_np[relative_y, relative_x] = 255
        plt.imshow(self.sketch_np, cmap="gray")
        plt.show()

    def __sendData2Sim__(self, position_data=False, car_control_data=False):
        if car_control_data:
            self.client.setCarControls(self.car_controls, vehicle_name="CAR_0")
        if position_data:
            self.client.simSetVehiclePose(Pose(self.reset_pos, self.reset_orientation), ignore_collision=True)

    def __reset_car_controls__(self):
        self.car_controls.throttle = 0
        self.car_controls.steering = 0
        self.car_controls.brake = 0
        self.car_controls.manual_gear = 1
        self.car_controls.is_manual_gear = True

    @staticmethod
    def read_points_from_txt(task_number):
        point_folder = open("points_" + str(task_number) + ".txt", 'r').read()
        points_location = point_folder.split("\n")[:-1]
        points = []
        for i in points_location:
            points.append(list(map(float, i.split(", "))))
        return points
