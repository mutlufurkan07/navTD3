import time

import airsim
import cv2
from multiprocessing import Process
import os
from utils import *
import keyboard
import numpy as np


# creates process that runs map
def map_process_creator(map_pathname, json_pathname, api_server_port, sim_clock_speed=1):
    json_viewmode_change(pathname=json_pathname, is_display=True)
    # json_image_width_height_change(pathname=json_pathname, width=320, height=240)
    json_port_change(pathname=json_pathname, new_port=api_server_port)
    json_clock_speed_change(pathname=json_pathname, new_clock_speed=sim_clock_speed)
    args_ = map_pathname + " --settings " + json_pathname
    map_process = Process(target=map_runner, args=(args_,), daemon=True)
    map_process.start()


def map_runner(path):
    os.system("start " + path)


def collect_points():
    env_path = os.path.join(os.path.expanduser("~"), "Desktop", "ksalka", "AirSim_release_maps", "office2_debug",
                            "WindowsNoEditor", "office2.exe")
    json_path = os.path.join(os.path.expanduser("~"), "Desktop", "ksalka", "RL_Research", "autocarKagan",
                             "settings_office.json")
    port_id = 41462

    map_process_creator(map_pathname=env_path, json_pathname=json_path, api_server_port=port_id, sim_clock_speed=1)
    time.sleep(5)

    client = airsim.CarClient(port=port_id)
    client.confirmConnection()
    client.enableApiControl(False)
    file = open("points.txt", 'w')

    cv2.namedWindow("depth_cam", cv2.WINDOW_FREERATIO)
    cv2.namedWindow("map_points", cv2.WINDOW_FREERATIO)

    print("Please start selecting points.")
    count = 0

    duddm = np.zeros((500, 500, 3), dtype=np.uint8)
    while 1:
        responses = client.simGetImages([airsim.ImageRequest("front", airsim.ImageType.DepthPlanar, True)])
        image = airsim.list_to_2d_float_array(responses[0].image_data_float, responses[0].width,
                                              responses[0].height).clip(0, 60) / 60 * 255
        d = np.asarray(image, dtype=np.uint8)
        cv2.imshow("depth_cam", d)
        cv2.imshow("map_points", duddm)
        cv2.waitKey(1)

        if keyboard.is_pressed('k'):
            count += 1

            current_state = client.getCarState()
            x_point = current_state.kinematics_estimated.position.x_val
            y_point = current_state.kinematics_estimated.position.y_val
            z_point = current_state.kinematics_estimated.position.z_val

            duddm = cv2.circle(duddm, (int(230 - x_point), int(130 + y_point)), 2, (255, 255, 255), thickness=1)
            cv2.imwrite("mappoints.png", duddm)

            w_orient = current_state.kinematics_estimated.orientation.w_val
            x_orient = current_state.kinematics_estimated.orientation.x_val
            y_orient = current_state.kinematics_estimated.orientation.y_val
            z_orient = current_state.kinematics_estimated.orientation.z_val

            pitch, roll, yaw = airsim.utils.to_eularian_angles(current_state.kinematics_estimated.orientation)

            state_string = str(x_point) + ", " + str(y_point) + ", " + str(z_point) + ", " + str(w_orient) + ", " + \
                           str(x_orient) + ", " + str(y_orient) + ", " + str(z_orient) + "\n"
            print(state_string)
            print(count)
            file.write(state_string)
            file.flush()

        time.sleep(0.01)


if __name__ == "__main__":
    collect_points()
