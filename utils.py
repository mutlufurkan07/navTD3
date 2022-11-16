import random
import torch

import psutil
import signal
import traceback
from airsim.utils import *


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.enabled = True
    torch.autograd.profiler.emit_nvtx(False)
    torch.autograd.profiler.profile(False)
    torch.autograd.set_detect_anomaly(False)


def state_generator(positions, orientations, min_threshold=6, max_threshold=40, seed=19):
    is_valid = False
    min_t = min_threshold ** 2
    max_t = max_threshold ** 2
    #np.random.seed(seed)
    k1 = np.random.randint(0, positions.shape[0] - 1)
    st_pos = Vector3r(x_val=positions[k1, 0], y_val=positions[k1, 1], z_val=positions[k1, 2])
    st_orientation = Quaternionr(w_val=orientations[k1, 0], x_val=orientations[k1, 1], y_val=orientations[k1, 2],
                                 z_val=orientations[k1, 3])
    k2 = np.random.randint(0, positions.shape[0] - 1)
    end_pos = Vector3r(x_val=positions[k2, 0], y_val=positions[k2, 1], z_val=positions[k2, 2])

    while not is_valid:
        d = ((st_pos.x_val - end_pos.x_val) ** 2 + (st_pos.y_val - end_pos.y_val) ** 2)
        # print(f"d: {math.sqrt(d)}")
        if max_t > d > min_t:
            break
        k2 = np.random.randint(0, positions.shape[0] - 1)
        end_pos = Vector3r(x_val=positions[k2, 0], y_val=positions[k2, 1], z_val=positions[k2, 2])

    # print(f"d: {math.sqrt(d)}")
    return st_pos, st_orientation, end_pos


def state_reader(name):
    # get positions and orientations for the car client in Airsim
    start_end_poses = []
    with open(name, "r") as s:
        for line in s:
            start_end_poses.append(line)

    number_of_states = len(start_end_poses)
    starts_pos = []
    start_quaternion = []

    for (i, pose) in enumerate(start_end_poses):
        kk = pose.split(", ")
        starts_pos.append(kk[:3])
        start_quaternion.append(kk[3:])

    start_pos_xyz = np.zeros((number_of_states, 3))
    start_orientation_quaternion = np.zeros((number_of_states, 4))

    for indx in range(number_of_states):
        start_pos_xyz[indx, 0] = float(starts_pos[indx][0])
        start_pos_xyz[indx, 1] = float(starts_pos[indx][1])
        start_pos_xyz[indx, 2] = float(starts_pos[indx][2])

        start_orientation_quaternion[indx, 0] = float(start_quaternion[indx][0])
        start_orientation_quaternion[indx, 1] = float(start_quaternion[indx][1])
        start_orientation_quaternion[indx, 2] = float(start_quaternion[indx][2])
        start_orientation_quaternion[indx, 3] = float(start_quaternion[indx][3][:-1])

    del kk, s
    del i, indx, line, pose
    print(f"Found: {number_of_states} points from txt: {name}")
    return start_pos_xyz, start_orientation_quaternion


def json_clock_speed_change(pathname="/Documents/AirSim/settings.json", new_clock_speed=7):
    setting_json_dir = pathname
    with open(setting_json_dir, 'r+') as json:
        k = json.readlines()
        for i, line in enumerate(k):
            index = line.find("ClockSpeed")
            if not index == -1:
                new_line = line.replace(line, f'  "ClockSpeed": {int(new_clock_speed)},\n')
                k[i] = new_line

    with open(setting_json_dir, 'w') as json:
        for line in k:
            json.write(line)


def json_port_change(pathname="/Documents/AirSim/settings.json", new_port=41451):
    setting_json_dir = pathname
    with open(setting_json_dir, 'r+') as json:
        k = json.readlines()
        for i, line in enumerate(k):
            index = line.find("ApiServerPort")
            if not index == -1:
                new_line = line.replace(line, f'  "ApiServerPort": {int(new_port)},\n')
                k[i] = new_line

    with open(setting_json_dir, 'w') as json:
        for line in k:
            json.write(line)


def json_viewmode_change(pathname="/Documents/AirSim/settings.json", is_display=False):
    if is_display:
        new_mode = ""
    else:
        new_mode = "NoDisplay"
    setting_json_dir = pathname
    with open(setting_json_dir, 'r+') as json:
        k = json.readlines()
        for i, line in enumerate(k):
            index = line.find("ViewMode")
            if not index == -1:
                new_line = line.replace(line, f'  "ViewMode": "{str(new_mode)}",\n')
                k[i] = new_line

    with open(setting_json_dir, 'w') as json:
        for line in k:
            json.write(line)


def json_image_width_height_change(pathname="/Documents/AirSim/settings.json", width=320, height=240):
    setting_json_dir = pathname
    found1 = False
    found2 = False
    with open(setting_json_dir, 'r+') as json:
        k = json.readlines()
        for i, line in enumerate(k):
            index = line.find("Width")
            if not index == -1:
                if not found1:
                    new_line = line.replace(line, f'                "Width": {int(width)},\n')
                    k[i] = new_line
                    found1 = True
                else:
                    new_line = line
                    k[i] = new_line

    with open(setting_json_dir, 'w') as json:
        for line in k:
            json.write(line)

    with open(setting_json_dir, 'r+') as json:
        k = json.readlines()
        for i, line in enumerate(k):
            index = line.find("Height")
            if not index == -1:
                if not found2:
                    new_line = line.replace(line, f'                "Height": {int(height)},\n')
                    k[i] = new_line
                    found2 = True
                else:
                    new_line = line
                    k[i] = new_line

    with open(setting_json_dir, 'w') as json:
        for line in k:
            json.write(line)


# runs sh file
def map_runner(path):
    os.system("start " + path)


# makes sure a process is dead given its pid
def kill_a_process(pid_):
    try:
        parent = psutil.Process(pid_)
        children = parent.children(recursive=True)
        for p in children:
            os.kill(p.pid, signal.SIGKILL)
        os.kill(parent.pid, signal.SIGKILL)
        try:
            # Wait for the child processes to guarantee their death
            for p in children:
                os.waitpid(p.pid, os.WNOHANG)
        except Exception as _:
            print(traceback.format_exc())
        try:
            # Wait for the parent processes to guarantee their death
            os.waitpid(parent.pid, os.WNOHANG)
        except Exception as _:
            print(traceback.format_exc())
    except Exception as _:
        print(f"OKEY, map process is already dead.")


current_seed = 12
print(f"Seed is set to : {current_seed}")
set_seed(current_seed)
