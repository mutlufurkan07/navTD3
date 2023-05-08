from multiprocessing import Process
from collections import deque

from TD3Agent import TD3Agent

from utils import *
from mSimulationCar import Simulation
import copy
import math
import random


class TrainingController:
    termination_statuses = {1: "Success", 2: "Collision", 4: "EpisodeMaxStep500", 5: "StoppingBrake"}

    def __init__(self,
                 training_path,
                 logs_file_path,
                 training_params,
                 log_files,
                 debugMode):

        # self.
        self.map_process, self.map_process_pid = None, None
        if log_files[0] is None or training_params is None:
            sys.exit()
        # some class variables to continue training
        self.training_log_file = log_files[0]
        self.parameters_l_file = log_files[1]
        self.training_params = training_params

        # success variables
        dd = training_params["success_queue_length"]
        self.success_queue = deque(maxlen=dd)
        [self.success_queue.append(0) for _ in range(dd)]
        self.average_reward, self.average_success = deque(maxlen=dd), deque(maxlen=dd)
        self.best_success_rate = 0.0
        self.success_ratio = np.mean(self.success_queue)

        # change to true map and settings path
        self.json_pathname = os.path.join(os.path.expanduser("~"), training_params["settingsjson_path_abs"])
        self.map_pathname = os.path.join(os.path.expanduser("~"), training_params["map_path_abs"])

        # create model parameters file
        self.saved_agents_path = os.path.join(logs_file_path, "model_params")
        os.mkdir(self.saved_agents_path)

        # open map
        self.map_process_creator()
        time.sleep(5)

        self.start_pos_xyzs, self.start_orientations = state_reader(os.path.join(training_path, "points2.txt"))

        self.agent = TD3Agent(training_params=training_params,
                              save_path=self.saved_agents_path,
                              std=0.1,
                              action_dim=2,
                              cam_dim=(4, 72, 128 * 3),
                              sketch_dim=(4, 80, 80),
                              pos_dim=7,
                              noise_mag=0.1,
                              max_action=1,
                              smoothing_noise_limit=0.1)

        # TRAINING_PARAMS
        self.current_episode_number, self.current_training_timestep = 0, 0
        self.training_start_time = time.time()

        while self.current_training_timestep < self.training_params["max_training_timesteps"]:
            self.env = Simulation(debugMode=debugMode, training_params=training_params)
            self.train_loop()
            break

    def train_loop(self):
        # 19, 31, 33, 8, 66
        # 10 ters tarafta
        # 33 sıkıntılı
        # 121 ve 123 tam sağında hedef
        # 127 biraz solunda
        # 267 biraz solunda
        klk = np.asarray([19, 31, 8, 66, 10, 121, 123, 127, 267, 44, 83, 1995, 2000, 467, 333, 94, 312, 989, 545, 4321])
        # 19, 31, 33, 8, 66, 10, 121, 123, 127, 267
        # 44, 83, 1995, 2000, 467, 333, 94, 312, 989, 111, 545, 4321
        # act_tanh = math.tanh(0.9)
        while self.current_training_timestep < self.training_params["max_training_timesteps"]:

            current_episode_reward = 0
            ss = random.choice(klk)
            print(ss)
            dummy1 = self.env.reset(*self.generate_state(seed=ss))
            sketch, depth_image, pos_state, past_steerings = dummy1

            while True:
                st = time.time()
                action = self.agent.action_selection(sketch, depth_image, pos_state, past_steerings,
                                                     self.current_training_timestep)  # 3ms

                # action = action / act_tanh
                dummy2 = self.env.step(action_gas=action[0], action_steering=action[1])  # 45ms
                # action = action * act_tanh
                next_sketch, next_depth_image, next_pos_state, next_past_steerings, reward, done, is_success, is_episode_maxstep, is_stopping_brake = dummy2

                if not is_episode_maxstep:
                    self.agent.memory.store(sketch, depth_image, pos_state, past_steerings, action, *dummy2)  # 1ms

                # self.agent.memory.store(sketch, depth_image, pos_state, past_steerings, action, *dummy2)  # 1ms

                current_episode_reward += reward
                sketch = next_sketch
                depth_image = next_depth_image
                pos_state = next_pos_state
                past_steerings = next_past_steerings

                self.current_training_timestep += 1
                a = self.agent.actor.action_before_tanh
                print(
                    f"\rone loop: {(time.time() - st) * 1000}, x,y: {a[0]:.4f},{a[1]:.4f}"
                    f"d:{pos_state[1]}, reward: {reward}", end="")

                if self.current_training_timestep > self.training_params["exploration_timesteps"]:
                    self.agent.update(is_policy_update=self.current_training_timestep % 2 == 0)

                if self.current_training_timestep % 5000 == 1:
                    self.agent.save_models()
                if done:
                    self.current_episode_number += 1
                    self.average_reward.append(current_episode_reward)

                    h, m, remaining_h, remaining_m = self.trainingCalculateTimeStatistics()

                    self.success_queue.append(int(is_success))
                    self.average_success = np.mean(self.success_queue)
                    termination_status = 2
                    if is_success:
                        termination_status = 1
                    elif is_episode_maxstep:
                        termination_status = 4
                    elif is_stopping_brake:
                        termination_status = 5

                    log_str = f"\nEpisode: {self.current_episode_number:5d} " \
                              f"Step: {self.current_training_timestep:7d}  " \
                              f"Reward: {current_episode_reward:8.2f} " \
                              f"Termination status: {self.termination_statuses[termination_status]} " \
                              f"Avg Reward: {np.mean(self.average_reward):5.2f} " \
                              f"Avg Success: {self.average_success:5.2f} " \
                              f"Time Passed: {h} h {m} m Remaining Time: {remaining_h} h {remaining_m} m Seed: {ss} "
                    print(log_str)
                    self.training_log_file.write(log_str + "\n")
                    self.training_log_file.flush()
                    print(f"steer crtc1 {self.agent.critic1.third.weight[0, 257]}")
                    print(f"steer crtc2 {self.agent.critic2.third.weight[0, 257]}")
                    print(f"acc crtc1 {self.agent.critic1.third.weight[0, 256]}")
                    print(f"acc crtc2 {self.agent.critic2.third.weight[0, 256]}")
                    depth = torch.from_numpy(depth_image).cuda().unsqueeze(0).float()
                    pos = torch.from_numpy(pos_state).cuda().unsqueeze(0).float()
                    sketch = torch.from_numpy(sketch).cuda().unsqueeze(0).float()
                    past_st = torch.from_numpy(past_steerings).cuda().unsqueeze(0).float()
                    act = torch.from_numpy(action).cuda().unsqueeze(0).float()
                    with torch.no_grad():
                        qq1 = self.agent.critic1.forward(sketch, depth, pos, act, past_st)
                        qq2 = self.agent.critic2.forward(sketch, depth, pos, act, past_st)
                    print(f"Q1: {qq1}")
                    print(f"Q2: {qq2}")
                    break

    def trainingCalculateTimeStatistics(self):
        time_difference = time.time() - self.training_start_time
        hour = time_difference // 3600
        minute = time_difference // 60 - hour * 60
        expected_total_time = time_difference * self.training_params[
            "max_training_timesteps"] / self.current_training_timestep
        expected_rem_time = expected_total_time - time_difference
        exp_hour = expected_rem_time // 3600
        exp_min = expected_rem_time // 60 - exp_hour * 60
        return hour, minute, exp_hour, exp_min

    def simPlotTargetPoint(self, end_pos):
        self.env.client.simFlushPersistentMarkers()
        self.env.client.simPlotPoints(points=[Vector3r(end_pos.x_val, end_pos.y_val, end_pos.z_val - 4)],
                                      color_rgba=[0, 0, 1.0, 0.01], size=15, duration=10, is_persistent=True)

    # samples a st pos and end pos from the map
    def generate_state(self, seed):
        st_pos, st_orientation, end_pos = state_generator(positions=self.start_pos_xyzs,
                                                          orientations=self.start_orientations,
                                                          min_threshold=self.training_params["min_distance_to_target"],
                                                          max_threshold=self.training_params["max_distance_to_target"],
                                                          seed=seed)
        a_flag = bool(random.randint(0, 1))
        if a_flag:
            # get reverse here
            new_end = copy.deepcopy(end_pos)
            end_pos = st_pos
            st_pos = new_end
            ori1 = to_eularian_angles(st_orientation)[2] * 180 / math.pi
            ori3 = to_eularian_angles(st_orientation)
            ori2 = (180 + ori1) / 180 * math.pi
            st_orientation = to_quaternion(ori3[0], ori3[1], ori2)
            return st_pos, st_orientation, end_pos

        return st_pos, st_orientation, end_pos

    # creates process that runs map
    def map_process_creator(self):
        json_viewmode_change(pathname=self.json_pathname,
                             is_display=True if self.training_params["sim_isDisplayMap"] == 1 else False)
        json_port_change(pathname=self.json_pathname, new_port=self.training_params["sim_apiServerPort"])
        json_clock_speed_change(pathname=self.json_pathname, new_clock_speed=self.training_params["sim_clock_speed"])
        args_ = self.map_pathname + " --settings " + self.json_pathname
        self.map_process = Process(target=map_runner, args=(args_,), daemon=True)
        self.map_process.start()
        self.map_process_pid = self.map_process.pid
