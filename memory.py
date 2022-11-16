import time
import numpy as np
import torch


def timeit(method):
    def timed(*args, **kw):
        ts = time.time()
        result = method(*args, **kw)
        te = time.time()
        print('%r  %2.2f ms' % (method.__name__, (te - ts) * 1000))
        return result

    return timed


class Memory:
    def __init__(self,
                 mem_size,
                 batch_size,
                 cam_state_dim,
                 sketch_dim,
                 pos_state_dim,
                 steering_queue_state_dim,
                 action_dim, device):
        self.device = device
        self.mem_size = mem_size
        self.batch_size = batch_size
        """
        cam_state_memory = (self.mem_size * (cam_state_dim[0] * cam_state_dim[1] * cam_state_dim[2])) * 1e-9  # uint8 bytes
        pos_state_memory = (self.mem_size * pos_state_dim) * 4e-9  # float32 bytes
        steering_queue_state_memory = (self.mem_size * steering_queue_state_dim) * 4e-9  # float32 bytes
        action_memory = self.mem_size * action_dim * 4e-9  # float32 bytes
        reward_memory = self.mem_size * 4e-9  # float32 bytes
        terminal_memory = self.mem_size * 4e-9  # float32 bytes
        apprx_mem_usage = cam_state_memory + pos_state_memory + steering_queue_state_memory + action_memory + reward_memory + terminal_memory
        print(f"Apprx RAM Usage by experience replay: {apprx_mem_usage:6.3f} GB!!!!!!!!")
        """
        # response = input("Do you want to continue?   (y/n")
        # if response == "n":
        #     print(f"okey stopping... ")
        #     sys.exit(1)

        self.depth_cam_memory = np.zeros((mem_size, cam_state_dim[0], cam_state_dim[1], cam_state_dim[2]), dtype=np.uint8)
        self.sketch_memory = np.zeros((mem_size, sketch_dim[0], sketch_dim[1], sketch_dim[2]), dtype=np.uint8)
        self.pos_state_memory = np.zeros((mem_size, pos_state_dim), dtype=np.float32)
        self.steering_queue_memory = np.zeros((mem_size, steering_queue_state_dim), dtype=np.float32)
        self.action_memory = np.zeros((mem_size, action_dim), dtype=np.float32)
        self.reward_memory = np.zeros(mem_size, dtype=np.float32)
        self.terminal_memory = np.zeros(mem_size, dtype=np.float32)
        self.gradmult_memory = np.ones(mem_size, dtype=np.float32)

        self.mem_pointer, self.mem_index, self.last_terminal = 0, 0, 0

    def store(self, sketch, depth_state, pos_state, steering_queue_state,
              action, next_sketch, new_depth_state, new_pos_state,
              new_steering_queue_state, reward, terminal, is_success,
              is_episode_maxstep, is_stopping_brake):
        index_2 = int((self.mem_index + 1) % self.mem_size)
        self.mem_index = int(self.mem_index)

        self.depth_cam_memory[self.mem_index] = depth_state
        self.depth_cam_memory[index_2] = new_depth_state

        self.sketch_memory[self.mem_index] = sketch
        self.sketch_memory[index_2] = next_sketch

        self.pos_state_memory[self.mem_index] = pos_state
        self.pos_state_memory[index_2] = new_pos_state

        self.steering_queue_memory[self.mem_index] = steering_queue_state
        self.steering_queue_memory[index_2] = new_steering_queue_state

        self.action_memory[self.mem_index] = action
        self.reward_memory[self.mem_index] = reward
        self.terminal_memory[self.mem_index] = int(terminal)

        self.gradmult_memory[self.mem_index] = 1 / ((1 - action[1]**2) + 1e-4)

        self.mem_index = (self.mem_index + 1) % self.mem_size
        self.mem_pointer = min(self.mem_pointer + 1, self.mem_size)

        if terminal:
            self.last_terminal = self.mem_index - 1

    def sample(self):
        batch_index = np.random.choice(self.mem_pointer, self.batch_size, replace=False)
        batch_index_2 = (batch_index + 1) % self.mem_size

        depth_batch = torch.from_numpy(self.depth_cam_memory[batch_index]).to(self.device)
        new_depth_batch = torch.from_numpy(self.depth_cam_memory[batch_index_2]).to(self.device)

        sketch_batch = torch.from_numpy(self.sketch_memory[batch_index]).to(self.device)
        next_sketch_batch = torch.from_numpy(self.sketch_memory[batch_index_2]).to(self.device)

        pos_state_batch = torch.from_numpy(self.pos_state_memory[batch_index]).to(self.device)
        new_pos_state_batch = torch.from_numpy(self.pos_state_memory[batch_index_2]).to(self.device)

        steering_queue_batch = torch.from_numpy(self.steering_queue_memory[batch_index]).to(self.device)
        new_steering_queue_batch = torch.from_numpy(self.steering_queue_memory[batch_index_2]).to(self.device)

        action_batch = torch.from_numpy(self.action_memory[batch_index]).to(self.device)
        reward_batch = torch.from_numpy(self.reward_memory[batch_index]).to(self.device)
        terminal_batch = torch.from_numpy(self.terminal_memory[batch_index]).to(self.device)

        gradmult_batch = torch.from_numpy(self.gradmult_memory[batch_index]).to(self.device)

        # st1 = time.time()
        """"
        depth_batch = torch.from_numpy(depth_batch).to(self.device)
        new_depth_batch = torch.from_numpy(new_depth_batch).to(self.device)
        
        
        pos_state_batch = torch.from_numpy(pos_state_batch).to(self.device)
        new_pos_state_batch = torch.from_numpy(new_pos_state_batch).to(self.device)
        
        steering_queue_batch = torch.from_numpy(steering_queue_batch).to(self.device)
        new_steering_queue_batch = torch.from_numpy(new_steering_queue_batch).to(self.device)
        
        action_batch = torch.from_numpy(action_batch).to(self.device)
        reward_batch = torch.from_numpy(reward_batch).to(self.device)
        terminal_batch = torch.from_numpy(terminal_batch).to(self.device)
        """
        # ft1 = time.time()
        # print(f"copy batch to gpu: {(ft1-st1)*1000} ms")

        return sketch_batch, next_sketch_batch, depth_batch, new_depth_batch, pos_state_batch, new_pos_state_batch, steering_queue_batch, new_steering_queue_batch, action_batch, reward_batch, terminal_batch, gradmult_batch, batch_index
