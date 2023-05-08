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
                 state_dim,
                 action_dim, device):
        self.device = device
        self.mem_size = mem_size
        self.batch_size = batch_size

        state_memory = self.mem_size * state_dim * 4e-9 * 2 # float32 bytes
        action_memory = self.mem_size * action_dim * 4e-9  # float32 bytes
        reward_memory = self.mem_size * 4e-9  # float32 bytes
        terminal_memory = self.mem_size * 4e-9  # float32 bytes
        apprx_mem_usage = state_memory + action_memory + reward_memory + terminal_memory
        print(f"Apprx RAM Usage by experience replay: {apprx_mem_usage:6.3f} GB!!!!!!!!")
        if device == torch.device("cpu"):
            self.state_memory = np.zeros((mem_size, state_dim), dtype=np.float32)
            self.next_state_memory = np.zeros((mem_size, state_dim), dtype=np.float32)
            self.action_memory = np.zeros((mem_size, action_dim), dtype=np.float32)
            self.reward_memory = np.zeros(mem_size, dtype=np.float32)
            self.terminal_memory = np.zeros(mem_size, dtype=np.float32)
            self.gradmult_memory = np.ones(mem_size, dtype=np.float32)
        elif device == torch.device("cuda"):
            self.state_memory = torch.zeros((mem_size, state_dim), dtype=torch.float32).to(device)
            self.next_state_memory = torch.zeros((mem_size, state_dim), dtype=torch.float32).to(device)
            self.action_memory = torch.zeros((mem_size, action_dim), dtype=torch.float32).to(device)
            self.reward_memory = torch.zeros(mem_size, dtype=torch.float32).to(device)
            self.terminal_memory = torch.zeros(mem_size, dtype=torch.float32).to(device)
            self.gradmult_memory = torch.ones(mem_size, dtype=torch.float32).to(device)
        else:
            print(f"there is no device {device}")
            exit(1)

        self.mem_pointer, self.mem_index, self.last_terminal = 0, 0, 0

    def store(self, state, action, reward, next_state, terminal):
        self.mem_index = int(self.mem_index)
        if self.device == torch.device("cpu"):
            self.state_memory[self.mem_index] = state
            self.action_memory[self.mem_index] = action
            self.reward_memory[self.mem_index] = reward
            self.next_state_memory[self.mem_index] = next_state
            self.terminal_memory[self.mem_index] = int(terminal)

            self.gradmult_memory[self.mem_index] = 1 / ((1 - action[1] ** 2) + 1e-4)
        else:
            self.state_memory[self.mem_index] = torch.from_numpy(state).to(self.device)
            self.action_memory[self.mem_index] = torch.from_numpy(action).to(self.device)
            self.reward_memory[self.mem_index] = reward
            self.next_state_memory[self.mem_index] = torch.from_numpy(next_state).to(self.device)
            self.terminal_memory[self.mem_index] = int(terminal)

            self.gradmult_memory[self.mem_index] = 1 / ((1 - action[1] ** 2) + 1e-4)

        self.mem_index = (self.mem_index + 1) % self.mem_size
        self.mem_pointer = min(self.mem_pointer + 1, self.mem_size)

        if terminal:
            self.last_terminal = self.mem_index - 1

    def sample(self):
        batch_index = np.random.choice(self.mem_pointer, self.batch_size, replace=False)

        state_batch = self.state_memory[batch_index]
        action_batch = self.action_memory[batch_index]
        reward_batch = self.reward_memory[batch_index]
        next_state_batch = self.next_state_memory[batch_index]
        terminal_batch = self.terminal_memory[batch_index]
        gradmult_batch = self.gradmult_memory[batch_index]
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, gradmult_batch, batch_index

    @staticmethod
    def move2cuda(transition_tuple, device):
        state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, gradmult_batch, batch_index = transition_tuple
        state_batch = torch.from_numpy(state_batch).to(device)
        action_batch = torch.from_numpy(action_batch).to(device)
        reward_batch = torch.from_numpy(reward_batch).to(device)
        next_state_batch = torch.from_numpy(next_state_batch).to(device)
        terminal_batch = torch.from_numpy(terminal_batch).to(device)
        gradmult_batch = torch.from_numpy(gradmult_batch).to(device)
        return state_batch, action_batch, reward_batch, next_state_batch, terminal_batch, gradmult_batch, batch_index
