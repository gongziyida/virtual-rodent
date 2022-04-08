import numpy as np

class TimeStep:
    def __init__(self, state, reward):
        self.state = state.astype(np.float32)
        self.reward = reward

    def last(self):
        return False

class TestEnv:
    def __init__(self, beta=0.5):
        self.beta = beta
        self.reset()

    def step(self, action):
        new_state = action * self.beta + self.state
        new_state /= np.linalg.norm(new_state)
        self.reward = 1 - np.abs(np.dot(self.state, new_state))
        self.state = new_state
        return TimeStep(self.state, self.reward)

    def reset(self):
        theta = np.random.rand() * 2 * np.pi
        self.state = np.array([np.cos(theta), np.sin(theta)])
        self.reward = None
        return TimeStep(self.state, self.reward)


