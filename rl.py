import os, random
from collections import deque
import torch

class TrainRodent:
    def __init__(self, model, batch_size, save_dir):
        self.model = model
        self.batch_size = batch_size
        self._memory = deque(maxlen=1e6)
        self.save_dir = save_dir

    def cache(self, cached):
        self._memory.append(cached)

    def recall(self):
        batch = random.sample(self._memory, self.batch_size)
        state, next_state, action, reward, done = map(torch.stack, zip(*batch))
        return state, next_state, action, reward, done

    def learn(self):
        state, next_state, action, reward, done = self.recall()

    def save(self):
        torch.save(dict(model=self.model.state_dict(), 
                        exploration_rate=self.exploration_rate,
                        step=self.step), 
                   os.path.join(self.save_path, '%d.pt' %  self.step))


