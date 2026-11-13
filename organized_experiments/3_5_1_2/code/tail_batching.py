"""Rank-local batches retain every complete GRPO group, including the last short batch."""
import math

class TailSafeRankBatchSampler:

    def __init__(self, size, num_generations, world_size, rank, per_device_batch):
        self.size = size
        self.n = num_generations
        self.num_processes = world_size
        self.rank = rank
        self.batch_size = per_device_batch
        self.drop_last = False
        self.split_batches = False
        if size < 1 or num_generations < 2 or world_size < 1 or per_device_batch < 1 or not 0 <= rank < world_size:
            raise ValueError('Invalid dataset size, group size, world size, batch or rank')
        if size * self.n % world_size:
            raise ValueError('Total trajectories must divide evenly across ranks; adjust selection size without dropping or duplicating sources')
        global_batch = world_size * per_device_batch
        if global_batch % self.n:
            raise ValueError('Global batch must contain whole generation groups')
        self.prompts_per_batch = global_batch // self.n

    def __len__(self):
        return math.ceil(self.size / self.prompts_per_batch)

    def __iter__(self):
        for start in range(0, self.size, self.prompts_per_batch):
            indices = [i for i in range(start, min(start + self.prompts_per_batch, self.size)) for _ in range(self.n)]
            width = len(indices) // self.num_processes
            yield indices[self.rank * width:(self.rank + 1) * width]
