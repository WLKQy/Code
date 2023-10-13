num_ue = 8
num_kx = 0
F = 7
Dd = 500
cgl = 8
import os
import time
import numpy as np
import datetime
import time

train_parameters = {
    "max_episode_len": 25000,
    "num_episodes": 20500,
}

model_parameters = {
    "lr": 2e-5,  # 1.0e-4
    "buffer_size": 10000,
    "sigma": 0.15,
    "gamma": 0.95,
    "batch_size": 1024,
    "num_episodes": 100000,
    "max_replay_buffer_len": 1024,
    "tau": 0.01,
    "epsilon": 0.99
}
