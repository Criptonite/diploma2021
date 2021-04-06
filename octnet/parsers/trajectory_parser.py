from models.trajectory import Trajectory

import numpy as np
import os.path as path

N = 3

def parse(file):
    if (not path.exists(file) or not path.isfile(file)): 
        return

    trajectories = []
    state = 0
    label = ""
    x_traj = []
    y_traj = []
    with open(file, "r") as f:
        for line in f:
            if state % N == 0:
                x_traj = []
                y_traj = []
                label = line
            elif state % N == 1:
                x_traj = [float(x) for x in line.split(" ") if x.strip()]
            else:
                y_traj = [float(x) for x in line.split(" ") if x.strip()]
                trajectories.append(Trajectory(label, x_traj, y_traj))
            state += 1

    return trajectories