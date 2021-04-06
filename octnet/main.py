# %matplotlib inline

from parsers import parser
from models.data_unit import DataUnit

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

data = parser.parse("/Users/gudmian/Documents/diploma2021/dataset")

def showRandomMapsAndTrajectories(num_of_maps, num_of_trajectories_for_map):
    if num_of_maps > 9: num_of_maps = 9
    
    counter = 1
    size = int(np.ceil(num_of_maps**0.5))
    fig = plt.figure(figsize=(size*3.25, size*3.25))
    
    for row in np.random.randint(400, size=(size, size)):
        for col in row:
            if counter > num_of_maps: continue
            
            ax = fig.add_subplot(size, size, counter)
            
            plt.imshow(np.abs(np.array(data[col].map.map, dtype='float32') - 1), cmap='gray')
            traj_size = 0
            if num_of_trajectories_for_map >= len(data[col].trajectories):
                traj_size = len(data[col].trajectories) - 1
            else:
                traj_size = num_of_trajectories_for_map
            for i in np.random.randint(len(data[col].trajectories), size=(traj_size)):
                trajs = data[col].trajectories[i]
                plt.plot(trajs.x, trajs.y)
            
            ax.set_title("Map #{}".format(col))
            
            counter += 1
    plt.show()

showRandomMapsAndTrajectories(5, 200)