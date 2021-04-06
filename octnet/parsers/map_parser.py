from models.map import Map

import numpy as np
import os.path as path

def parse(file):
    if (not path.exists(file) or not path.isfile(file)): 
        return

    map = []
    with open(file, "r") as f:
        for line in f:
            row = [int(x) for x in line if x.isdigit()]
            map.append(row)

    return Map(map)
