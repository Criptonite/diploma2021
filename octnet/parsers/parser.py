import parsers.map_parser as m_parser
import parsers.trajectory_parser as t_parser
from models.data_unit import DataUnit

import os.path as path
from os import listdir

def parse(dir):
    if not path.exists(dir):  
        print("Dirrectory not found: ", dir)
        return
        
    if not path.isdir(dir):
        print("Path is not a dirrectory: ", dir)
        return

    print("parse: ", dir)
    data_units = []

    def is_map(filename):
        return filename[19:22] == "map"

    def prefix(filename):
        return filename[:19]

    def mutate_dict(dict_to_mutate, filename):
        if (is_map(filename)):
            dict_to_mutate["map"] = filename
        else:
            dict_to_mutate["traj"] = filename
        return dict_to_mutate

    def group_files_by_prefix(dir):
        files_dict = {}
        all_files = [f for f in listdir(dir) if path.isfile(path.join(dir,f))]
        for filename in all_files:
            if files_dict.get(prefix(filename)) == None:
                files_dict[prefix(filename)] = mutate_dict({}, filename)
            else:
                files_dict[prefix(filename)] = mutate_dict(files_dict.get(prefix(filename)), filename)
        return files_dict

    files = group_files_by_prefix(dir)
    for key, val in files.items():
        map = m_parser.parse(path.join(dir, val["map"]))
        traj = t_parser.parse(path.join(dir, val["traj"]))
        data_units.append(DataUnit(map, traj))
    
    return data_units
    