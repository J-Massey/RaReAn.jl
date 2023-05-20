from pathlib import Path
import numpy as np

from lotusvis.flow_field import ReadIn
from lotusvis.assign_props import AssignProps



def fluid_snap(sim_dir):
    fsim = ReadIn(sim_dir, "fluid", 1024, ext="vti")
    fs = fsim.snaps(part=False, save=True, save_path='./data')
    fs = AssignProps(fs)
    return fs


if __name__ == "__main__":
    sim_dir = f"/home/masseyj/Workspace/thicc-swimmer/data/bumps/256"
    sim_dir = "/home/masseyj/Workspace/lotus_projects/mode-one-roughness/data/outer-scaling/12000/12-2d"


    fs = fluid_snap(sim_dir)
    # test(fs, interpolator)