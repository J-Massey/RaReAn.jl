from pathlib import Path
import numpy as np
import os
from tkinter import Tcl

from lotusvis.flow_field import ReadIn
from lotusvis.assign_props import AssignProps



def fluid_snap(sim_dir):
    fsim = ReadIn(sim_dir, fluid_ext, 4096, ext="vti")
    vort = fsim.vort_low_memory_saver(save_path='./data')
    bsim = ReadIn(sim_dir, body_ext, 4096, ext="vti")
    sdf = bsim.save_sdf_low_memory(save_path='./data')
    # List files in data directory

    # bmask = np.where(sdf <= 1, False, True)
    # vort = np.where(bmask, vort, 0)
    # np.save(f"data/{save_ext}.npy", vort)
    return sdf


def bmask():
     fnsf, fnsb = fns()
     assert len(fnsf) == len(fnsb)
     for idx, (fnf, fnb) in enumerate(zip(fnsf, fnsb)):
        print(fnf, fnb)
        f = np.load(os.path.join("./data", fnf))
        b = np.load(os.path.join("./data", fnb))
        bmask = np.where(b <= 1, False, True)
        vort = np.where(bmask, f, 0)
        np.save(os.path.join("./data", f"{save_ext}_{idx}"), vort)
            

def fns():
        fnsf = [fn for fn in os.listdir("./data")
               if fn.startswith(fluid_ext) and fn.endswith(f'.npy')]
        fnsf = Tcl().call('lsort', '-dict', fnsf)
        fnsb = [fn for fn in os.listdir("./data")
               if fn.startswith(body_ext) and fn.endswith(f'.npy')]
        fnsb = Tcl().call('lsort', '-dict', fnsb)
        return fnsf, fnsb

if __name__ == "__main__":
    sim_dir = f"/home/masseyj/Workspace/thicc-swimmer/data/resolvant-data"
    fluid_ext = "fluid"
    body_ext = "bodyF"
    save_ext = "smooth"
    # fs = fluid_snap(sim_dir)
    bmask()
    # sim_dir = f"./4096"
    # fluid_ext = "fluid"
    # body_ext = "fluid"
    # save_ext = "smooth"
    # fs = fluid_snap(sim_dir)