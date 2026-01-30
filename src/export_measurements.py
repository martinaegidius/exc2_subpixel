import os 

import numpy as np 
from helpers import load_measurements, filter_measurements, save_measurements


if __name__=="__main__":
    in_dir = f"data/images/with_calib"
    METHOD = "lstsq"
    out_dir = f"data/measurements/{METHOD}"
    os.makedirs(out_dir,exist_ok=True)
    measurements = load_measurements(in_dir,METHOD)
    range_to_exclude = np.arange(start=730,stop=900) #a period where light was different 
    measurements = filter_measurements(measurements, range_to_exclude)
    save_measurements(measurements,out_dir)