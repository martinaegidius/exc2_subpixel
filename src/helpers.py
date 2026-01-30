import numpy as np 
import matplotlib.pyplot as plt 


def load_measurements(path, method="simple"):
    path += "/measurements"
    
    if method!="lstsq": 
        card_h = np.loadtxt(f"{path}/{method}/card_heights.csv",delimiter=",")
        card_w = np.loadtxt(f"{path}/{method}/card_widths.csv",delimiter=",")
        paper_h = np.loadtxt(f"{path}/{method}/paper_heights.csv",delimiter=",")
        paper_w = np.loadtxt(f"{path}/{method}/paper_widths.csv",delimiter=",")
        paper_corners = np.load(f"{path}/{method}/paper_corners.npy")
        filenames = np.loadtxt(f"{path}/{method}/filenames.csv",delimiter=",",dtype=str)        
    else:
        print("Loading lstsq paper measurements. Using simple card measurements!")
        simple_path = "data/images/with_calib/measurements/simple"
        card_h = np.loadtxt(f"{simple_path}/card_heights.csv",delimiter=",")
        card_w = np.loadtxt(f"{simple_path}/card_widths.csv",delimiter=",")
        paper_corners = None #not estimated, but not used either #np.load(f"{path}/{method}/paper_corners.npy")
        paper_h = np.loadtxt(f"{path}/{method}/paper_heights.csv",delimiter=",")
        paper_w = np.loadtxt(f"{path}/{method}/paper_widths.csv",delimiter=",")
        filenames = np.loadtxt(f"{path}/{method}/filenames.csv",delimiter=",",dtype=str)        
    
    return {"card_h":card_h,"card_w":card_w,"paper_h":paper_h,"paper_w":paper_w, "paper_c": paper_corners, "filenames": filenames}


def save_measurements(measurements,path):
        outname_dict = {"card_h": "card_heights", "card_w": "card_widths", "paper_w": "paper_widths", "paper_h":"paper_heights", "filenames":"filenames"}
        for key, value in measurements.items():
            print(key)
            if key not in outname_dict:
                continue
            file_path = f"{path}/{outname_dict[key]}.csv"
            if key=="filenames":
                np.savetxt(file_path,value,fmt='%s', delimiter=",")        
            else:
                np.savetxt(file_path, value, delimiter=",")
        print("Saved measurements to: ",path)

def filter_corners(measurements: dict) -> np.array: 
    corners = measurements["paper_c"]
    idx_remove = np.argwhere(corners[:,0]<200).squeeze()
    print(f"Example files removed: {measurements["filenames"][idx_remove][:10]}")
    return idx_remove

def plot_corners(measurements):
    plt.plot(measurements["paper_c"][:,0],label="0")
    plt.plot(measurements["paper_c"][:,1],label="1")
    plt.plot(measurements["paper_c"][:,2],label="2")
    plt.plot(measurements["paper_c"][:,3],label="3")
    plt.legend()
    plt.show()

def filter_measurements(measurements: dict, idx: np.array) -> dict:
    print(f"removing {len(idx)} samples from current {len(measurements["card_w"])} samples")
    measurements["card_w"] = np.array([x for i, x in enumerate(measurements["card_w"]) if i not in idx])
    measurements["card_h"] = np.array([x for i, x in enumerate(measurements["card_h"]) if i not in idx])
    measurements["paper_w"] = np.array([x for i, x in enumerate(measurements["paper_w"]) if i not in idx])
    measurements["paper_h"] = np.array([x for i, x in enumerate(measurements["paper_h"]) if i not in idx])
    if measurements["paper_c"] is not None:
        measurements["paper_c"] = np.array([x for i, x in enumerate(measurements["paper_c"]) if i not in idx])
    measurements["filenames"] = np.array([x for i, x in enumerate(measurements["filenames"]) if i not in idx])
    
    print(f"remaining samples: N={len(measurements["card_w"])} ")
    return measurements 
    
    


if __name__=="__main__":
    in_dir = f"data/images/with_calib"
    METHOD="rectangle"
    measurements = load_measurements(in_dir,METHOD)
    
    #plot_corners(measurements)
    range_to_exclude = np.arange(start=730,stop=900) #a period where light was different 
    measurements = filter_measurements(measurements, range_to_exclude)
    plot_corners(measurements)
    #idx = filter_corners(measurements)
    #measurements = filter_measurements(measurements, idx)
    #plot_corners(measurements)
    
    