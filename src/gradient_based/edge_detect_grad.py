import os 
from glob import glob

from skimage.io import imread 
import matplotlib.pyplot as plt 
import numpy as np 
from scipy.stats import linregress



def debug():
    im = imread("data/images/with_calib/crops/700.png")
    fig, ax = plt.subplots(3,2)
    ax[1,1].imshow(im)

    im_f = im.astype(float)
    grad_y, grad_x = np.gradient(im_f) #im.shape 
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    ax[0,0].imshow(grad_x)
    ax[0,1].imshow(grad_y)
    ax[1,0].imshow(grad_mag)
    grad_x, grad_y = filter_grad(grad_x,paper_ROI), filter_grad(grad_y,paper_ROI) #remove stuff outside paper ROI to reduce noise 
    #ax[2,0].plot(grad_mag[500,:])
    #ax[2,0].imshow(grad_filtered) #checked, looks good
    ax[2,0].plot(grad_y[:,500])
    height_coords, width_coords = estimate_subpixel(grad_x,grad_y) # width format: [row_index, left_x, right_x]
    ax[2,1].imshow(im, cmap='gray')
    # We filter out rows that were zeroed out by the ROI
    valid_w = width_coords[width_coords[:, 1] > 0]
    valid_h = height_coords[height_coords[:, 1] > 0]
    ax[2,1].scatter(valid_w[:, 1], valid_w[:, 0], c='r', s=1, label='Left Subpixel Edge')
    ax[2,1].scatter(valid_w[:, 2], valid_w[:, 0], c='b', s=1, label='Right Subpixel Edge')
    ax[2,1].scatter(valid_h[:, 0], valid_h[:, 1], c='orange', s=1, label='Upper Subpixel edge')
    ax[2,1].scatter(valid_h[:, 0], valid_h[:, 2], c='m', s=1, label='Bottom Subpixel Edge')
    ax[2,1].set_title("Subpixel Edge Localization Overlay")
    plt.show()
    
    #additional median-basd noise filtering with tolerance
    tol = 10 
    raw_widths = width_coords[:, 2] - width_coords[:, 1]
    median_w = np.median(raw_widths[raw_widths > 0]) 
    refined_widths, mask_widths_clean = clean_measurements(width_coords, median_w,tolerance=tol)
    raw_heights = height_coords[:, 2] - height_coords[:, 1]
    median_h = np.median(raw_heights[raw_heights > 0]) 
    refined_heights, mask_heights_clean = clean_measurements(height_coords, median_h,tolerance=tol)

    #keep non-noisy
    width_coords = width_coords[mask_widths_clean]
    height_coords = height_coords[mask_heights_clean]


    plt.imshow(im, cmap="gray")
    plt.scatter(width_coords[:, 1], width_coords[:, 0], c='r', s=1, label='Left Subpixel Edge')
    plt.scatter(width_coords[:, 2], width_coords[:, 0], c='b', s=1, label='Right Subpixel Edge')
    plt.scatter(height_coords[:, 0], height_coords[:, 1], c='orange', s=1, label='Upper Subpixel edge')
    plt.scatter(height_coords[:, 0], height_coords[:, 2], c='m', s=1, label='Bottom Subpixel Edge')
    plt.title("Subpixel Edge Localization Overlay")
    plt.show()
    line_segments = get_regression(width_coords,height_coords) #regress lines per edge
    visualize_line_segments(line_segments, width_coords, height_coords, im) #just for debugging
    height, width = calculate_distances(line_segments,width_coords,height_coords) #point to line distance
    print(height,width)


def visualize_for_report():
    paper_ROI = [[190,277],[779,711]] #fmt (y,x upper left, lower right)
    im = imread("data/images/with_calib/crops/0.png")
    plt.imshow(im,cmap="gray")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("image_crop.png")
    plt.close()

    im_f = im.astype(float)
    grad_y, grad_x = np.gradient(im_f) #im.shape 
    grad_mag = np.sqrt(grad_x**2 + grad_y**2)
    
    plt.imshow(grad_x)
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("grad_x.png")
    plt.close()

    plt.imshow(grad_x)
    plt.axis("off")
    plt.tight_layout()
    plt.xlim([270,330])
    plt.ylim([475,510])
    plt.savefig("grad_x_close.png")
    plt.close()

    
    grad_x, grad_y = filter_grad(grad_x,paper_ROI), filter_grad(grad_y,paper_ROI) #remove stuff outside paper ROI to reduce noise 
    
    #ax[2,0].plot(grad_mag[500,:])
    #ax[2,0].imshow(grad_filtered) #checked, looks good
    
    plt.plot(grad_x[500,:])
    plt.savefig("grad_x_profile.png")
    plt.close()
    height_coords, width_coords = estimate_subpixel(grad_x,grad_y) # width format: [row_index, left_x, right_x]
    plt.imshow(im, cmap='gray')
    # We filter out rows that were zeroed out by the ROI
    valid_w = width_coords[width_coords[:, 1] > 0]
    valid_h = height_coords[height_coords[:, 1] > 0]
    plt.scatter(valid_w[:, 1], valid_w[:, 0], c='r', s=1, label='Left Subpixel Edge')
    plt.scatter(valid_w[:, 2], valid_w[:, 0], c='b', s=1, label='Right Subpixel Edge')
    #plt.scatter(valid_h[:, 0], valid_h[:, 1], c='orange', s=1, label='Upper Subpixel edge')
    #plt.scatter(valid_h[:, 0], valid_h[:, 2], c='m', s=1, label='Bottom Subpixel Edge')
    plt.title("Subpixel Edge Localization Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("subpixel_scatter.png")
    plt.close()
    


    

    #additional median-basd noise filtering with tolerance
    tol = 10 
    raw_widths = width_coords[:, 2] - width_coords[:, 1]
    median_w = np.median(raw_widths[raw_widths > 0]) 
    refined_widths, mask_widths_clean = clean_measurements(width_coords, median_w,tolerance=tol)
    raw_heights = height_coords[:, 2] - height_coords[:, 1]
    median_h = np.median(raw_heights[raw_heights > 0]) 
    refined_heights, mask_heights_clean = clean_measurements(height_coords, median_h,tolerance=tol)

    #keep non-noisy
    width_coords = width_coords[mask_widths_clean]
    height_coords = height_coords[mask_heights_clean]


    plt.imshow(im, cmap="gray")
    plt.scatter(width_coords[:, 1], width_coords[:, 0], c='r', s=1, label='Left Subpixel Edge')
    plt.scatter(width_coords[:, 2], width_coords[:, 0], c='b', s=1, label='Right Subpixel Edge')
    plt.scatter(height_coords[:, 0], height_coords[:, 1], c='orange', s=1, label='Upper Subpixel edge')
    plt.scatter(height_coords[:, 0], height_coords[:, 2], c='m', s=1, label='Bottom Subpixel Edge')
    plt.title("Subpixel Edge Localization Overlay")
    plt.axis("off")
    plt.tight_layout()
    plt.savefig("subpixel_scatter_clean.png")
    
    plt.xlim([270,330])
    plt.ylim([475,510])
    plt.savefig("scatter_clean_close.png")
    
    plt.close()

    
    line_segments = get_regression(width_coords,height_coords) #regress lines per edge
    visualize_line_segments(line_segments, width_coords, height_coords, im, out_name="regressed_lines.png", only_width=True) #just for debugging
    height, width = calculate_distances(line_segments,width_coords,height_coords) #point to line distance
    print(height,width)


def filter_grad(G, ROI):
    """
    Zeros out everyting outside of ROI crop. ROI format: list of lists in format y,x with upper left, lower right [[y_u_l,x_u_l],[y_b_r,x_b_r]]
    """
    G[:,:ROI[0][1]] = 0.0 #left 
    G[:ROI[0][0],:] = 0.0 #up
    G[ROI[1][0]:,:] = 0.0 #down
    G[:,ROI[1][1]:] = 0.0
    return G

def compute_centroid(grad_slice, peak_coord):
    if peak_coord>=grad_slice.shape[0]-1:
        indices = np.array([peak_coord-1,peak_coord])
    elif peak_coord==0:
        indices = np.array([peak_coord,peak_coord+1])
    else:
        indices = np.array([peak_coord-1,peak_coord,peak_coord+1])

    weights = np.abs(grad_slice[indices])
    #normalize
    centroid_coord = np.sum(indices*weights)/np.sum(weights)
    return centroid_coord

def estimate_subpixel(grad_x,grad_y):
    """
    Calculates a centroid-based subpixel edge coordinate 
    """
    width_coords = np.zeros((grad_x.shape[0],3))
    for row in range(grad_x.shape[1]):
        row_grad = grad_x[row,:]
        peak_left = np.argmax(row_grad)
        #compute centroid coordinates weighted by gradient 
        left_coord = compute_centroid(row_grad,peak_left)
        peak_right = np.argmin(row_grad)
        right_coord = compute_centroid(row_grad,peak_right)
        width_coords[row,0] = row 
        width_coords[row,1:] = np.array([left_coord,right_coord])
    

    height_coords = np.zeros((grad_y.shape[0],3))
    for col in range(grad_y.shape[0]):
        col_grad = grad_y[:,col]
        peak_up = np.argmax(col_grad)
        up_coord = compute_centroid(col_grad,peak_up)
        peak_bottom = np.argmin(col_grad)
        bottom_coord = compute_centroid(col_grad,peak_bottom)
        height_coords[col,0] = col
        height_coords[col,1:] = np.array([up_coord,bottom_coord])

    return height_coords, width_coords


def clean_measurements(width_coords, expected_width_px, tolerance=20):
    """
    Filter out detections which are too small / too large. Mainly removes text-detections within card 
    """
    widths = width_coords[:, 2] - width_coords[:, 1]
    # Filter: Keep only widths that are within 'tolerance' of the expected size
    # This kills the outliers where the argmax jumped to a floor tile grout line.
    mask = (widths > expected_width_px - tolerance) & (widths < expected_width_px + tolerance)
    clean_widths = widths[mask]
    return clean_widths, mask 


def get_regression(width_coords,height_coords):
    row_coords, col_coords = width_coords[:,0], height_coords[:,0]
    left_edge, right_edge = width_coords[:,1], width_coords[:,2]
    up_edge, bottom_edge = height_coords[:,1], height_coords[:,2]
    slope_l, intercept_l, _,_,_ = linregress(row_coords, left_edge)
    slope_r, intercept_r, _,_,_ = linregress(row_coords, right_edge)
    slope_u, intercept_u, _,_,_ = linregress(col_coords, up_edge)
    slope_d, intercept_d, _,_,_ = linregress(col_coords, bottom_edge)
    return {"left":[slope_l,intercept_l],"right":[slope_r,intercept_r], "up":[slope_u,intercept_u], "down":[slope_d,intercept_d]}

def visualize_line_segments(lines,width_coords, height_coords, im, out_name: str = None, only_width: bool = False):
    row_range = np.arange(width_coords[:,0].min(),width_coords[:,0].max())
    col_range = np.arange(height_coords[:,0].min(),height_coords[:,0].max())
    
    x_l = row_range*lines["left"][0]+lines["left"][1]
    x_r = row_range*lines["right"][0]+lines["right"][1]
    y_u = col_range*lines["up"][0]+lines["up"][1]
    y_d = col_range*lines["down"][0]+lines["down"][1]
    

    plt.imshow(im, cmap="gray")
    plt.scatter(x_l,row_range, s=1, c="r")
    plt.scatter(x_r,row_range, s=1, c="r")
    if not only_width:
        plt.scatter(col_range,y_u, s=1, c="magenta")
        plt.scatter(col_range,y_d, s=1, c="magenta")
    if out_name==None:
        plt.show()
    else:
        plt.savefig(out_name)
    return 

def calculate_distances(lines, width_coords, height_coords):
    """
    Calculate width and height by using shortest distance from 
    point in right to line in left and point in top to line in bottom

    Args:
        lines (_type_): _description_
        width_coords (_type_): format: [row number x left x x right x]
        height_coords (_type_): [column number x top y x bottom y]
    
    """
    x_right = width_coords[:,-1]
    y_right = width_coords[:,0]
    #form: Ax = By + C <-> 1x - By - C = 0
    A, B, C = 1, -lines["left"][0], -lines["left"][1]
    widths = np.abs(A*x_right+B*y_right+C)/np.sqrt(A**2+B**2)
    
    x_top = height_coords[:,0]
    y_top = height_coords[:,1]
    #form: By = Ax + C <-> By - Ax - C = 0
    A, B, C = -lines["down"][0], 1, -lines["down"][1]
    heights = np.abs(A*x_top+B*y_top+C)/np.sqrt(A**2+B**2)

    return heights.mean(), widths.mean() 


if __name__=="__main__":
    """
    See debug() for comments
    """
  
    
    in_folder = "data/images/with_calib/crops"
    METHOD = "lstsq"
    #output_dir_all = in_folder.rsplit("/",1)[0]+f"/tuned_threshold_masks_card/{METHOD}"
    output_dir_paper = in_folder.rsplit("/",1)[0]+f"/tuned_threshold_masks_paper/{METHOD}"
    output_dir_stats = in_folder.rsplit("/",1)[0]+f"/measurements/{METHOD}"
    #os.makedirs(output_dir_all,exist_ok=True)
    os.makedirs(output_dir_paper,exist_ok=True)
    os.makedirs(output_dir_stats,exist_ok=True)
    all_files = sorted(glob(f"{in_folder}/*.png")) #data has already been proc with burn in 250 and burn out 50; only need to filter bad frames 
    NSAMPLES = len(all_files)
    paper_ROI = [[190,277],[779,711]] #fmt (y,x upper left, lower right)
    
    p_widths, p_heights, p_corners = [], [], []
    #c_widths, c_heights = [], []
    filenames = [] 
    for idx in range(NSAMPLES):
        im_str = f"{in_folder}/{idx}.png"
        if(idx%50==0):
            print(f"{idx}/{len(all_files)}")
        im = imread(im_str)
        im_f = im.astype(float)
        grad_y, grad_x = np.gradient(im_f) 
        grad_mag = np.sqrt(grad_x**2 + grad_y**2)
        grad_x, grad_y = filter_grad(grad_x,paper_ROI), filter_grad(grad_y,paper_ROI)
        height_coords, width_coords = estimate_subpixel(grad_x,grad_y)
        tol = 10 
        raw_widths = width_coords[:, 2] - width_coords[:, 1]
        median_w = np.median(raw_widths[raw_widths > 0]) 
        refined_widths, mask_widths_clean = clean_measurements(width_coords, median_w,tolerance=tol)
        raw_heights = height_coords[:, 2] - height_coords[:, 1]
        median_h = np.median(raw_heights[raw_heights > 0]) 
        refined_heights, mask_heights_clean = clean_measurements(height_coords, median_h,tolerance=tol)
        width_coords = width_coords[mask_widths_clean]
        height_coords = height_coords[mask_heights_clean]
        line_segments = get_regression(width_coords,height_coords)
        visualize_line_segments(line_segments, width_coords, height_coords, im, out_name=f"{output_dir_paper}/{idx}.png")
        p_height, p_width = calculate_distances(line_segments,width_coords,height_coords)
        #c_widths.append(float(c_width)) #omitted for now
        #c_heights.append(float(c_height))
        filenames.append(im_str)
        p_widths.append(float(p_width))
        p_heights.append(float(p_height))
        #p_corners.append(np.array(p_corner)) #omitted for now
        
    #c_heights, c_widths = np.array(c_heights), np.array(c_widths)
    #p_corners = np.stack(p_corners)
    p_heights, p_widths = np.array(p_heights), np.array(p_widths)

    #np.savetxt(f"{output_dir_stats}/card_heights.csv",c_heights)
    #np.savetxt(f"{output_dir_stats}/card_widths.csv",c_widths)
    np.savetxt(f"{output_dir_stats}/paper_heights.csv",p_heights)
    np.savetxt(f"{output_dir_stats}/paper_widths.csv",p_widths)
    np.savetxt(f"{output_dir_stats}/filenames.csv",filenames, delimiter=",", fmt='%s')
    #np.save(f"{output_dir_stats}/paper_corners.csv",p_corners)
            




