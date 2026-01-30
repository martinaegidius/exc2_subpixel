import os
import json 
from glob import glob 
import numpy as np 
import cv2
import matplotlib.pyplot as plt 

def find_card(card_mask):
    x_min, x_max = 310, 480 
    y_min, y_max = 270, 380
    idx = np.argwhere(card_mask==1)
    idx_row, idx_col = idx[:,0], idx[:,1]
    keep_mask = (idx_row<y_max) & (idx_col<x_max) & (idx_row>y_min) & (idx_col>x_min)
    card_mask = np.zeros_like(card_mask)
    idx_to_keep = idx[keep_mask]
    card_mask[idx_to_keep[:,0],idx_to_keep[:,1]] = 1
    return card_mask 


def subpixel_edges_per_row(mask):
    """
    Returns subpixel left and right edge x-coordinates for each valid row.
    """
    h, w = mask.shape
    xs = np.arange(w)

    left_edges = []
    right_edges = []

    for y in range(h):
        row = mask[y, :]

        if row.sum() < 3:
            continue  # no detection on this row

        # Left edge: use only the left half of the object
        left_part = row * (xs < xs[row > 0].mean())
        if left_part.sum() > 0:
            x_left = np.sum(xs * left_part) / np.sum(left_part)
            left_edges.append(x_left)

        # Right edge: use only the right half
        right_part = row * (xs > xs[row > 0].mean())
        if right_part.sum() > 0:
            x_right = np.sum(xs * right_part) / np.sum(right_part)
            right_edges.append(x_right)

    return np.array(left_edges), np.array(right_edges)


def measure_mask(mask, stat = "median", method="simple"):
    if method=="simple":
        col_start = np.argmax(mask,axis=1)
        col_end = mask.shape[1]-np.argmax(mask[:,::-1],axis=1) #find the right most index
        row_start = np.argmax(mask,axis=0)
        row_end = mask.shape[0]-np.argmax(mask[::-1,:],axis=0) #find the lower index
        width_px = col_end-col_start
        width_px = width_px[width_px<mask.shape[1]] #filter out non-detections
 
        height_px = row_end-row_start
        height_px = height_px[height_px<mask.shape[0]] #filter out non-detections
    
        #find corner coords 
        col_start = col_start[col_start>0]
        col_end = col_end[col_end<mask.shape[1]]
        row_start = row_start[row_start>0]
        row_end = row_end[row_end<mask.shape[1]]
        #also return corner coordinates, fmt lower left, upper right (x,y)
        corner = [col_start.min(),row_end.max(),col_end.max(),row_start.min()] #possibly need to sort and statistic for better robustness; seems to be unstable
        if stat=="median":
            width_px, height_px = np.median(width_px), np.median(height_px)
        elif stat=="mean":
            width_px, height_px = np.mean(width_px), np.mean(height_px)        
        return width_px, height_px, corner, None
    

    elif method=="rectangle":
        # Find contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), 
                                        cv2.RETR_EXTERNAL, 
                                        cv2.CHAIN_APPROX_SIMPLE)
        MIN_AREA = 100
        
        if len(contours) == 0:
            return None, None, None, None
    
        # Filter contours by area
        valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]
        
        if len(valid_contours) == 0:
            return None, None, None, None
        
        # Get largest valid contour
        largest_contour = max(valid_contours, key=cv2.contourArea)
        
        rect = cv2.minAreaRect(largest_contour)
        box = cv2.boxPoints(rect)
        box = np.intp(box)
        
        width, height = rect[1] #also angle accessible using rect[-1]
        if width < height:
            width, height = height, width
        
        # Format corners for your existing convention
        # box contains 4 corners, typically in order: bottom-left, top-left, top-right, bottom-right
        # but order can vary, so let's extract min/max explicitly
        x_coords = box[:, 0]
        y_coords = box[:, 1]
        
        corner = [x_coords.min(), y_coords.max(), x_coords.max(), y_coords.min()]
        # Format: [left, bottom, right, top] to match your existing convention
        return width, height, corner, box
    
    elif method=="PCA":
        # Get all edge coordinates
        y_coords, x_coords = np.where(mask)
        points = np.column_stack([x_coords, y_coords])
        # Center the data
        centroid = points.mean(axis=0)
        centered = points - centroid
        
        # PCA to find principal axes
        cov = np.cov(centered.T)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Project onto principal axes
        projections = centered @ eigenvectors
        
        # Measure extent along each axis
        width = projections[:, 0].max() - projections[:, 0].min()
        height = projections[:, 1].max() - projections[:, 1].min()
        
        return width, height, None, None
    
def visualize_detection(image, box):
    vis = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if len(image.shape) == 2 else image.copy()
    # Convert box to integer coordinates for drawing
    box_int = box.astype(np.int32)
    
    cv2.drawContours(vis, [box_int], 0, (0, 255, 0), 2)
    
    # Draw corner points
    for i, point in enumerate(box_int):
        pt = tuple(point)
        cv2.circle(vis, pt, 5, (255, 0, 0), -1)
        cv2.putText(vis, str(i), pt, cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, (255, 255, 255), 1)
    
    return vis

def draw_mask_measurements(im_out, corner, width_px, height_px):
    """
    Draw corners and lines showing measured width and height on the image.
    - im_out: RGB/BGR image as uint8
    - corner: [x_min, y_min, x_max, y_max] in image coordinates
    - width_px, height_px: measurements from measure_card function
    """

    x_min, y_max, x_max, y_min = corner

    colors = [(255,0,255) for _ in range(3)]
    # Draw corner points (red)
    corner_color = colors[0]
    radius = 5
    thickness = -1
    corners = [
        (x_min, y_max),  # lower-left
        (x_max, y_max),  # lower-right
        (x_min, y_min),  # upper-left
        (x_max, y_min),  # upper-right
    ]
    for c in corners:
        cv2.circle(im_out, c, radius, corner_color, thickness)

    # Draw measured width line from upper-left corner
    width_line_color = colors[1]  
    start_point = (x_min, y_min)
    end_point = (x_min + int(width_px), y_min)  # horizontal
    cv2.line(im_out, start_point, end_point, width_line_color, 2)

    # Draw measured height line from upper-left corner
    height_line_color = colors[-1]  
    end_point = (x_min, y_min + int(height_px))  # vertical
    cv2.line(im_out, (x_min, y_min), end_point, height_line_color, 2)

    # Annotate width and height
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.5
    font_color = colors[1]
    thickness_text = 1
    cv2.putText(im_out, f"W: {int(width_px)} px", (x_min, y_min - 10), font, font_scale, font_color, thickness_text)
    cv2.putText(im_out, f"H: {int(height_px)} px", (x_min - 50, y_min + int(height_px)//2), font, font_scale, font_color, thickness_text)

    return im_out

if __name__=="__main__":
    with open("configs/thresholds.json") as f:
        thresholds = json.load(f)
    
    in_folder = "data/images/with_calib/crops"
    METHOD = "rectangle"
    output_dir_all = in_folder.rsplit("/",1)[0]+f"/tuned_threshold_masks_card/{METHOD}"
    output_dir_paper = in_folder.rsplit("/",1)[0]+f"/tuned_threshold_masks_paper/{METHOD}"
    output_dir_stats = in_folder.rsplit("/",1)[0]+f"/measurements/{METHOD}"
    os.makedirs(output_dir_all,exist_ok=True)
    os.makedirs(output_dir_paper,exist_ok=True)
    os.makedirs(output_dir_stats,exist_ok=True)
    
    all_files = sorted(glob(f"{in_folder}/*.png")) #data has already been proc with burn in 250 and burn out 50; only need to filter bad frames 
    NSAMPLES = len(all_files)
    
    c_widths, c_heights = [], []
    p_widths, p_heights, p_corners = [], [], []

    
    filenames = [] 
    for idx in range(NSAMPLES):
        im_str = f"{in_folder}/{idx}.png"
        if(idx%50==0):
            print(f"{idx}/{len(all_files)}")
        im = cv2.imread(im_str, cv2.IMREAD_GRAYSCALE)

        #detect card
        paper_mask = im>thresholds["card_to_paper"]
        card_mask = (thresholds["background_to_card"]<=im) & (im<thresholds["card_to_paper"])
        card_mask = find_card(card_mask)
        #find widths of card per frame 
        c_width, c_height, c_corner, box = measure_mask(card_mask, stat="median", method=METHOD)
        if c_width is None or c_width < 10:  # Noise detection
            print(f"Skipping frame {idx}: invalid detection")
            # Don't append to measurement arrays
            continue
        c_widths.append(float(c_width))
        c_heights.append(float(c_height))
        filenames.append(im_str)
        background_mask = im<thresholds["background_to_card"]

        im_out = np.zeros((im.shape[0],im.shape[1],3))
        im_out[background_mask,0] = 255 #R 
        im_out[card_mask,1] = 255 #G
        im_out[paper_mask,2] = 255 #B
        
        #add corners and draw width and height from upper left corner on frame 
        if isinstance(box,np.ndarray):
            im_out = visualize_detection(im,box)
        else:
            im_out = draw_mask_measurements(im_out,c_corner, c_width, c_height)
            

        cv2.imwrite(f"{output_dir_all}/{idx}.png", im_out)
        
        #plt.imshow(im_out)
        #plt.show()
        del box 
        
        #detect paper 
        im_out = np.zeros((im.shape[0],im.shape[1],3))
        paper_mask = im>=thresholds["card_to_paper"]
        background_mask = im<thresholds["card_to_paper"]
        im_out[background_mask,0] = 255 #R 
        im_out[paper_mask,1] = 255 #G
        p_width, p_height, p_corner, box = measure_mask(paper_mask, stat="mean",method=METHOD)
        if isinstance(box,np.ndarray):
            im_out = visualize_detection(im,box)
        else:
            im_out = draw_mask_measurements(im_out,c_corner, c_width, c_height)
        p_widths.append(float(p_width))
        p_heights.append(float(p_height))
        p_corners.append(np.array(p_corner))
        cv2.imwrite(f"{output_dir_paper}/{idx}.png", im_out)
        del box 

c_heights, c_widths = np.array(c_heights), np.array(c_widths)
p_heights, p_widths = np.array(p_heights), np.array(p_widths)
p_corners = np.stack(p_corners)

np.savetxt(f"{output_dir_stats}/card_heights.csv",c_heights)
np.savetxt(f"{output_dir_stats}/card_widths.csv",c_widths)
np.savetxt(f"{output_dir_stats}/paper_heights.csv",p_heights)
np.savetxt(f"{output_dir_stats}/paper_widths.csv",p_widths)
np.save(f"{output_dir_stats}/paper_corners.csv",p_corners)
np.savetxt(f"{output_dir_stats}/filenames.csv",filenames, delimiter=",", fmt='%s')
            






    
