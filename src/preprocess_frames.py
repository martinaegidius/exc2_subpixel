import cv2
import matplotlib.pyplot as plt 
import numpy as np 

video_file = "data/videos/seventee_step_0f_calib.mp4"
RESULT_OUTPUT = "data/images/with_calib"	

cap = cv2.VideoCapture(video_file) #727 frames
# Check if the video was opened successfully
if not cap.isOpened():
    print("Error: Could not open video file.")
else:
    print("Video file opened successfully!")
 
# Read the first frame to confirm reading
ret, frame = cap.read()
 
if not ret:
    print("Error: Could not read the frame.")

N_frames = 0
BURN_IN = 250
BURN_OUT = 50


widths = []
heights = []
im_middle = (frame.shape[0]//2,frame.shape[1]//2)
crop_w = crop_h = 1000
heights = []
widths = []
i = 0
while ret is not None:
    ret, frame = cap.read()
    if not ret: 
        break
    
    else:
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        frame_cropped = frame_gray[im_middle[0]-crop_h//2:im_middle[0]+crop_h//2,im_middle[1]-crop_w//2:im_middle[1]+crop_w//2]
        cv2.imwrite(f"{RESULT_OUTPUT}/crops/{i}.png",frame_cropped)
        T = 235
        ims_mask = frame_cropped>T
        print(frame_cropped.max(),frame_cropped.min())
        print(type(frame_cropped))
        print(frame_cropped.dtype)
        print(frame_cropped.max(),frame_cropped.min())
        print(ims_mask.max())

        frame_crop_test = cv2.imread(f"{RESULT_OUTPUT}/crops/{i}.png", cv2.IMREAD_GRAYSCALE)
        print(np.linalg.norm(frame_cropped-frame_crop_test,np.inf))
        
        break
        cv2.imwrite(f"{RESULT_OUTPUT}/masks/{i}.png",ims_mask*255)
        im2 = cv2.imread(f"{RESULT_OUTPUT}/masks/{i}.png")
        
        col_start = np.argmax(ims_mask,axis=1)
        col_end = ims_mask.shape[1]-np.argmax(ims_mask[:,::-1],axis=1)
        row_start = np.argmax(ims_mask,axis=0)
        row_end = ims_mask.shape[0]-np.argmax(ims_mask[::-1,:],axis=0)
        width_px = (col_end-col_start).mean()
        height_px = (row_end-row_start).mean()
        widths.append(float(width_px))
        heights.append(float(height_px))
        
        if i%10==0:
            frame_rgb = cv2.cvtColor(frame_cropped, cv2.COLOR_GRAY2BGR)

            # Find and draw contours
            contours, _ = cv2.findContours(ims_mask.astype(np.uint8), 
                                            cv2.RETR_EXTERNAL, 
                                            cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(frame_rgb, contours, -1, (0, 0, 255), 2)  # Red outline

            # Draw mean edge lines
            cv2.line(frame_rgb, (int(col_start.mean()), 0), 
                    (int(col_start.mean()), frame_rgb.shape[0]), (0, 255, 255), 1)  # Cyan
            cv2.line(frame_rgb, (int(col_end.mean()), 0), 
                    (int(col_end.mean()), frame_rgb.shape[0]), (0, 255, 255), 1)
            cv2.line(frame_rgb, (0, int(row_start.mean())), 
                    (frame_rgb.shape[1], int(row_start.mean())), (255, 255, 0), 1)  # Yellow
            cv2.line(frame_rgb, (0, int(row_end.mean())), 
                    (frame_rgb.shape[1], int(row_end.mean())), (255, 255, 0), 1)

            # Add text
            cv2.putText(frame_rgb, f'W={width_px:.2f} H={height_px:.2f}', 
                        (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imwrite(f"{RESULT_OUTPUT}/threshold_lines/{i}.png", frame_rgb)
        i+=1

cap.release()



np.savetxt(f"{RESULT_OUTPUT}/widths.txt", np.array(widths))
np.savetxt(f"{RESULT_OUTPUT}/heights.txt", np.array(heights))

BURN_OUT_IDX = len(widths)-BURN_OUT

# Plot time series of measurements
fig, axes = plt.subplots(2, 1, figsize=(12, 6))

axes[0].plot(widths, 'b-', alpha=0.7)
axes[0].set_ylabel('Width (pixels)')
axes[0].set_title('Width measurements over time')
axes[0].grid(True)
axes[0].axvline(BURN_IN, color='r', linestyle='--', label=f'N={BURN_IN} point')
axes[0].axvline(BURN_OUT_IDX, color='r', linestyle='--', label=f'N={BURN_OUT_IDX} point')

axes[1].plot(heights, 'r-', alpha=0.7)
axes[1].set_ylabel('Height (pixels)')
axes[1].set_xlabel('Frame number')
axes[1].set_title('Height measurements over time')
axes[1].grid(True)
axes[1].axvline(BURN_IN, color='r', linestyle='--', label=f'N={BURN_IN} point')
axes[1].axvline(BURN_OUT_IDX, color='r', linestyle='--', label=f'N={BURN_OUT_IDX} point')

plt.tight_layout()
plt.savefig(f'{RESULT_OUTPUT}/measurement_series.png')
plt.show()

# Check for outliers
print(f"Height outliers (>3 std from mean):")

widths = widths[BURN_IN:-BURN_OUT_IDX]
heights = heights[BURN_IN:-BURN_OUT_IDX]



height_mean = heights.mean()
height_std = heights.std()
outliers = np.where(np.abs(heights - height_mean) > 3 * height_std)[0]
print(f"Found {len(outliers)} outliers at frames: {outliers}")
if len(outliers) > 0:
    print(f"Outlier values: {heights[outliers]}")


Ns = np.arange(10, len(widths), 10)  # More points for smoother curve

# Calculate metrics correctly
width_means = []
height_means = []
width_sem = []  # Standard error of the mean
height_sem = []
width_std_of_sample = []  # Sample standard deviation
height_std_of_sample = []

for n in Ns:
    width_slice = widths[:n]
    height_slice = heights[:n]
    
    # Mean
    width_means.append(width_slice.mean())
    height_means.append(height_slice.mean())
    
    # Sample standard deviation (of individual measurements)
    width_std = width_slice.std(ddof=1)
    height_std = height_slice.std(ddof=1)
    width_std_of_sample.append(width_std)
    height_std_of_sample.append(height_std)
    
    # Standard error of the mean = std / sqrt(n)
    width_sem.append(width_std / np.sqrt(n))
    height_sem.append(height_std / np.sqrt(n))

# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top left: Standard error of the mean vs N (should decrease as 1/√N)
axes[0, 0].plot(Ns, width_sem, 'b-', marker='o', markersize=3, label='Width SEM')
axes[0, 0].plot(Ns, height_sem, 'r-', marker='o', markersize=3, label='Height SEM')
axes[0, 0].set_xlabel('Number of samples (N)')
axes[0, 0].set_ylabel('Standard Error of Mean (pixels)')
axes[0, 0].set_title('Standard Error vs N (should decrease)')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Top right: Log-log plot of SEM (should have slope -0.5)
axes[0, 1].loglog(Ns, width_sem, 'b-', marker='o', markersize=3, label='Width SEM')
axes[0, 1].loglog(Ns, height_sem, 'r-', marker='o', markersize=3, label='Height SEM')

# Theoretical 1/√N line
theoretical = width_sem[0] * np.sqrt(Ns[0]) / np.sqrt(Ns)
axes[0, 1].loglog(Ns, theoretical, 'k--', linewidth=2, label='1/√N reference')

axes[0, 1].set_xlabel('Number of samples (N)')
axes[0, 1].set_ylabel('Standard Error (pixels)')
axes[0, 1].set_title('SEM vs N (log-log, slope = -0.5)')
axes[0, 1].legend()
axes[0, 1].grid(True, which='both', alpha=0.3)

# Bottom left: Sample std (should be roughly constant)
axes[1, 0].plot(Ns, width_std_of_sample, 'b-', marker='o', markersize=3, label='Width std')
axes[1, 0].plot(Ns, height_std_of_sample, 'r-', marker='o', markersize=3, label='Height std')
axes[1, 0].set_xlabel('Number of samples (N)')
axes[1, 0].set_ylabel('Sample Std Dev (pixels)')
axes[1, 0].set_title('Sample Std Dev vs N (should stabilize)')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Bottom right: Mean convergence
axes[1, 1].plot(Ns, width_means, 'b-', marker='o', markersize=3, label='Width mean')
axes[1, 1].plot(Ns, height_means, 'r-', marker='o', markersize=3, label='Height mean')
axes[1, 1].axhline(widths.mean(), color='b', linestyle='--', alpha=0.5, label='Final width mean')
axes[1, 1].axhline(heights.mean(), color='r', linestyle='--', alpha=0.5, label='Final height mean')
axes[1, 1].set_xlabel('Number of samples (N)')
axes[1, 1].set_ylabel('Mean estimate (pixels)')
axes[1, 1].set_title('Mean Convergence')
axes[1, 1].legend()
axes[1, 1].grid(True)

plt.tight_layout()
plt.savefig(f'{RESULT_OUTPUT}/convergence_analysis.png', dpi=150)
plt.show()

# Summary
print(f"Final estimates (using all {len(widths)} frames):")
print(f"Width:  {widths.mean():.3f} ± {widths.std()/np.sqrt(len(widths)):.3f} pixels")
print(f"Height: {heights.mean():.3f} ± {heights.std()/np.sqrt(len(heights)):.3f} pixels")
print(f"\nIndividual measurement noise:")
print(f"Width std:  {widths.std():.3f} pixels")
print(f"Height std: {heights.std():.3f} pixels")
