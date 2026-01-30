import math 

from skimage.io import imread
import matplotlib.pyplot as plt 
import numpy as np 

def bmatrix(a):
    """Returns a LaTeX bmatrix

    :a: numpy array
    :returns: LaTeX bmatrix as a string
    """
    if len(a.shape) > 2:
        raise ValueError('bmatrix can at most display two dimensions')
    lines = str(a).replace('[', '').replace(']', '').splitlines()
    rv = [r'\begin{bmatrix}']
    rv += ['  ' + r'\text{m} \\ '.join(l.split()) + r'\text{m}\\' for l in lines]
    rv +=  [r'\end{bmatrix}']
    return '\n'.join(rv)


"""
Math for making exercise in first place 
"""

h_p, w_p = 8160, 6144 
print(f"resolution: {h_p*w_p/10**6}MP")

pitch = 1.2*10**(-6)
h_s, w_s = h_p*pitch, w_p*pitch
diagonal = math.sqrt(h_s**2+w_s**2)

print("Sensor h x w x diagonal: ", h_s,w_s,diagonal)

#calc physical equivalent focal length
full_frame_diagonal = 43.3 #mm 
crop_factor = full_frame_diagonal/diagonal
f = 24/crop_factor 

print(f"focal length: {f}m")


"""
End math 
"""

LATEX = True 
UNIT = "mm"
unit = {"mm_scale":1000, "cm_scale":100,"m_scale":1}
scale_factor = unit[f"{UNIT}_scale"]

#3.1: pixel-to-physical height conversion 
m_per_pix = 1.2*10**(-6)
pixel_height = 385/2 #half height because it is easier with book 
height_measured = m_per_pix * pixel_height #B
print(f"Equivalent height in mm: {height_measured*scale_factor:.6f} {UNIT}")

#3.2: image distance 
g = 2.9968 #m 
G = 0.290/2
b = g*height_measured/G

print(f"Image formation distance: {b*scale_factor:.6f} {UNIT}")


#3.3: CCD chip size 
res_H, res_W = 3840, 2160 
ccd_size = (res_H*m_per_pix,res_W*m_per_pix)
print(f"Sensor size: {[str(f"{x*scale_factor} {UNIT}") for x in ccd_size]}")


#3.4: Horizontal FOV
f_h = 2*math.atan((ccd_size[-1]/2)/f)
print(f"Horizontal FOV: {f_h*180/math.pi:.6f} degrees")

#3.5: Vertical FOV 
f_v = 2*math.atan((ccd_size[0]/2)/f)
print(f"Vertical FOV: {f_v*180/math.pi:.6f} degrees")



#3.6: Credit card in pixels 
g = 2.9968 #m 
f = 6.8*10**(-3)
G = 53.98*10**(-3)
b = 1/(1/f - 1/g)
B = b*G/g
n_pixels = B/m_per_pix
n_pixels = B/m_per_pix
print(f"The height of the credit card on the sensor is {n_pixels}, and {math.ceil(n_pixels)} pixels ceiled.")


G = 86.50*10**(-3)
b = 1/(1/f - 1/g)
B = b*G/g
n_pixels = B/m_per_pix
print(f"The width of the credit card on the sensor is {n_pixels}, and {math.ceil(n_pixels)} pixels ceiled.")


#3.7: How large is the side length of the physical region in the object plane which a pixel records?  
pix_side_len_phys = g*m_per_pix/b
print(f"Side length of physical pixel in object plane: {pix_side_len_phys*scale_factor:.6f} {UNIT}")

#3.8: How large in pixels is the credit card in the recorded image?
#im = imread("data/images/with_calib/crops/0.png")
#plt.imshow(im, cmap="gray")
#plt.show()
H_pix, W_pix = 381-282, 473-315
print(f"Measured card width in pixels: {W_pix}")
print(f"Measured card height in pixels: {H_pix}")


#4.1: camera coordinates
H_c = 53.98*10**(-3)
W_c = 86.50*10**(-3)

camera_coords = np.array([[-W_c/2,0,g],[W_c/2,0,g],[0,-H_c/2,g],[0,H_c/2,g]]).T #(3 x N)
np. set_printoptions(precision=4)
print("Camera coords:")
print(camera_coords)

if LATEX:
    for i in range(camera_coords.shape[1]):
        print(f"{bmatrix(camera_coords[:,i])},\qquad ")

#4.2: sensor coordinates 
sensor_coords = np.zeros_like(camera_coords)
sensor_coords[0,:] = f*camera_coords[0,:]/g #x
sensor_coords[1,:] = f*camera_coords[1,:]/g #y
sensor_coords[2,:] = f #z
print("Physical sensor coords")
print(sensor_coords)
if LATEX:
    for i in range(sensor_coords.shape[1]):
        print(f"{bmatrix(sensor_coords[:,i])},\qquad ")

#4.3: height and width on sensor using triangulation equations

delta_x = sensor_coords[0,1]-sensor_coords[0,0]
n_pix = delta_x/m_per_pix
print(f"Delta x: {delta_x*scale_factor:.6f} {UNIT}-> The width of the credit card on the sensor is {n_pix}, and {math.ceil(n_pix)} pixels ceiled.")


delta_y = sensor_coords[1,-1]-sensor_coords[1,-2]
n_pix = delta_y/m_per_pix
print(f"Delta y: {delta_y*scale_factor:.6f} {UNIT} -> The height of the credit card on the sensor is {n_pix}, and {math.ceil(n_pix)} pixels ceiled.")

#(Optional) 4.5: Recompute 3.6 using z = f
g = 2.9968 #m 
f = 6.8*10**(-3)
G = 53.98*10**(-3)
b = f#1/(1/f - 1/g)
B = b*G/g
n_pixels = B/m_per_pix
n_pixels = B/m_per_pix
print(f"The height of the credit card on the sensor is {n_pixels}, and {math.ceil(n_pixels)} pixels ceiled.")


G = 86.50*10**(-3)
b = f#1/(1/f - 1/g)
B = b*G/g
n_pixels = B/m_per_pix
print(f"The width of the credit card on the sensor is {n_pixels}, and {math.ceil(n_pixels)} pixels ceiled.")


#Exercise 5.1
def load_measurements(path, method="simple"):
    print(f"{path}/{method}/card_heights.csv")
    card_w = np.loadtxt(f"{path}/{method}/card_widths.csv",delimiter=",")
    paper_w = np.loadtxt(f"{path}/{method}/paper_widths.csv",delimiter=",")
    filenames = np.loadtxt(f"{path}/{method}/filenames.csv",delimiter=",",dtype=str)
    return {"card_w":card_w,"paper_w":paper_w, "filenames": filenames}

in_dir = f"data/measurements/"
METHOD="lstsq"
measurements = load_measurements(in_dir,METHOD)
np. set_printoptions(precision=6)
print(f"Example data of card widths: {measurements["card_w"]}")
#Exercise 5.2 
card_w = 85.60 #mm 
mm_per_pixel_w = (card_w/(measurements["card_w"].mean())) 

print(f"width mm/pixel resolution in object plane: {mm_per_pixel_w:.4f}")

#Exercise 5.3: use all temporal data for estimation 
w_phys = measurements["paper_w"].mean()*mm_per_pixel_w
print(f"Estimated width of paper: {w_phys:.4f}")

#Exercise 5.4: Make a loop and gather all values

widths = measurements["paper_w"]
Ns = np.arange(10, len(widths), 10)  # More points for smoother curve
# Calculate metrics correctly
width_means = []
width_sem = []  # Standard error of the mean
width_std_of_sample = []  # Sample standard deviation
#plot_path = in_dir+"/plots"
#os.makedirs(plot_path,exist_ok=True)

for n in Ns:
    # Average pixel measurements
    width_slice = widths[:n]
    mean_paper_width_pixels = widths[:n].mean()        
    # Apply calibration
    width_means.append(mean_paper_width_pixels * mm_per_pixel_w)
    width_std = (width_slice*mm_per_pixel_w).std(ddof=1) #ddof=1 because sample std
    width_std_of_sample.append(width_std)
    # Standard error of the mean = std / sqrt(n)
    width_sem.append(width_std / np.sqrt(n))
    
# Exercise 5.5 - plotting and scaling demonstration 
# Create plots
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# Top left: Standard error of the mean vs N (should decrease as 1/√N)
axes[0, 0].plot(Ns, width_sem, 'b-', marker='o', markersize=3, label='Width SEM')
axes[0, 0].set_xlabel('Number of samples (N)')
axes[0, 0].set_ylabel('Standard Error of Mean (mm)')
axes[0, 0].set_title('Standard Error vs N')
axes[0, 0].legend()
axes[0, 0].grid(True)

# Top right: Log-log plot of SEM (should have slope -0.5)
axes[0, 1].loglog(Ns, width_sem, 'b-', marker='o', markersize=3, label='Width SEM')
# Theoretical 1/√N line
theoretical = width_sem[0] * np.sqrt(Ns[0]) / np.sqrt(Ns)
axes[0, 1].loglog(Ns, theoretical, 'k--', linewidth=2, label='1/√N reference')

axes[0, 1].set_xlabel('Number of samples (N)')
axes[0, 1].set_ylabel('Standard Error (mm)')
axes[0, 1].set_title('SEM vs N (log-log, slope = -0.5)')
axes[0, 1].legend()
axes[0, 1].grid(True, which='both', alpha=0.3)

# Bottom left: Sample std (should be roughly constant)
axes[1, 0].plot(Ns, width_std_of_sample, 'b-', marker='o', markersize=3, label='Width std')
axes[1, 0].set_xlabel('Number of samples (N)')
axes[1, 0].set_ylabel('Sample Std Dev (mm)')
axes[1, 0].set_title('Sample Std Dev vs N')
axes[1, 0].legend()
axes[1, 0].grid(True)

# Bottom right: Mean convergence
axes[1, 1].plot(Ns, width_means, 'b-', marker='o', markersize=3, label='Width mean')
#add 95% confint
z_score = 1.96
# Convert lists to numpy arrays for vector math
w_means_arr = np.array(width_means)
w_sem_arr = np.array(width_sem)
# Width Plot
axes[1, 1].plot(Ns, w_means_arr, 'b-', marker='o', markersize=3, label='Width Mean')
# Shade the 95% Confidence Interval (± 1.96 * SEM)
axes[1, 1].fill_between(Ns, 
                        w_means_arr - z_score * w_sem_arr, 
                        w_means_arr + z_score * w_sem_arr, 
                        color='b', alpha=0.15, label='95% CI')
axes[1, 1].set_xlabel('Number of samples (N)')
axes[1, 1].set_ylabel('Mean estimate (mm)')
axes[1, 1].set_title('Mean Convergence')
axes[1,1].ticklabel_format(useOffset=False)
axes[1, 1].legend()
axes[1, 1].grid(True)
for ax in axes.flatten():
    ax.set_xlim([Ns[0],Ns[-1]])

plt.tight_layout()
plt.show()

# Summary
print(f"\nIndividual measurement noise:")
w_phys = widths * mm_per_pixel_w
print(f"Final width estimate:  {w_phys.mean():.3f} ± {w_phys.std(ddof=1)/np.sqrt(len(w_phys)):.3f} mm")

