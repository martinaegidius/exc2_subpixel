import os

import numpy as np 
import matplotlib.pyplot as plt 
from helpers import load_measurements, filter_corners, filter_measurements


if __name__=="__main__":
    INCLUDE_HEIGHT = False 
    REMOVE_CHANGE_PERIOD = True 
    in_dir = f"data/images/with_calib"
    METHOD="lstsq" #simple, rectangle or lstsq
    card_dim = 53.98, 85.60 #mm 
    paper_size = 297, 210 
    measurements = load_measurements(in_dir,METHOD)
    range_to_exclude = np.arange(start=730,stop=900) #a period where light was different 
    
    #apply to paper dimension as a function of N
    widths = measurements["paper_w"]
    print(np.corrcoef(widths,measurements["card_w"]))
    heights = measurements["paper_h"]
    print("NSAMPLES BEFORE REMOVE: ", len(heights))
    
    
    if REMOVE_CHANGE_PERIOD:
        measurements = filter_measurements(measurements, range_to_exclude)
    widths = measurements["paper_w"]
    heights = measurements["paper_h"]
    
    print("NSAMPLES AFTER REMOVE: ", len(heights))
    
    
    #calculate the "physical" resolution of a pixel (mm/pixel) using our calibration rod
    mm_per_pixel_w = (card_dim[1]/(measurements["card_w"].mean())) 
    mm_per_pixel_h = (card_dim[0]/(measurements["card_h"].mean()))
    
    print(mm_per_pixel_w,mm_per_pixel_h)

    Ns = np.arange(100, len(widths), 10)  # More points for smoother curve

    # Calculate metrics correctly
    width_means = []
    height_means = []
    width_sem = []  # Standard error of the mean
    height_sem = []
    width_std_of_sample = []  # Sample standard deviation
    height_std_of_sample = []
    plot_path = in_dir+"/plots"
    os.makedirs(plot_path,exist_ok=True)

    for n in Ns:
        # Average pixel measurements
        width_slice = widths[:n]
        height_slice = heights[:n]

        mean_paper_width_pixels = widths[:n].mean()
        mean_paper_height_pixels = heights[:n].mean()
                
        # Apply calibration
        width_means.append(mean_paper_width_pixels * mm_per_pixel_w)
        height_means.append(mean_paper_height_pixels * mm_per_pixel_h)
        # Sample standard deviation (of individual measurements)
        width_std = (width_slice*mm_per_pixel_w).std(ddof=1)
        height_std = (height_slice*mm_per_pixel_h).std(ddof=1)
        width_std_of_sample.append(width_std)
        height_std_of_sample.append(height_std)
        
        # Standard error of the mean = std / sqrt(n)
        width_sem.append(width_std / np.sqrt(n))
        height_sem.append(height_std / np.sqrt(n))

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Top left: Standard error of the mean vs N (should decrease as 1/√N)
    axes[0, 0].plot(Ns, width_sem, 'b-', marker='o', markersize=3, label='Width SEM')
    if INCLUDE_HEIGHT:
        axes[0, 0].plot(Ns, height_sem, 'r-', marker='o', markersize=3, label='Height SEM')
    axes[0, 0].set_xlabel('Number of samples (N)')
    axes[0, 0].set_ylabel('Standard Error of Mean (mm)')
    axes[0, 0].set_title('Standard Error vs N (should decrease)')
    axes[0, 0].legend()
    axes[0, 0].grid(True)

    # Top right: Log-log plot of SEM (should have slope -0.5)
    axes[0, 1].loglog(Ns, width_sem, 'b-', marker='o', markersize=3, label='Width SEM')
    if INCLUDE_HEIGHT:
        axes[0, 1].loglog(Ns, height_sem, 'r-', marker='o', markersize=3, label='Height SEM')

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
    if INCLUDE_HEIGHT:
        axes[1, 0].plot(Ns, height_std_of_sample, 'r-', marker='o', markersize=3, label='Height std')
    axes[1, 0].set_xlabel('Number of samples (N)')
    axes[1, 0].set_ylabel('Sample Std Dev (mm)')
    axes[1, 0].set_title('Sample Std Dev vs N (should stabilize)')
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
    if INCLUDE_HEIGHT:
        axes[1, 1].plot(Ns, height_means, 'r-', marker='o', markersize=3, label='Height mean')
        axes[1, 1].axhline(paper_size[0], color='r', linestyle='--', alpha=0.5, label='GT H')
    axes[1, 1].set_xlabel('Number of samples (N)')
    axes[1, 1].set_ylabel('Mean estimate (mm)')
    axes[1, 1].set_title('Mean Convergence')
    axes[1,1].ticklabel_format(useOffset=False)
    axes[1, 1].legend()
    axes[1, 1].grid(True)
    for ax in axes.flatten():
        ax.set_xlim([Ns[0],Ns[-1]])

    plt.tight_layout()
    plt.savefig(f'{plot_path}/convergence_analysis.png', dpi=150)
    plt.show()

    # Summary
    print(f"Final pixel estimates (using all {len(widths)} frames except excluded range):")
    print(f"Width:  {widths.mean():.3f} ± {widths.std()/np.sqrt(len(widths)):.3f} mm")
    if INCLUDE_HEIGHT:
        print(f"Height: {heights.mean():.3f} ± {heights.std()/np.sqrt(len(heights)):.3f} mm")
    print(f"\nIndividual measurement noise:")
    print(f"Width std:  {widths.std():.3f} pixels")
    if INCLUDE_HEIGHT:
        print(f"Height std: {heights.std():.3f} pixels")

    print(f"Final physical estimates (using all {len(widths)} frames except excluded range):")
    
    w_phys = widths*mm_per_pixel_w
    
    h_phys = heights*mm_per_pixel_h
    print(f"Width:  {w_phys.mean():.3f} ± {w_phys.std(ddof=1)/np.sqrt(len(w_phys)):.3f} mm")
    if INCLUDE_HEIGHT:
        print(f"Height: {h_phys.mean():.3f} ± {h_phys.std(ddof=1)/np.sqrt(len(h_phys)):.3f} mm")




    # fig = plt.figure(figsize=(12, 10))
    # gs = fig.add_gridspec(3, 2, height_ratios=[1, 1, 1])
    # ax_sem      = fig.add_subplot(gs[0, 0])
    # ax_loglog   = fig.add_subplot(gs[0, 1])
    # ax_std      = fig.add_subplot(gs[1, 0])
    # if INCLUDE_HEIGHT:
    #     ax_height   = fig.add_subplot(gs[1, 1])
    # ax_width    = fig.add_subplot(gs[2, 1])
    # ax_sem.plot(Ns, width_sem, 'b-', marker='o', markersize=3, label='Width SEM')
    # if INCLUDE_HEIGHT:
    #     ax_sem.plot(Ns, height_sem, 'r-', marker='o', markersize=3, label='Height SEM')
    # ax_sem.set_xlabel('Number of samples (N)')
    # ax_sem.set_ylabel('Standard Error of Mean (mm)')
    # ax_sem.set_title('Standard Error vs N')
    # ax_sem.legend()
    # ax_sem.grid(True)
    # ax_loglog.loglog(Ns, width_sem, 'b-', marker='o', markersize=3, label='Width SEM')
    # if INCLUDE_HEIGHT:
    #     ax_loglog.loglog(Ns, height_sem, 'r-', marker='o', markersize=3, label='Height SEM')

    # theoretical = width_sem[0] * np.sqrt(Ns[0]) / np.sqrt(Ns)
    # ax_loglog.loglog(Ns, theoretical, 'k--', linewidth=2, label='1/√N reference')

    # ax_loglog.set_xlabel('Number of samples (N)')
    # ax_loglog.set_ylabel('Standard Error (mm)')
    # ax_loglog.set_title('SEM vs N (log-log)')
    # ax_loglog.legend()
    # ax_loglog.grid(True, which='both', alpha=0.3)
    # ax_std.plot(Ns, width_std_of_sample, 'b-', marker='o', markersize=3, label='Width std')
    # if INCLUDE_HEIGHT:
    #     ax_std.plot(Ns, height_std_of_sample, 'r-', marker='o', markersize=3, label='Height std')
    # ax_std.set_xlabel('Number of samples (N)')
    # ax_std.set_ylabel('Sample Std Dev (mm)')
    # ax_std.set_title('Sample Std Dev vs N')
    # ax_std.legend()
    # ax_std.grid(True)
    # tol = 2.0  # mm
    # if INCLUDE_HEIGHT:
    #     ax_height.plot(Ns, height_means, 'r-', marker='o', markersize=3, label='Height mean')

    #     ax_height.axhspan(
    #         paper_size[0] - tol,
    #         paper_size[0] + tol,
    #         color='r',
    #         alpha=0.2,
    #         label='A4 tolerance (±2 mm)'
    #     )

    #     ax_height.set_ylabel('Height (mm)')
    #     ax_height.set_title('Height Mean Convergence')
    #     ax_height.legend()
    #     ax_height.grid(True)
    # ax_width.plot(Ns, width_means, 'b-', marker='o', markersize=3, label='Width mean')

    # ax_width.axhspan(
    #     paper_size[1] - tol,
    #     paper_size[1] + tol,
    #     color='b',
    #     alpha=0.2,
    #     label='A4 tolerance (±2 mm)'
    # )

    # ax_width.set_xlabel('Number of samples (N)')
    # ax_width.set_ylabel('Width (mm)')
    # ax_width.set_title('Width Mean Convergence')
    # ax_width.legend()
    # ax_width.grid(True)
    # plt.tight_layout()
    # plt.show()



