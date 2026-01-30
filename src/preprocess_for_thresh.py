
import os 

import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
from skimage.io import imread 
from glob import glob 
import numpy as np 
from PIL import Image 





def plot(pixels_l_stats,pixels_r_stats,pixels_p_stats,pixels_c_stats, output_dir ):
    # Create figure with subplots
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('ROI Intensity Statistics Across Frames', fontsize=16)

    # Frame indices
    frames = np.arange(len(pixels_l_stats))

    # Define colors for consistency
    color_min = '#e74c3c'    # red
    color_mean = '#3498db'   # blue
    color_max = '#2ecc71'    # green

    # Plot Left Background Box
    ax = axes[0, 0]
    ax.fill_between(frames, pixels_l_stats[:, 0], pixels_l_stats[:, 2], 
                    alpha=0.2, color='gray', label='Min-Max Range')
    ax.plot(frames, pixels_l_stats[:, 0], color=color_min, linewidth=1, label='Min')
    ax.plot(frames, pixels_l_stats[:, 1], color=color_mean, linewidth=2, label='Mean')
    ax.plot(frames, pixels_l_stats[:, 2], color=color_max, linewidth=1, label='Max')
    ax.set_title('Left Background Box')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Intensity [0-255]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot Right Background Box
    ax = axes[0, 1]
    ax.fill_between(frames, pixels_r_stats[:, 0], pixels_r_stats[:, 2], 
                    alpha=0.2, color='gray', label='Min-Max Range')
    ax.plot(frames, pixels_r_stats[:, 0], color=color_min, linewidth=1, label='Min')
    ax.plot(frames, pixels_r_stats[:, 1], color=color_mean, linewidth=2, label='Mean')
    ax.plot(frames, pixels_r_stats[:, 2], color=color_max, linewidth=1, label='Max')
    ax.set_title('Right Background Box')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Intensity [0-255]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot Paper
    ax = axes[1, 0]
    ax.fill_between(frames, pixels_p_stats[:, 0], pixels_p_stats[:, 2], 
                    alpha=0.2, color='gray', label='Min-Max Range')
    ax.plot(frames, pixels_p_stats[:, 0], color=color_min, linewidth=1, label='Min')
    ax.plot(frames, pixels_p_stats[:, 1], color=color_mean, linewidth=2, label='Mean')
    ax.plot(frames, pixels_p_stats[:, 2], color=color_max, linewidth=1, label='Max')
    ax.set_title('Paper')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Intensity [0-255]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot Calibration Card
    ax = axes[1, 1]
    ax.fill_between(frames, pixels_c_stats[:, 0], pixels_c_stats[:, 2], 
                    alpha=0.2, color='gray', label='Min-Max Range')
    ax.plot(frames, pixels_c_stats[:, 0], color=color_min, linewidth=1, label='Min')
    ax.plot(frames, pixels_c_stats[:, 1], color=color_mean, linewidth=2, label='Mean')
    ax.plot(frames, pixels_c_stats[:, 2], color=color_max, linewidth=1, label='Max')
    ax.set_title('Calibration Card')
    ax.set_xlabel('Frame')
    ax.set_ylabel('Intensity [0-255]')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/roi_statistics_temporal.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Optional: Comparative plot showing mean ± std across all ROIs
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(frames, pixels_l_stats[:, 1], label='Left BG Mean', linewidth=2, alpha=0.8)
    ax.plot(frames, pixels_r_stats[:, 1], label='Right BG Mean', linewidth=2, alpha=0.8)
    ax.plot(frames, pixels_p_stats[:, 1], label='Paper Mean', linewidth=2, alpha=0.8)
    ax.plot(frames, pixels_c_stats[:, 1], label='Card Mean', linewidth=2, alpha=0.8)

    ax.set_title('Mean Intensities Comparison Across ROIs', fontsize=14)
    ax.set_xlabel('Frame', fontsize=12)
    ax.set_ylabel('Mean Intensity [0-255]', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f'{output_dir}/roi_means_comparison.png', dpi=150, bbox_inches='tight')
    plt.show()

    # Print summary statistics
    print("\n=== Summary Statistics Across All Frames ===")
    print(f"\nLeft Background Box:")
    print(f"  Global Min: {pixels_l_stats[:, 0].min():.1f}")
    print(f"  Global Max: {pixels_l_stats[:, 2].max():.1f}")
    print(f"  Mean of Means: {pixels_l_stats[:, 1].mean():.1f} ± {pixels_l_stats[:, 1].std():.1f}")

    print(f"\nRight Background Box:")
    print(f"  Global Min: {pixels_r_stats[:, 0].min():.1f}")
    print(f"  Global Max: {pixels_r_stats[:, 2].max():.1f}")
    print(f"  Mean of Means: {pixels_r_stats[:, 1].mean():.1f} ± {pixels_r_stats[:, 1].std():.1f}")

    print(f"\nPaper:")
    print(f"  Global Min: {pixels_p_stats[:, 0].min():.1f}")
    print(f"  Global Max: {pixels_p_stats[:, 2].max():.1f}")
    print(f"  Mean of Means: {pixels_p_stats[:, 1].mean():.1f} ± {pixels_p_stats[:, 1].std():.1f}")

    print(f"\nCalibration Card:")
    print(f"  Global Min: {pixels_c_stats[:, 0].min():.1f}")
    print(f"  Global Max: {pixels_c_stats[:, 2].max():.1f}")
    print(f"  Mean of Means: {pixels_c_stats[:, 1].mean():.1f} ± {pixels_c_stats[:, 1].std():.1f}")

    # calculate center point between classes and knowing that bg < card < paper
    paper_mean = pixels_p_stats[:, 1].mean()
    bg_mean = (pixels_l_stats[:, 1].mean() + pixels_r_stats[:, 1].mean()) / 2
    card_mean = pixels_c_stats[:, 1].mean()
    suggested_threshold_paper = (paper_mean + card_mean) / 2
    suggested_threshold_card = (bg_mean + card_mean) / 2
    

    print(f"\n=== Threshold Suggestion ===")
    print(f"Paper mean intensity: {paper_mean:.1f}")
    print(f"Background mean intensity: {bg_mean:.1f}")
    print(f"Suggested threshold (midpoint) card -> paper: {suggested_threshold_paper:.1f}")
    print(f"Suggested threshold (midpoint) bg -> card: {suggested_threshold_card:.1f}")



def print_threshold_suggestion(pixels_l,pixels_r,pixels_p,pixels_c):
    # calculate center point between classes and knowing that bg < card < paper
    paper_mean = pixels_p.mean()
    bg_mean = (pixels_l.mean() + pixels_r.mean()) / 2
    card_mean = pixels_c.mean()
    suggested_threshold_paper = (paper_mean + card_mean) / 2
    suggested_threshold_card = (bg_mean + card_mean) / 2
        

    print(f"\n=== Threshold Suggestion ===")
    print(f"Paper mean intensity: {paper_mean:.1f}")
    print(f"Background mean intensity: {bg_mean:.1f}")
    print(f"Suggested threshold (midpoint) card -> paper: {suggested_threshold_paper:.1f}")
    print(f"Suggested threshold (midpoint) bg -> card: {suggested_threshold_card:.1f}")

    return suggested_threshold_card, suggested_threshold_paper

def plot_histogram(pixels_l,pixels_r,pixels_p,pixels_c,theta_card, theta_paper, output_dir):
    all_pixels = np.concatenate(
        [pixels_l, pixels_r, pixels_p, pixels_c]
    )
    bins = np.linspace(all_pixels.min(), all_pixels.max(), 100)

    plt.hist(pixels_l, bins=bins, density=True,
            color="red", alpha=0.8, label="Left box")
    plt.hist(pixels_r, bins=bins, density=True,
            color="orange", alpha=0.8, label="Right box")
    plt.hist(pixels_p, bins=bins, density=True,
            color="blue", alpha=0.8, label="Paper")
    plt.hist(pixels_c, bins=bins, density=True,
            color="magenta", alpha=0.8, label="Card")
    ymax_theta = 0.43
    plt.vlines(x=theta_card,ymin=0,ymax=0.43,linestyles="--",colors="black")
    plt.vlines(x=theta_paper,ymin=0,ymax=0.43,linestyles="--",colors="black")
    
    plt.xlim([all_pixels.min(),255])
    plt.ylim([0,ymax_theta])
    plt.legend()
    plt.savefig(f"{output_dir}/hists.pdf",dpi=300)
    plt.show()



def main():
    #let's load some frames 
    in_folder = "data/images/with_calib/crops"
    output_dir = f"{in_folder.rsplit("/",1)[0]}/measurements/pixel_ROI"
    os.makedirs(output_dir,exist_ok=True)

    NORMALIZE = False 
    all_files = glob(f"{in_folder}/*.png") #data has already been proc with burn in 250 and burn out 50; only need to filter bad frames 
    NSAMPLE = len(all_files)
    all_idx = np.arange(len(all_files))
    range_to_exclude = np.arange(start=730,stop=900)

    if NORMALIZE:
        img_out_folder = "data/images/with_calib/processed_crops"
        os.makedirs(img_out_folder,exist_ok=True)
    
    print(f"Checking {output_dir} for precomputed stats")
    if os.path.exists(f"{output_dir}/card_stats.npy") and os.path.exists(f"{output_dir}/card.npy"): 
        print(f"Loading stats from {output_dir}")
        pixels_l_stats = np.load(f"{output_dir}/left_box_stats.npy")
        pixels_r_stats = np.load(f"{output_dir}/right_box_stats.npy")
        pixels_p_stats = np.load(f"{output_dir}/paper_stats.npy")
        pixels_c_stats = np.load(f"{output_dir}/card_stats.npy")
        
        pixels_l = np.load(f"{output_dir}/left_box.npy")
        pixels_r = np.load(f"{output_dir}/right_box.npy")
        pixels_p = np.load(f"{output_dir}/paper.npy")
        pixels_c = np.load(f"{output_dir}/card.npy")
        
        print(f"Loaded arrays from {output_dir}/..")
        #remove idx_to_exclude 
        all_idx = [x for x in all_idx if x not in range_to_exclude]
        pixels_l = pixels_l[all_idx]
        print(len(pixels_l))
        pixels_r = pixels_r[all_idx]
        pixels_c = pixels_c[all_idx]
        pixels_p = pixels_p[all_idx]
        theta_card, theta_paper = print_threshold_suggestion(pixels_l,pixels_r,pixels_p,pixels_c)
        #plot(pixels_l_stats,pixels_r_stats,pixels_p_stats,pixels_c_stats, output_dir)
        
        #make histogram
        plot_histogram(pixels_l,pixels_r,pixels_p,pixels_c,theta_card,theta_paper,output_dir)
        return
    return 


    #now we need to maximize contrast such that background as much as possible becomes black. Let us tune by gathering statistics on a large background area 
    #fmt: upper left corner, lower right corner 
    y_bg_l = [400,600]
    x_bg_l = [80,280]
    y_bg_r = [400,600]
    x_bg_r = [710,910]
    ###these two slice ranges are the cocmpletely same format but do not look correct when plotting the rect
    y_p = [410,610]
    x_p = [360,660]
    y_c = [313,342]
    x_c = [330,410]


    l_h = y_bg_l[1]-y_bg_l[0]
    l_w = x_bg_l[1]-x_bg_l[0]
    r_h = y_bg_r[1]-y_bg_r[0]
    r_w = x_bg_r[1]-x_bg_r[0]
    p_h = y_p[1]-y_p[0]
    p_w = x_p[1]-x_p[0]
    c_h = y_c[1]-y_c[0]
    c_w = x_c[1]-x_c[0]


    pixels_l = np.zeros((NSAMPLE,l_h*l_w))
    pixels_r = np.zeros((NSAMPLE,r_h*r_w))
    pixels_p = np.zeros((NSAMPLE,p_h*p_w))
    pixels_c = np.zeros((NSAMPLE,c_h*c_w))

    pixels_l_stats = np.zeros((NSAMPLE,3))
    pixels_r_stats = np.zeros((NSAMPLE,3))
    pixels_p_stats = np.zeros((NSAMPLE,3))
    pixels_c_stats = np.zeros((NSAMPLE,3))
    filenames = []

    
    for i, idx in enumerate(all_idx): 
        if(i%100==0):
            print(f"{i}/{NSAMPLE}")
        fname = all_files[idx]
        im = imread(fname)
        if NORMALIZE: #histogram stretch - decided against 
            im = im/255.0
            norm_im = (im-im.min())/(im.max()-im.min())
            norm_im = (norm_im*255).astype(np.uint8)
            im_out = Image.fromarray(norm_im)
            im_out.save(f"{img_out_folder}/{idx}.png")
        else: 
            norm_im = im
        pix_slice_left = norm_im[y_bg_l[0]:y_bg_l[1],x_bg_l[0]:x_bg_l[1]]
        pix_slice_right = norm_im[y_bg_r[0]:y_bg_r[1],x_bg_r[0]:x_bg_r[1]]
        pix_slice_paper = norm_im[y_p[0]:y_p[1],x_p[0]:x_p[1]]
        pix_slice_card = norm_im[y_c[0]:y_c[1],x_c[0]:x_c[1]]
        pixels_l[i] = pix_slice_left.ravel()
        pixels_r[i] = pix_slice_right.ravel()
        pixels_p[i] = pix_slice_paper.ravel()
        pixels_c[i] = pix_slice_card.ravel()
        pixels_l_stats[i,:] = np.array([pixels_l[i].min(),pixels_l[i].mean(),pixels_l[i].max()])
        pixels_r_stats[i,:] = np.array([pixels_r[i].min(),pixels_r[i].mean(),pixels_r[i].max()])
        pixels_p_stats[i,:] = np.array([pixels_p[i].min(),pixels_p[i].mean(),pixels_p[i].max()])
        pixels_c_stats[i,:] = np.array([pixels_c[i].min(),pixels_c[i].mean(),pixels_c[i].max()])   
        filenames.append(fname) 
        

    pixels_l = pixels_l.ravel()
    pixels_r = pixels_r.ravel()
    pixels_p = pixels_p.ravel()
    pixels_c = pixels_c.ravel()



    np.save(f"{output_dir}/left_box.npy",pixels_l)
    np.save(f"{output_dir}/right_box.npy",pixels_r)
    np.save(f"{output_dir}/paper.npy",pixels_p)
    np.save(f"{output_dir}/card.npy",pixels_c)
    np.save(f"{output_dir}/left_box_stats.npy",pixels_l_stats)
    np.save(f"{output_dir}/right_box_stats.npy",pixels_r_stats)
    np.save(f"{output_dir}/paper_stats.npy",pixels_p_stats)
    np.save(f"{output_dir}/card_stats.npy",pixels_c_stats)
    np.save(f"{output_dir}/filenames.csv",delimiter=",",object=np.array(filenames))
    

    print(f"Saved arrays to {output_dir}/..")


    #remove idx_to_exclude 
    print("Showing histograms etc for included frames only")
    all_idx = [x for x in all_idx if x not in range_to_exclude]
    pixels_l = pixels_l[all_idx]
    print(len(pixels_l))
    pixels_r = pixels_r[all_idx]
    pixels_c = pixels_c[all_idx]
    pixels_p = pixels_p[all_idx]


    theta_card, theta_paper = print_threshold_suggestion(pixels_l,pixels_r,pixels_p,pixels_c)
        #make histogram
    plot_histogram(pixels_l,pixels_r,pixels_p,pixels_c,theta_card,theta_paper,output_dir)


    plot(pixels_l_stats,pixels_r_stats,pixels_p_stats,pixels_c_stats, output_dir)
    return 

if __name__=="__main__":
    main()
    # in_folder = "data/images/with_calib/crops"
    # flist = glob(f"{in_folder}/*.png")
    # fig, ax = plt.subplots(1,2)
    # ax[0].imshow(imread(flist[0]))
    # ax[1].imshow(imread(flist[100]))
    # plt.show()