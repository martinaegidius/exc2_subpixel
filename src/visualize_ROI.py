from glob import glob 
import os


import matplotlib.pyplot as plt 
from matplotlib.patches import Rectangle
import numpy as np 
from skimage.io import imread 

#let's load some frames 
in_folder = "data/images/with_calib/crops"
all_files = glob(f"{in_folder}/*.png") #data has already been proc with burn in 250 and burn out 50; only need to filter bad frames 
print(len(all_files))

im = imread(all_files[0])
NSAMPLE = len(all_files)

all_idx = np.arange(len(all_files))
#all_idx = np.random.permutation(np.arange(len(all_files)))[:NSAMPLE]

train_ims = np.zeros((NSAMPLE,im.shape[0],im.shape[1])).astype(np.uint8)
for i, idx in enumerate(all_idx):
    print(all_files[idx]) 
    im = imread(all_files[idx])
    im = im/255.0
    #print(im.min(),im.max())
    norm_im = (im-im.min())/(im.max()-im.min())
    #print(norm_im.min(),norm_im.max())
    #fig, ax = plt.subplots(1,2)
    #ax[0].hist(im.ravel(),bins=50)
    #ax[1].hist(norm_im.ravel(),bins=50)
    train_ims[i] = (norm_im*255).astype(np.uint8)
    #print(train_ims[i].min(),train_ims[i].max())
    #plt.show() 
    break





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



# #checking boxes
fig, ax = plt.subplot_mosaic(
    [
        ["main", "main"],
        ["left", "right"],
        ["card", "paper"]
    ],
    figsize=(10, 6)
)

ax["main"].imshow(train_ims[0],cmap="gray")
rect_l = Rectangle(
    (x_bg_l[0], y_bg_l[0]),
    x_bg_l[1] - x_bg_l[0],   # width
    y_bg_l[1] - y_bg_l[0],   # height
    linewidth=1, edgecolor='r', facecolor='none'
)

rect_r = Rectangle(
    (x_bg_r[0], y_bg_r[0]),
    x_bg_r[1] - x_bg_r[0],
    y_bg_r[1] - y_bg_r[0],
    linewidth=1, edgecolor='orange', facecolor='none'
)

rect_p = Rectangle(
    (x_p[0], y_p[0]),
    x_p[1] - x_p[0],
    y_p[1] - y_p[0],
    linewidth=1, edgecolor='b', facecolor='none'
)

rect_c = Rectangle(
    (x_c[0], y_c[0]),
    x_c[1] - x_c[0],
    y_c[1] - y_c[0],
    linewidth=1, edgecolor='magenta', facecolor='none'
)

def add_rect(ax, x, y, color):
    rect = Rectangle(
        (x[0], y[0]),
        x[1] - x[0],
        y[1] - y[0],
        linewidth=1,
        edgecolor=color,
        facecolor="none"
    )
    ax.add_patch(rect)

# ax["main"].add_patch(rect_r)
# ax["main"].add_patch(rect_l)
# ax["main"].add_patch(rect_p)
# ax["main"].add_patch(rect_c)

add_rect(ax["main"], x_bg_l, y_bg_l, "r")
add_rect(ax["main"], x_bg_r, y_bg_r, "orange")
add_rect(ax["main"], x_p, y_p, "b")
add_rect(ax["main"], x_c, y_c, "magenta")

pix_slice_left = train_ims[0][y_bg_l[0]:y_bg_l[1],x_bg_l[0]:x_bg_l[1]]
pix_slice_right = train_ims[0][y_bg_r[0]:y_bg_r[1],x_bg_r[0]:x_bg_r[1]]
pix_slice_card = train_ims[0][y_c[0]:y_c[1],x_c[0]:x_c[1]]
pix_slice_paper = train_ims[0][y_p[0]:y_p[1],x_p[0]:x_p[1]]

os.makedirs("plots",exist_ok=True)
ax["left"].imshow(pix_slice_left,cmap="gray")
ax["right"].imshow(pix_slice_right,cmap="gray")
ax["card"].imshow(pix_slice_paper,cmap="gray")
ax["paper"].imshow(pix_slice_card,cmap="gray")
plt.savefig("plots/complete_fig_roi.pdf",dpi=300)
plt.show()



fig, ax = plt.subplots(1,1,figsize=(6, 6))
ax.imshow(train_ims[0],cmap="gray")
add_rect(ax, x_bg_l, y_bg_l, "r")
add_rect(ax, x_bg_r, y_bg_r, "orange")
add_rect(ax, x_p, y_p, "b")
add_rect(ax, x_c, y_c, "magenta")
ax.axis("off")
plt.savefig("plots/image_ROI.pdf",dpi=300)


def save_slice(slice_arr, name):
    plt.figure()
    plt.imshow(slice_arr,cmap="gray")
    plt.axis("off")
    plt.savefig(f"plots/{name}",dpi=300)
    plt.close()

save_slice(pix_slice_left,"left_bg_slice.pdf")
save_slice(pix_slice_right,"right_bg_slice.pdf")
save_slice(pix_slice_paper,"paper_slice.pdf")
save_slice(pix_slice_card,"card_slice.pdf")

def save_slice_w_border(slice_arr, name, border_col, lw=2):
    h, w = slice_arr.shape[:2]

    dpi = 100  # arbitrary, but fixed
    fig = plt.figure(figsize=(w / dpi, h / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # fill figure exactly

    ax.imshow(slice_arr, cmap="gray", interpolation="nearest")
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    inset = lw / 2
    border = Rectangle(
        (inset, inset),
        w - lw,
        h - lw,
        linewidth=lw,
        edgecolor=border_col,
        facecolor="none"
    )
    ax.add_patch(border)

    fig.savefig(f"plots/{name}", dpi=dpi)
    plt.close(fig)

# def save_slice_w_border(slice_arr, name, border_col, lw=2):
#     h, w = slice_arr.shape[:2]

#     fig, ax = plt.subplots()

#     ax.imshow(
#         slice_arr,
#         cmap="gray",
#         extent=(0, w, h, 0),  # explicit pixel grid
#         aspect="auto"        # disable square-pixel enforcement
#     )

#     ax.set_xlim(0, w)
#     ax.set_ylim(h, 0)
#     ax.axis("off")

#     inset = lw / 2
#     border = Rectangle(
#         (inset, inset),
#         w - lw,
#         h - lw,
#         linewidth=lw,
#         edgecolor=border_col,
#         facecolor="none"
#     )
#     ax.add_patch(border)

#     plt.savefig(
#         f"plots/{name}",
#         dpi=300,
#         bbox_inches="tight",
#         pad_inches=0
#     )
#     plt.close()

save_slice_w_border(pix_slice_left,"left_bg_slice_b.pdf","r",lw=4)
save_slice_w_border(pix_slice_right,"right_bg_slice_b.pdf","orange",lw=4)
save_slice_w_border(pix_slice_paper,"paper_slice_b.pdf","b",lw=4)
save_slice_w_border(pix_slice_card,"card_slice_b.pdf","magenta",lw=1)


# pixels_l = np.zeros((NSAMPLE,l_h*l_w))
# pixels_r = np.zeros((NSAMPLE,r_h*r_w))
# pixels_p = np.zeros((NSAMPLE,p_h*p_w))
# pixels_c = np.zeros((NSAMPLE,c_h*c_w))


# for i, im in enumerate(train_ims):
#     pix_slice_left = im[y_bg_l[0]:y_bg_l[1],x_bg_l[0]:x_bg_l[1]]
#     pix_slice_right = im[y_bg_r[0]:y_bg_r[1],x_bg_r[0]:x_bg_r[1]]
#     pix_slice_paper = im[y_p[0]:y_p[1],x_p[0]:x_p[1]]
#     pix_slice_card = im[y_c[0]:y_c[1],x_c[0]:x_c[1]]
    
    
#     pixels_l[i] = pix_slice_left.ravel()
#     pixels_r[i] = pix_slice_right.ravel()
#     pixels_p[i] = pix_slice_paper.ravel()
#     pixels_c[i] = pix_slice_card.ravel()
    

# pixels_l = pixels_l.ravel()
# pixels_r = pixels_r.ravel()
# pixels_p = pixels_p.ravel()
# pixels_c = pixels_c.ravel()



# fig, ax = plt.subplots(1,1)
# plt.hist(pixels_l,density=True,bins=30, label="Left box")
# plt.hist(pixels_r,density=True,bins=30, label="Right box")
# plt.hist(pixels_p,density=True,bins=30, label="Paper")
# plt.hist(pixels_c,density=True,bins=30, label="Card")

# plt.show()

