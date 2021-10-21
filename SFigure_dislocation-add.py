# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.12.0
#   kernelspec:
#     display_name: pyGPA-cupy
#     language: python
#     name: pygpa-cupy
# ---

# %% [markdown]
# # Generate SFigures: Dislocations
#
# Sources: 
# - `code/twistedbilayer/pyGPA/dislocationGPA-real_and_artificial.ipynb`
# - `code/twistedbilayer/pyGPA/Figure-dislocation_arbitrary_burgers_vector.ipynb`

# %%
from skimage.io import imread
import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import colorcet

import pyGPA.geometric_phase_analysis as GPA
import pyGPA.property_extract as pe
from pyGPA.imagetools import indicate_k, gauss_homogenize2
from moisan2011 import per


# %% [markdown]
# ## Additional dislocations

# %%
folder = "/mnt/storage-linux/speeldata/20200715-XTBLG02/20200715_154404_0.66um_495.9_sweep-STAGE_X-STAGE_Y_highres_highE"
name = "stitch_v10_2020-11-20_1843_sobel_5_bw_200.tif"

oimage = imread(os.path.join(folder,name)).squeeze()[2700:4200,1850:3200].astype(np.float64)
plt.imshow(oimage.T)

# %%
NMPERPIXEL=0.88

layout = """
    aaa
    bcd
"""

fig = plt.figure(figsize=[6,7.7], constrained_layout=True)
axs = fig.subplot_mosaic(layout,
                             gridspec_kw={"height_ratios": [3, 1],
                                          #"width_ratios": [4, 1],
                             })
                    
kwargs = dict(alpha=0.5)
ikwargs = dict(cmap='gray', vmax=np.quantile(oimage, 1-1e-6), 
               vmin=np.quantile(oimage, 0.03), interpolation='none')
im = axs['a'].imshow(oimage.T, **ikwargs)
#im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1e3)
axs['a'].set_ylabel('y (μm)')
axs['a'].set_xlabel('x (μm)')

coords = np.array([[550,420],
                  [1275,1050],
                  [250,930]])
axi = [axs[l] for l in 'bcd']
dr = 150
dr2 = 100
for i,coord in enumerate(coords):
    #axs['a'].scatter(*(coord*NMPERPIXEL/1e3), **kwargs)
    rect = plt.Rectangle((coord-dr2)*NMPERPIXEL/1e3, 
                     dr2*2*NMPERPIXEL/1e3,
                     dr2*2*NMPERPIXEL/1e3,
                      edgecolor=f'C{i}',
                      fill=False, alpha=0.99, lw=1,
                      #path_effects=[patheffects.withStroke(linewidth=2, foreground="white", alpha=0.3)]
                        )
    axs['a'].add_artist(rect)
    lim = oimage[coord[0]-dr:coord[0]+dr,
                 coord[1]-dr:coord[1]+dr]
    lim2 = oimage[coord[0]-dr2:coord[0]+dr2,
                 coord[1]-dr2:coord[1]+dr2]
    im2 = axi[i].imshow(lim2.T, **ikwargs)
    im2.set_extent(np.array(im2.get_extent())*NMPERPIXEL)
    
    axi[i].set_xlabel('x (nm)')
    pks,_ = GPA.extract_primary_ks(lim - lim.mean(), pix_norm_range=(4,50))
    print("pks", pks)
    props = pe.Kerelsky_plus(pks, nmperpixel=NMPERPIXEL, sort=1)
    axi[i].set_title(f"$\\theta \\approx ${props[0]:.2f}°, $\\epsilon \\approx ${props[2]*100:.1f}%", 
                     fontsize='small')
    for axis in ['top','bottom','left','right']:
        axi[i].spines[axis].set_linewidth(2)
        axi[i].spines[axis].set_color(f'C{i}')
axi[1].tick_params(axis='y', which='both', labelleft=False, labelright=False)
axi[2].tick_params(axis='y', which='both', labelleft=False, labelright=True)
axi[2].yaxis.tick_right()
axi[2].yaxis.set_label_position("right")
axi[0].set_ylabel('y (nm)')
axi[2].set_ylabel('y (nm)')
im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1e3)
for key in axs.keys():
    axs[key].set_title(key, loc='left', fontweight='bold')
plt.savefig(os.path.join('figures', 'SI_dislocations.pdf'))

# %% [markdown]
# ## Overlay

# %%
from skimage.registration import phase_cross_correlation
from skimage.transform import warp_polar, rotate, rescale
import scipy.ndimage as ndi
import pyGPA.geometric_phase_analysis as GPA
import pyGPA.property_extract as pe
from latticegen import hexlattice_gen, physical_lattice_gen
from latticegen.transformations import epsilon_to_kappa, a_0_to_r_k, r_k_to_a_0, rotation_matrix, apply_transformation_matrix
import latticegen

# %%
folder2 = "/mnt/storage-linux/speeldata/20200713-XTBLG02/20200713_163811_5.7um_501.2_sweep-STAGE_X-STAGE_Y_domainboundaries"
name2 = "stitch_v10_2020-11-20_1649_sobel_4_bw_200.tif"

r1=1500
r2=500
c0 = [3200,3000]
c1 = [4075, 6900]
image1 = imread(os.path.join(folder,name)).squeeze()[c0[0]-r1:c0[0]+r1,c0[1]-r1:c0[1]+r1].astype(np.float64)
image2 = imread(os.path.join(folder2,name2)).squeeze()[c1[0]-r2:c1[0]+r2,c1[1]-r2:c1[1]+r2].astype(np.float64)

# %%
fig, axs = plt.subplots(ncols=2, figsize=[12,12])
axs[0].imshow(image1[::-1,::-1])
axs[0].scatter(r1,r1, color='red')
axs[1].imshow(image2)
axs[1].scatter(r2,r2, color='red')

# %%
scaling = 0.88/3.7 * 0.999
scaling

# %%
fig, axs = plt.subplots(ncols=2, figsize=[24,14])
scaled = rescale(image1[::-1,::-1], scaling)

rotated = rotate(image2,-10+0.9+0.45, resize=True)
r1s = scaled.shape[0]//2
center = np.array(rotated.shape)//2
rcropped = rotated[center[0]-r1s:center[0]+r1s+scaled.shape[0] % 2,
                      center[1]-r1s:center[1]+r1s+scaled.shape[0] % 2]
im = axs[0].imshow(scaled, vmax=np.quantile(scaled,0.9), vmin=np.quantile(scaled,0.05))

im = axs[1].imshow(rcropped, vmax=np.quantile(rcropped,0.85), vmin=np.quantile(rcropped,0.07))
for ax in axs:
    ax.grid()

# %%
plt.figure(figsize=[24,24])
green = scaled
purple = rcropped
normed_green = green - np.quantile(green,0.05)
normed_green /= np.quantile(normed_green,0.9)
norm_purple = purple - np.quantile(purple,0.05)
norm_purple /= np.quantile(norm_purple,0.9)
plt.imshow(np.stack([norm_purple,
                     normed_green, 
                     norm_purple,
                     ], axis=-1))

# %%
shifts = [0,0]

# %%
plt.figure(figsize=[24,24])
green = scaled[320:570,300:500].copy()
purple = rcropped[320:570,300:500].copy()
purple = ndi.shift(purple, shifts, mode='reflect')
green = gauss_homogenize2(green,sigma=6, mask=np.ones_like(green))
purple = gauss_homogenize2(purple, sigma=6, mask=np.ones_like(purple))
norm_green = green - np.quantile(green,0.001)
norm_green /= np.quantile(norm_green,0.999)
norm_purple = purple - np.quantile(purple,0.005)
norm_purple /= np.quantile(norm_purple,0.995)
plt.imshow(np.stack([norm_purple,
                     norm_green, 
                     norm_purple,
                     ], axis=-1))

# %%
shifts, error, phasediff = phase_cross_correlation(green-green.mean(),
                                                   purple-purple.mean(), 
                                                   upsample_factor=100)
shifts, error

# %%
kvecs,_ = GPA.extract_primary_ks(green, plot=True, pix_norm_range=(10,70), threshold=0.3)
theta, psi, epsilon, xi = pe.Kerelsky_plus(kvecs, nmperpixel=NMPERPIXEL / scaling, sort=1)
print(theta, xi, epsilon, psi)

# %%
fig, axs = plt.subplots(ncols=2, figsize=[12,5.1], sharex=True, sharey=True)
im = axs[0].imshow(purple.T, cmap='gray')
im.set_extent(np.array(im.get_extent())* 3.7)
im = axs[1].imshow(green.T, cmap='gray', )
im.set_extent(np.array(im.get_extent())* 3.7)

axs[0].set_title('Before')
axs[1].set_title('After')
for ax in axs:
    ax.set_xlabel('x (nm)')
axs[0].set_ylabel('y (nm)')

#circle1 = plt.Circle((800, 275), 100, edgecolor='r', facecolor='none')
#axs[0].add_patch(circle1)

axs[0].annotate('', (800, 275), (800-150, 275-150), arrowprops=dict(facecolor='red', shrink=0.2),
               horizontalalignment='left', verticalalignment='bottom',)
axs[1].annotate('', (350, 390), (350-150, 390-150), arrowprops=dict(facecolor='red', shrink=0.2),
               horizontalalignment='left', verticalalignment='bottom',)

for ax, label in zip(axs, 'ab'):
    ax.set_title(label, loc='left', fontweight='bold')
plt.tight_layout()
#plt.savefig(os.path.join('figures', 'SI_movement_top.pdf'))

# %%
S = 2000
pixelspernm=scaling/NMPERPIXEL*100
l1 = physical_lattice_gen(0.246, xi, 
                          2, 
                          pixelspernm=pixelspernm, size=S).compute()
l1 = l1-l1.min()
r_k2, kappa2 = epsilon_to_kappa(a_0_to_r_k(0.246), epsilon)
ks1 = latticegen.generate_ks(a_0_to_r_k(0.246), xi, kappa=1)
ks2 = latticegen.generate_ks(r_k2, xi+theta, kappa=kappa2, psi=psi)

l2 = physical_lattice_gen(r_k_to_a_0(r_k2), xi+theta, 
                          2, 
                          pixelspernm=pixelspernm, kappa=kappa2, psi=psi, size=S).compute()
l2 = l2-l2.min()
gauss_sigma = 3
moire = ndi.gaussian_filter(np.sqrt(l1*l1+l2*l2), gauss_sigma)[2*gauss_sigma:-2*gauss_sigma,
                                                               2*gauss_sigma:-2*gauss_sigma]

# %%
f, axs = plt.subplots(ncols=3, figsize=[12,4.2])
r = 950
im = axs[0].imshow(l1[r:-r,r:-r].T, cmap='cet_fire')
im.set_extent(np.array(im.get_extent())/ pixelspernm)
im = axs[1].imshow(moire.T, cmap='cet_fire')
im.set_extent(np.array(im.get_extent())/ pixelspernm)
im = axs[2].imshow(l2[r:-r,r:-r].T, cmap='cet_fire')
im.set_extent(np.array(im.get_extent())/ pixelspernm)
axs[0].set_title(f'Layer 1: {-xi %60:.2f}°')
axs[1].set_title('Simulated moiré')
axs[2].set_title(f'Layer 2: {(-theta-xi) %60:.2f}°, ϵ={epsilon*100:.2f}%')
for ax in axs:
    ax.set_xlabel('x (nm)')
axs[0].set_ylabel('y (nm)')
for ax, label in zip(axs, 'cde'):
    ax.set_title(label, loc='left', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join('figures', 'SI_movement_bottom.pdf'))
