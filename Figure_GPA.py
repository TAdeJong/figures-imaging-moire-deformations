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
#     display_name: Python [conda env:pyGPA-test]
#     language: python
#     name: conda-env-pyGPA-test-py
# ---

# %%
import dask.array as da
import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.io import imread

import colorcet

from skimage.feature import peak_local_max
import scipy.ndimage as ndi

from moisan2011 import per
from pyGPA.phase_unwrap import phase_unwrap
from pyGPA.imagetools import fftplot, gauss_homogenize2, gauss_homogenize3, trim_nans, indicate_k
from pyGPA.mathtools import wrapToPi
import pyGPA.geometric_phase_analysis as GPA
import pyGPA.property_extract as pe

# %%
from skimage import util, filters, color
from skimage.segmentation import watershed
import skimage.morphology as morpho
from skimage.filters import gaussian
from skimage.measure import regionprops, label

def select_large_areas(labeling, threshold=5000):
    lbl = labeling
    nlbl = np.zeros_like(lbl)
    rprops = regionprops(lbl)
    #print([prop.area for prop in rprops if prop.area > 5000])
    i = 1
    for prop in rprops:
        if prop.area > threshold:
            nlbl[lbl==prop.label] = i
            i += 1
    return nlbl


# %%
folder = "/mnt/storage-linux/speeldata/20200713-XTBLG02/20200713_163811_5.7um_501.2_sweep-STAGE_X-STAGE_Y_domainboundaries"
name = "stitch_v10_2020-11-20_1649_sobel_4_bw_200.tif"

# %%
image = imread(os.path.join(folder,name)).squeeze()

# %%
plt.imshow(image)

# %%
coords = np.array([[3642.89786153, 5400.49488182],
        [3721.72436096, 6384.55472946],
        [3507.55425954, 8281.36876338],
        [3380.03002363, 6921.11024705],
        [4187.6835177 , 6938.49991558],
        [3807.04299538, 7066.02415149],
        [4812.93836045, 7729.48655758],
        [4099.87750689, 9225.98507689],
        [ 850.69462839, 4398.48610047],
        [ 363.96565178, 1325.27028875],
        [4271.33017045, 7833.81755442],
        [2735.31323933, 4710.03264254]])

# %%
NMPERPIXEL=3.7

# %%
glasbey = plt.get_cmap('cet_glasbey_dark')(np.linspace(0,1,255))
fig, axs = plt.subplots(3, 4, figsize=[15,10], constrained_layout=True, sharex=True, sharey=True)
axs = axs.flat
s = 500
sprimes = []
bbox_images = []
regions = []
full_crops = []
all_dists = []
for i, coord in enumerate(coords):
    x = slice(max(int(coord[0])-s,0), int(coord[0])+s)
    y = slice(max(int(coord[1])-s,0), int(coord[1])+s)
    lim = image[x, y]
    if lim.shape[0] != 2*s:
        lim = np.pad(lim,((s-lim.shape[0]//2, s-lim.shape[0]//2), (0,0)))
    full_crops.append(lim)
    clip = np.clip(lim, *np.quantile(lim[lim>0], [0.05,0.9999]))
    dclip = (clip / gaussian(clip,50)) >= 0.9
    ero = morpho.binary_erosion(dclip, selem=morpho.disk(5))
    mask2 = morpho.remove_small_holes(ero,25**2)
    nextone = morpho.binary_erosion(gauss_homogenize3(clip, mask2,  25) > 0.91, selem=morpho.disk(8))
    labeling = label(nextone)
    llabels = select_large_areas(labeling, threshold=200)
    #labelim = np.where(llabels == 0, np.nan, llabels)
    wres = watershed(-(gauss_homogenize3(clip, mask2,  15) ), llabels, mask=clip>1e4) #1e4 is determined from histograms of images. Maybe use Otsu's methods instead?
    if wres[0].max() == 0:
        region = regionprops(wres, intensity_image=lim)[wres[s,s]-2]
    else:
        region = regionprops(wres, intensity_image=lim)[wres[s,s]-1]
    regions.append(region)
    print(wres.shape, wres[s,s]+1, region.label, wres[0].max())
    dists = ndi.distance_transform_cdt(morpho.remove_small_holes(wres == wres[s,s], 25**2), metric='chessboard')
    all_dists.append(dists)
    sprimes.append(dists[s,s])
    bbox_images.append(lim[region.slice])
    coord = np.unravel_index(np.argmax(dists), dists.shape)
    axs[i].scatter(coord[0]*NMPERPIXEL,coord[1]*NMPERPIXEL)
    axs[i].scatter(s*NMPERPIXEL,s*NMPERPIXEL)
    
    im = axs[i].imshow(lim.T, vmax=np.quantile(lim, 0.999),
                 vmin=np.quantile(lim, 0.001), cmap='gray')
    im.set_extent(np.array(im.get_extent())*NMPERPIXEL)
    im = axs[i].imshow(np.where(wres == 0, np.nan, wres).T-0.5, cmap='cet_glasbey_bw_minc_20', interpolation='nearest', alpha=0.3, vmax=256, vmin=0)
    im.set_extent(np.array(im.get_extent())*NMPERPIXEL)
    im = axs[i].imshow(np.where(dists > 0, dists, np.nan).T, alpha=0.2)
    #im = axs[i].imshow(np.where(clip>clip.min()*1.2, 1, np.nan).T)
    im.set_extent(np.array(im.get_extent())*NMPERPIXEL)
    if i > 7:
        axs[i].set_xlabel('nm')
    if i %4 == 0:
        axs[i].set_ylabel('nm')
    if i %4 == 3:
        axs[i].yaxis.tick_right()
        axs[i].set_ylabel('nm')
        axs[i].yaxis.set_label_position("right")
        axs[i].tick_params(axis='y', which='both', labelleft=False, labelright=True)
    for axis in ['top','bottom','left','right']:
        axs[i].spines[axis].set_linewidth(4)
        axs[i].spines[axis].set_color(glasbey[i])
sprimes = np.array(sprimes)
#plt.savefig('crops.pdf', interpolation='none')

# %%
sprimes

# %%
fig, axs = plt.subplots(3, 4, figsize=[15,10], constrained_layout=True, sharex=True, sharey=True)
axs = axs.flat
thetas = []
kvec_lists = []
for i, coord in enumerate(coords):
    d = 20
    x = slice(int(coord[0])-sprimes[i]+d, int(coord[0])+sprimes[i]-d)
    y = slice(int(coord[1])-sprimes[i]+d, int(coord[1])+sprimes[i]-d)
    lim = image[x, y].astype(np.float64)
    #lim = np.clip(lim, *np.quantile(lim, [0.001,0.999]))
    pks,_ = GPA.extract_primary_ks(gauss_homogenize2(lim, np.ones_like(lim), 50), 
                                   pix_norm_range=(4,50), plot=False, threshold=0.7, sigma=1.5)
    if i == 0:
        print(pks)
        pks[2] = pks[0]-pks[1]
    props = pe.Kerelsky_plus(pks, nmperpixel=NMPERPIXEL, sort=1)
    print("props:", pe.Kerelsky_plus(pks, nmperpixel=NMPERPIXEL, sort=1)[0])
    kvec_lists.append(pks)
    p,_ =  per(lim-lim.mean(), inverse_dft=False)
    fftim = np.abs(np.fft.fftshift(p))
    fftplot(fftim, ax=axs[i], pcolormesh=False, vmax=np.quantile(fftim, 0.999),
                 vmin=np.quantile(fftim, 0.01), cmap='cet_fire_r', d=NMPERPIXEL, interpolation='none')
    axs[i].scatter(*(pks/NMPERPIXEL).T+0.5/NMPERPIXEL/s, edgecolor='C0', alpha=0.7, s=150, marker='o', color='none')
    axs[i].scatter(*(-pks/NMPERPIXEL).T+0.5/NMPERPIXEL/s, edgecolor='C0', alpha=0.7, s=150, marker='o', color='none')
    fftr=0.08
    axs[i].set_xlim([-fftr,fftr])
    axs[i].set_ylim([-fftr,fftr])
    thetas.append(GPA.f2angle(np.linalg.norm(pks, axis=1).mean(), nmperpixel=NMPERPIXEL))
    axs[i].set_title(f"$\\theta = ${props[0]:.2f}°, $\\epsilon = ${props[2]*100:.2f}%")
    
    if i > 7:
        axs[i].set_xlabel('nm$^{-1}$')
    if i %4 == 0:
        axs[i].set_ylabel('nm$^{-1}$')
    if i %4 == 3:
        axs[i].yaxis.tick_right()
        axs[i].set_ylabel('nm$^{-1}$')
        axs[i].yaxis.set_label_position("right")
        axs[i].tick_params(axis='y', which='both', labelleft=False, labelright=True)
    for axis in ['top','bottom','left','right']:
        axs[i].spines[axis].set_linewidth(4)
        axs[i].spines[axis].set_color(glasbey[i])
fig.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72.,
        hspace=0., wspace=-0.15)

# %%
all_props = np.array([pe.Kerelsky_plus(pks, nmperpixel=NMPERPIXEL, sort=1) for pks in kvec_lists])
fig, ax = plt.subplots(ncols=2, figsize=[12,5])

ax[0].scatter(all_props[:,0], (all_props[:,3]+25) % 60 - 25, c=np.arange(all_props.shape[0]), cmap='cet_glasbey_dark', vmax=255)
ax[0].set_ylabel('angle ξ, (deg)')
ax[0].set_xlabel('twist angle θ, (deg)')

ax[1].scatter(all_props[:,0], all_props[:,2]*100, c=np.arange(all_props.shape[0]), cmap='cet_glasbey_dark', vmax=255)
ax[1].set_ylabel('heterostrain ϵ (%)')
ax[1].set_xlabel('twist angle θ, (deg)')
ax[1].set_ylim(0, None)
for i in range(2):
    ax[i].set_xlim(-0.02, 0.7)

# %%
spm = np.min(sprimes)
sprimes, spm

# %%
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
fig, axs = plt.subplots(3, 4, figsize=[18,13.5], constrained_layout=True)#, sharex=True, sharey=True)
axs = axs.flat
sindices = np.argsort(thetas)
sdp = sprimes[sindices]
coordsl = coords[sindices]
for i, coord in enumerate(coordsl):
    x = slice(int(coord[0])-sdp[i], int(coord[0])+sdp[i])
    y = slice(int(coord[1])-sdp[i], int(coord[1])+sdp[i])
    #x = slice(int(coord[0])-spm, int(coord[0])+spm)
    #y = slice(int(coord[1])-spm, int(coord[1])+spm)
    lim = image[x, y]
    print(lim.shape)
    lim = np.clip(lim, *np.quantile(lim, [0.001,0.999]))
    #pks,_ = extract_primary_ks(lim-lim.mean(), pix_norm_range=(4,50), plot=False, threshold=0.2)
    #pks,_ = extract_primary_ks(gauss_homogenize2(lim, np.ones_like(lim), 50)[d:-d,d:-d], pix_norm_range=(4,50), plot=False, threshold=0.7, sigma=1.5)
    pks = kvec_lists[sindices[i]]
    props = pe.Kerelsky_plus(pks, nmperpixel=NMPERPIXEL, sort=1)
    p,_ =  per(lim-lim.mean(), inverse_dft=False)
    fftim = np.abs(np.fft.fftshift(p))
    axin = inset_axes(axs[i], width="25%", height="25%", loc=2)
    axin.tick_params(labelleft=False, labelbottom=False, direction='in', length=0)
    for axis in ['top','bottom','left','right']:
        axin.spines[axis].set_color(f"white")

    fftplot(fftim, ax=axin, pcolormesh=False, vmax=np.quantile(fftim, 0.995),
                 vmin=np.quantile(fftim, 0.01), cmap='cet_fire_r', d=NMPERPIXEL, interpolation='none')
    axin.scatter(*(pks/NMPERPIXEL).T+0.5/NMPERPIXEL/s, edgecolor=f'C{sindices[i]}', alpha=0.7, s=70, marker='o', color='none')
    axin.scatter(*(-pks/NMPERPIXEL).T+0.5/NMPERPIXEL/s, edgecolor=f'C{sindices[i]}', alpha=0.7, s=70, marker='o', color='none')
    rf = 0.08
    axin.set_xlim([-rf,rf])
    axin.set_ylim([-rf,rf])
    #axs[i].set_title(f"$\\theta \\approx ${thetas[sindices[i]]:.2f}°", weight='bold')
    axs[i].set_title(f"$\\theta \\approx ${props[0]:.2f}°, $\\epsilon \\approx ${props[2]*100:.2f}%", weight='bold')
    plim = lim
    
    im = axs[i].imshow(plim,#[d:-d,d:-d], 
                       #vmax=3.8e4,#
                       vmax=np.quantile(plim[d:-d,d:-d], 0.9999),
                       #vmin= 1.5e4, #
                       vmin=np.quantile(plim[d:-d,d:-d], 0.0001), 
                       cmap='gray')
    im.set_extent(np.array(im.get_extent())*NMPERPIXEL)
    #plt.colorbar(im, ax=axs[i])
    if i > 7:
        axs[i].set_xlabel('nm')
    if i %4 == 0:
        axs[i].set_ylabel('nm')
    elif i %4 == 3:
        axs[i].yaxis.tick_right()
        axs[i].set_ylabel('nm')
        axs[i].yaxis.set_label_position("right")
        axs[i].tick_params(axis='y', which='both', labelleft=False, labelright=True)
    else:
        axs[i].tick_params(axis='y', which='both', labelleft=False, labelright=False)
    for axis in ['top','bottom','left','right']:
        axs[i].spines[axis].set_linewidth(2)
        axs[i].spines[axis].set_color(glasbey[sindices[i]])
    if i == 1:
        axs[i].set_title(f"TDBG: " + axs[i].get_title(), weight='bold')
fig.set_constrained_layout_pads(w_pad=2./72., h_pad=2./72.,
        hspace=0., wspace=-0.15)
#plt.savefig('fftinsets2.pdf', interpolation='none')

# %%
from skimage.morphology import binary_erosion, selem
from matplotlib.cm import get_cmap

# %%
from importlib import reload
reload(pe)

# %%
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=10, threads_per_worker=2, memory_limit='2GB')  
client = Client(cluster)
client

# %%
gs2=[]
for pk in pks:
    g = GPA.wfr2_grad(lim - lim.mean(), sigma,
                 pk[0], pk[1], kw=kw, kstep=kstep, grad='diff')
    gs2.append(g)
grads2 = np.stack([g['grad'] for g in gs2])
print(grads2.shape)
#grads2 = np.moveaxis(GPA.gaussian_deconvolve(np.moveaxis(grads2,-1,0), sigma=sigma, dr=dr), 0, -1)
print(grads2.shape)
weights2 = np.stack([np.abs(g['lockin']) for g in gs2]) * (mask+1e-6)
unew2 = GPA.reconstruct_u_inv_from_phases(pks, -2*np.pi*grads2, weights2, pre_diff=True)

# %%
sigma

# %%
u_deconv = GPA.gaussian_deconvolve(unew2[:,dr:-dr,dr:-dr], sigma, dr, balance=5000)
plt.imshow(unew2[0,dr:-dr,dr:-dr]-u_deconv[0], vmax=0.5,vmin=-0.5)
plt.colorbar()
plt.figure()
plt.imshow(unew2[0,dr:-dr,dr:-dr])
plt.colorbar()
plt.figure()
plt.imshow(u_deconv[0])
plt.colorbar()

# %%


J1 = pe.u2J(unew2, nmperpixel=NMPERPIXEL)
J2 = pe.u2J(u_deconv, nmperpixel=NMPERPIXEL)
res1,_ = pe.Kerelsky_J(J1, pks, sort=1, nmperpixel=NMPERPIXEL)
res1 = res1.compute()
res2,_ = pe.Kerelsky_J(J2, pks, sort=1, nmperpixel=NMPERPIXEL)
res2 = res2.compute()
res1[...,1] = res1[...,1] % 180
res1[...,3] = res1[...,3] % 180
res1 = np.moveaxis(res1, -1,0)
res2[...,1] = res2[...,1] % 180
res2[...,3] = res2[...,3] % 180
res2 = np.moveaxis(res2, -1,0)

# %%
J = pe.phasegradient2J(pks, grads, weights, nmperpixel=NMPERPIXEL, iso_ref=False)
res, refest = pe.Kerelsky_J(J, pks, sort=1, nmperpixel=NMPERPIXEL)
res = res.compute()
res[...,1] = res[...,1] % 180
res[...,3] = res[...,3] % 180
res = np.moveaxis(res, -1,0)

# %%
Jd = GPA.gaussian_deconvolve(J.T, sigma, dr, balance=10).T
Jd.shape

# %%
resd, refestd = pe.Kerelsky_J(Jd, pks, sort=1, nmperpixel=NMPERPIXEL)
resd = resd.compute()
resd[...,1] = resd[...,1] % 180
resd[...,3] = resd[...,3] % 180
resd = np.moveaxis(resd, -1,0)

# %%
pe.Kerelsky_Jac(pks, nmperpixel=NMPERPIXEL, sort=1)

# %%
resd.shape

# %%
fig, axs = plt.subplots(2,2,figsize=[15,12])
kwarglist = [{}, {'cmap': 'twilight', 'interpolation': 'none', 'vmin': 0, 'vmax': 180}, {'cmap': 'cet_fire', 'vmin': 0}, {}]
for prop, ax, kwargs in zip(res1, axs.flat, kwarglist):    
    im = ax.imshow(prop[(dr):-(dr), (dr):-(dr)].T, **kwargs)
    plt.colorbar(im,ax=ax)
axs.flat[0].set_title(r'$\theta$')
axs.flat[1].set_title(r'$\psi$')
axs.flat[2].set_title(r'$\epsilon$')
axs.flat[3].set_title(r'$\xi$')

# %%
fig, axs = plt.subplots(2,2,figsize=[15,12])
kwarglist = [{}, {'cmap': 'twilight', 'interpolation': 'none', 'vmin': 0, 'vmax': 180}, {'cmap': 'cet_fire', 'vmin': 0}, {}]
for prop, ax, kwargs in zip(res, axs.flat, kwarglist):    
    im = ax.imshow(prop[(dr):-(dr), (dr):-(dr)].T, **kwargs)
    plt.colorbar(im,ax=ax)
axs.flat[0].set_title(r'$\theta$')
axs.flat[1].set_title(r'$\psi$')
axs.flat[2].set_title(r'$\epsilon$')
axs.flat[3].set_title(r'$\xi$')

# %%
fig, axs = plt.subplots(2,2,figsize=[15,12])
kwarglist = [{}, {'cmap': 'twilight', 'interpolation': 'none'}, {'cmap': 'cet_fire', 'vmin': 0}, {}]
for prop, prop2, ax, kwargs in zip(res, resd, axs.flat, kwarglist):    
    n, bins,_ = ax.hist(prop2[(dr):-(dr), (dr):-(dr)].ravel(), bins=100)
#for prop, ax, kwargs in zip(res, axs.flat, kwarglist):    
    ax.hist(prop[(dr):-(dr), (dr):-(dr)].ravel(), bins=bins, alpha=0.6)
axs.flat[0].set_title(r'$\theta$')
axs.flat[1].set_title(r'$\psi$')
axs.flat[2].set_title(r'$\epsilon$')
axs.flat[3].set_title(r'$\xi$')

# %%
from importlib import reload
reload(pe)

# %%
for i in [5]:
    coord = coordsl[i]
    pks = kvec_lists[sindices[i]]
    kw = np.linalg.norm(pks, axis=1).mean()/4
    kstep = kw/3
    sigma = max(10, 0.9/np.linalg.norm(pks,axis=1).min())
    print(sigma, 1/np.linalg.norm(pks,axis=1).mean())
    dr = int(np.ceil(3*sigma))
    x = slice(int(coord[0])-sdp[i]-dr, int(coord[0])+sdp[i]+dr)
    y = slice(int(coord[1])-sdp[i]-dr, int(coord[1])+sdp[i]+dr)
    lim = image[x, y].astype(float)
    mask = np.zeros_like(lim, dtype=bool)
    drp = dr//3
    mask[drp:-drp, drp:-drp] = True
    mask2 = (all_dists[sindices[i]] > drp)[500-sdp[i]-dr:500+sdp[i]+dr, 500-sdp[i]-dr:500+sdp[i]+dr]
    mask = np.logical_and(mask, mask2)
    lim = gauss_homogenize2(lim, mask, sigma=sigma/1)
    lim -= np.nanmean(lim)
    lim = np.nan_to_num(lim)
    mask3 = np.zeros_like(lim, dtype=bool)
    mask3[dr:-dr,dr:-dr] = True
    mask = mask3 
    gs = []
    for pk in pks:
        g = GPA.wfr2_grad_opt(lim, sigma, pk[0]*1., pk[1]*1., kw=kw, kstep=kstep)
        gs.append(g)
    
    wadvs = []
    for l in range(3):
        phase = np.angle(gs[l]['lockin'])[dr:-dr, dr:-dr]
       
        gphase = np.moveaxis(gs[l]['grad'],-1,0)
        w = gphase/2/np.pi + pks[l,:, None, None]
        wadvs.append(w)
    wadvs = np.stack(wadvs)
    
    wxs = np.concatenate([wadvs[:,0], -wadvs[:,0]]) / NMPERPIXEL
    wys = np.concatenate([wadvs[:,1], -wadvs[:,1]]) / NMPERPIXEL
    
    fig, ax = plt.subplots(figsize=[5,5])
    p,_ =  per((lim-lim.mean())[dr:-dr,dr:-dr], inverse_dft=False)
    fftim = np.abs(np.fft.fftshift(p))
    fftplot(fftim, ax=ax, pcolormesh=False, vmax=np.quantile(fftim, 0.999),
                 vmin=np.quantile(fftim, 0.01), cmap='cet_fire_r', d=NMPERPIXEL, interpolation='none')
    ax.scatter(*(pks/NMPERPIXEL).T+0.5/NMPERPIXEL/s, 
               edgecolor='C0', alpha=0.5, s=50, marker='o', color='none')
    ax.scatter(*(-pks/NMPERPIXEL).T+0.5/NMPERPIXEL/s, 
               edgecolor='C0', alpha=0.5, s=50, marker='o', color='none')
    ax.set_xlim([-0.03,0.03])
    ax.set_ylim([-0.03,0.03])

    
    fig, ax = plt.subplots(figsize=[5,5])
    im = ax.hist2d(wxs.ravel()[np.stack([mask]*6).ravel()], 
               wys.ravel()[np.stack([mask]*6).ravel()], 
               bins=500, cmap='cet_fire_r',vmax=100)
    ax.set_aspect('equal')
    ax.set_xlim([-0.03,0.03])
    ax.set_ylim([-0.03,0.03])

    
    xx, yy = np.meshgrid(np.arange(lim.shape[0]), np.arange(lim.shape[1]), indexing='ij')

    fig, axs = plt.subplots(ncols=3, nrows=3, figsize=[8,8], constrained_layout=True)
    axs = axs.T
    for j in range(len(gs)):
        dx =  pks[j][0] - gs[j]['w'][0]
        dy = pks[j][1] - gs[j]['w'][1]
        kx,ky = pks[j]
        ref = np.exp(np.pi*2*1j*(xx*kx + yy*ky))
        im = axs[0,j].imshow(ref.real[dr:-dr,dr:-dr].T, origin='lower')
        im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
        phase = np.exp(1j*np.angle(gs[j]['lockin']))
        recon = ref/phase
        recon = recon.real
        im = axs[2,j].imshow(recon[dr:-dr,dr:-dr].T, origin='lower')
        im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
        indicate_k(pks, j, ax=axs[1,j])
        im = axs[1,j].imshow(trim_nans(np.where(mask, np.angle(phase), np.nan)).T, 
                             cmap='twilight', interpolation='none', origin='lower')
        im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
   

    
    phases = np.stack([np.angle(g['lockin']) for g in gs])
    maskzero = 0.000001
    weights = np.stack([np.abs(g['lockin']) for g in gs])*(mask+maskzero)
    grads = np.stack([g['grad'] for g in gs])
    
    unew = GPA.reconstruct_u_inv_from_phases(pks, phases, weights)
    uwi = GPA.invert_u_overlap(unew)

    plt.figure(figsize=[5,5])
    im = plt.imshow(image[x, y].astype(float)[dr:-dr,dr:-dr].T,
                    cmap='gray', origin='lower',
                    vmin=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.002),
                    vmax=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.999),
                    ) #
    
    a=10
    plt.quiver((xx[dr:-dr:a,dr:-dr:a]-dr)*NMPERPIXEL/1000, 
               (yy[dr:-dr:a,dr:-dr:a]-dr)*NMPERPIXEL/1000, 
               *-uwi[:,dr:-dr:a,dr:-dr:a],
           angles='xy', scale_units='xy', headwidth=5, ec='white', linewidth=0.5, pivot='tip',
           scale=1000/NMPERPIXEL, color=f'C2')
    im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
    plt.xlabel('μm')

    
    plt.figure(figsize=[5,5])
    im = plt.imshow(image[x, y].astype(float)[dr:-dr,dr:-dr].T,
                    cmap='gray', origin='lower',
                    vmin=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.002),
                    vmax=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.999),
                    ) #
    im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)

    props = pe.moire_props_from_phasegradient(pks, grads, weights, nmperpixel=3.7)
    
    r = 1
    labels = [r'$\xi[^\circ]$', r'$\psi[^\circ]$', r'$\theta^*[^\circ]$', '$\kappa []$']
    cmaps = ['plasma', 'twilight', None, 'cet_fire']
    fig,ax = plt.subplots(ncols=2, nrows=4, figsize=[4.5,8], gridspec_kw={'width_ratios': [1.2,1]},
                         )
    ax = ax.T.flat
    props = np.stack([p[dr:-dr,dr:-dr] for p in props])
    print(sindices[i])
    for j in range(4):
        if j == 1:
            vmax=180
            vmin=0
            im = ax[j].imshow(props[j].T,
                              vmax=vmax,
                              vmin=vmin, 
                              cmap=cmaps[j], interpolation='nearest', origin="lower")
            
        elif j == 0:
            vmax=np.nanquantile(props[j][r:-r,r:-r], 0.99)
            vmin=np.nanquantile(props[j][r:-r,r:-r], 0.01)
            im = ax[j].imshow(props[j].T, cmap=cmaps[j], interpolation='nearest', origin="lower",
                              #vmax=48, vmin=33,
                              #vmax=np.nanquantile(props[j][r:-r,r:-r], 0.99),
                              #vmin=np.nanquantile(props[j][r:-r,r:-r], 0.01)
                              vmax=vmax,
                              vmin=vmin, 
                             )
        elif j == 2:
            vmax=np.nanquantile(props[j][r:-r,r:-r], 0.99)
            vmin=np.nanquantile(props[j][r:-r,r:-r], 0.01)
            im = ax[j].imshow(props[j].T, cmap=cmaps[j], interpolation='nearest', origin="lower",
                              vmax=vmax,
                              vmin=vmin, 
                              #vmax=np.nanquantile(props[j][r:-r,r:-r], 0.99),
                              #vmin=np.nanquantile(props[i][r:-r,r:-r], 0.01),
                              #vmin=0.245
                             )
            #CS = ax[i].contour(np.clip(props[i], *np.nanquantile(props[i],[0.1,0.9])).T, colors='black', alpha=0.3, origin="lower")
        else:
            vmax=np.nanquantile(props[j][r:-r,r:-r], 0.999)
            vmin=np.minimum(1, props[j][r:-r,r:-r].min())
            im = ax[j].imshow(props[j].T, cmap=cmaps[j], interpolation='nearest', origin="lower",
                              vmax=vmax, vmin=vmin,
                              #vmax=np.nanquantile(props[j][r:-r,r:-r], 0.999),
                              #vmin=1,
                             )
        im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1e3)
        ax[j].set_ylabel('μm')
        #ax[i].imshow(lim.T, cmap='gray', alpha=0.2, origin="lower")
        
        #cbar = plt.colorbar(im, ax=ax[j])
        print(i,j, vmin,vmax)
        nbins=300
        n, bins, patches = ax[j+4].hist(props[j].ravel(), bins=nbins, range=(vmin,vmax),
                     weights=np.full_like(props[j].ravel(), NMPERPIXEL**2/1e4), color=f'C2',
                    orientation='vertical')
        cm = get_cmap(cmaps[j])
        #if j == 1:
        #    ax[j]
        for v,p in enumerate(patches):
            plt.setp(p, 'facecolor', cm(v/nbins))
        ax[j+4].set_xlabel(labels[j])
        ax[j+4].yaxis.set_label_position("right")
        ax[j+4].tick_params(axis='y', which='both', labelleft=False, labelright=True)
        ax[j+4].yaxis.tick_right()
        ax[j].set_title(labels[j])
        ax[j+4].margins(x=0)
    ax[3].set_xlabel('μm')
    #ax[7].set_xlabel('Area [10$^4$ nm$^2$]')
    plt.tight_layout()

    #plt.figure()
    #plt.imshow(extract.calc_eps_from_phasegradient(pks, grads, weights, nmperpixel=3.7)[dr:-dr,dr:-dr].T,
    #          origin='lower', )
    #plt.colorbar()
    #plt.title('epsilon ()')
    plt.figure(figsize=[10,10])
    im = plt.imshow(image[x, y].astype(float)[dr:-dr,dr:-dr].T,
                    cmap='gray', origin='lower',
                    vmin=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.002),
                    vmax=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.999),
                    ) #

    angle = np.deg2rad(props[1])
    magnitude = props[3] #- 1
    kx = magnitude*np.cos(angle)
    ky = magnitude*np.sin(angle)
    a=15
    plt.quiver((xx[dr:-dr:a,dr:-dr:a]-dr)*NMPERPIXEL/1000, 
               (yy[dr:-dr:a,dr:-dr:a]-dr)*NMPERPIXEL/1000, 
               kx[::a, ::a],
               ky[::a, ::a],
               np.clip(magnitude[::a,::a]+1, 1, np.quantile(magnitude, 0.98)+1),
               #np.clip(props[2][::a,::a], np.quantile(props[2], 0.02), np.quantile(props[2], 0.98)),
               cmap='cet_fire',
               #*-np.swapaxes(uwi[:,dr:-dr:a,dr:-dr:a], -1, -2), 
               angles='xy', scale_units='xy', #ec='white', 
               width=0.005, pivot='mid', ec='black', linewidth=0.5,
               #color=f'C1', 
               headlength=0., headwidth=0.5, headaxislength=0)
    im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
    plt.title('$\kappa$ []')

# %% [markdown]
# ## Old GPA toppart

# %%
glasbey = plt.get_cmap('cet_glasbey_dark')(np.linspace(0,1,255))
import matplotlib as mpl

# %%
i = 5
coord = coordsl[i]
pks = kvec_lists[sindices[i]][:,::-1]
kw = np.linalg.norm(pks, axis=1).mean()/4
kstep = kw/3
sigma = max(10, 0.9/np.linalg.norm(pks,axis=1).min())
dr = int(np.ceil(3*sigma))
x = slice(int(coord[0])-sdp[i]-dr, int(coord[0])+sdp[i]+dr)
y = slice(int(coord[1])-sdp[i]-dr, int(coord[1])+sdp[i]+dr)
lim = image[x, y].astype(float).T
mask = np.zeros_like(lim, dtype=bool)
drp = dr//3
mask[drp:-drp, drp:-drp] = True
mask2 = (all_dists[sindices[i]] > drp)[500-sdp[i]-dr:500+sdp[i]+dr, 500-sdp[i]-dr:500+sdp[i]+dr]
mask = np.logical_and(mask, mask2)
lim = gauss_homogenize2(lim, mask, sigma=sigma/1)
lim -= np.nanmean(lim)
lim = np.nan_to_num(lim)
mask3 = np.zeros_like(lim, dtype=bool)
mask3[dr:-dr,dr:-dr] = True
mask = mask3 
gs = []
for pk in pks:
    g = GPA.wfr2_grad_opt(lim, sigma, pk[0]*1., pk[1]*1., kw=kw, kstep=kstep)
    gs.append(g)
    
    
phases = np.stack([np.angle(g['lockin']) for g in gs])
maskzero = 0.000001
weights = np.stack([np.abs(g['lockin']) for g in gs])*(mask+maskzero)
grads = np.stack([g['grad'] for g in gs])

unew = GPA.reconstruct_u_inv_from_phases(pks, phases, weights)
uwi = GPA.invert_u_overlap(unew)

# %%
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=[12,3.5], 
                        constrained_layout=True, sharex=True, sharey=True)
fig2, axs2 = plt.subplots(ncols=2, nrows=1, figsize=[12,4.8], 
                          constrained_layout=True)
j = 0
dx =  pks[j][0] - gs[j]['w'][0]
dy = pks[j][1] - gs[j]['w'][1]
kx, ky = pks[j]
xx, yy = np.meshgrid(np.arange(lim.shape[0]), np.arange(lim.shape[1]), indexing='ij')
ref = np.exp(np.pi*2*1j*(xx*kx + yy*ky))
#im = axs[0,0].imshow(ref.real[dr:-dr,dr:-dr].T, origin='lower')
im = axs[0].imshow(np.angle(ref)[dr:-dr,dr:-dr].T, origin='lower', 
                   cmap='twilight', 
                   interpolation='none')
im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
phase = np.exp(1j*np.angle(gs[j]['lockin']))
recon = ref / phase
recon = recon.real


p,_ =  per(lim-lim.mean(), inverse_dft=False)
fftim = np.abs(np.fft.fftshift(p))
axin = inset_axes(axs[0], width="30%", height="30%", loc=2)
#axin.tick_params(labelleft=False, labelbottom=False, direction='in', length=0)
fftplot(fftim, ax=axin, pcolormesh=False, 
        vmax=np.quantile(fftim, 0.998),
        vmin=np.quantile(fftim, 0.01), 
        cmap='cet_fire_r', d=NMPERPIXEL, interpolation='none')
axin.scatter(*(pks/NMPERPIXEL).T+0.5/NMPERPIXEL/s, edgecolor='black', alpha=0.7, s=70, marker='o', color='none')
axin.scatter(*(-pks/NMPERPIXEL).T+0.5/NMPERPIXEL/s, edgecolor='black', alpha=0.7, s=70, marker='o', color='none')
axin.scatter(pks[j, 0]/NMPERPIXEL+0.5/NMPERPIXEL/s, 
             pks[j, 1]/NMPERPIXEL+0.5/NMPERPIXEL/s, 
             color='blue', s=70, alpha=0.6)
#indicate_k(pks/NMPERPIXEL, j, ax=axin, inset=False, origin='lower', s=20)
rf = 0.04
axin.set_xlim([-rf,rf])
axin.set_ylim([-rf,rf])
axin.tick_params(labelleft=False, labelbottom=False, direction='in')

im = axs[1].imshow(trim_nans(np.where(mask, np.angle(phase), np.nan)).T, 
                   cmap='twilight', 
                   interpolation='none', origin='lower')
im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
im = axs[1].imshow(recon[dr:-dr,dr:-dr].T, origin='lower', 
                   cmap='gray', alpha=0.3, 
                   extent=im.get_extent())




im = axs[2].imshow(image[x, y].astype(float)[dr:-dr,dr:-dr],
                cmap='gray', origin='lower',
                vmin=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.002),
                vmax=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.999),
                ) #

a=20
axs[2].quiver((xx[dr:-dr:a,dr:-dr:a]-dr)*NMPERPIXEL/1000, 
              (yy[dr:-dr:a,dr:-dr:a]-dr)*NMPERPIXEL/1000, 
              *-uwi[:,dr:-dr:a,dr:-dr:a], 
              angles='xy', scale_units='xy', headwidth=5, ec=glasbey[2], linewidth=1, pivot='tip',
              scale=1000/NMPERPIXEL, color=glasbey[2])
im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
for ax in axs:
    ax.set_xlabel('μm')
axs[0].set_ylabel('μm')
axs[0].set_title('Reference phase')
axs[1].set_title('Phase difference')
axs[2].set_title('Deformation field')

for ax in axs2:
    ax.set_xlabel('μm')
    ax.set_ylabel('μm')

props = pe.calc_props_from_phasegradient(pks, grads, weights, nmperpixel=3.7)    
props = np.stack([p[dr:-dr,dr:-dr] for p in props])
im = axs2[0].imshow(allprops[1][0], origin="lower",
                  vmax=np.nanquantile(allprops[1][0], 0.9999),
                  vmin=0.235 #np.nanquantile(props[2], 0.01), 
                 )
im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
plt.colorbar(im, ax=axs2[0], label = '$\\theta^*$ ($^\circ$)')

im = axs2[1].imshow(image[x, y].astype(float)[dr:-dr,dr:-dr],
                    cmap='gray', origin='lower',
                    vmin=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.002),
                    vmax=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.999),
                    )

angle = np.deg2rad(props[1])
magnitude = (props[3] - 1)
kx = magnitude*np.cos(angle)
ky = magnitude*np.sin(angle)
a = 20
Q = axs2[1].quiver((xx[dr:-dr:a,dr:-dr:a]-dr)*NMPERPIXEL/1000, 
                   (yy[dr:-dr:a,dr:-dr:a]-dr)*NMPERPIXEL/1000, 
                   kx[::a, ::a],
                   ky[::a, ::a],
                   np.clip(magnitude[::a,::a]+1, 1, np.quantile(magnitude, 0.98)+1),
                   cmap='cet_fire',
                   norm=mpl.colors.Normalize(vmin=1),
                   angles='xy', scale_units='xy', 
                   width=0.008, pivot='mid',
                           ec='black', linewidth=0.5,
                   headlength=0., headwidth=0.5, headaxislength=0)
plt.colorbar(Q, ax=axs2[1], label='$\kappa$ ()', ticks=[1, 1.1, 1.2])
im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
axs2[0].set_title('Local twist angle')
axs2[1].set_title('Anisotropy')

for l, ax in zip('abc', axs):
    ax.set_title(l, loc='left', fontweight='bold')
for l, ax in zip('de', axs2):
    ax.set_title(l, loc='left', fontweight='bold')
    for axis in ['top','bottom','left','right']:
        #ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color(glasbey[2])
fig.savefig(os.path.join('figures', 'GPA_top_new.pdf'))
fig2.savefig(os.path.join('figures', 'GPA_center_new.pdf'))

# %% [markdown]
# ## New GPA top part

# %%
extent

# %%
fig, axs = plt.subplots(ncols=3, nrows=1, figsize=[12,3.5], 
                        constrained_layout=True, sharex=True, sharey=True)
fig2, axs2 = plt.subplots(ncols=2, nrows=1, figsize=[12,4.8], 
                          constrained_layout=True)
contours=True
j = 0
dx =  pks[j][0] - gs[j]['w'][0]
dy = pks[j][1] - gs[j]['w'][1]
kx, ky = pks[j]
xx, yy = np.meshgrid(np.arange(lim.shape[0]), np.arange(lim.shape[1]), indexing='ij')
ref = np.exp(np.pi*2*1j*(xx*kx + yy*ky))
#im = axs[0,0].imshow(ref.real[dr:-dr,dr:-dr].T, origin='lower')
im = axs[0].imshow(np.angle(ref)[dr:-dr,dr:-dr].T, origin='lower', 
                   cmap='twilight', 
                   interpolation='none')
im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
phase = np.exp(1j*np.angle(gs[j]['lockin']))
recon = ref / phase
recon = recon.real


p,_ =  per(lim-lim.mean(), inverse_dft=False)
fftim = np.abs(np.fft.fftshift(p))
axin = inset_axes(axs[0], width="30%", height="30%", loc=2)
#axin.tick_params(labelleft=False, labelbottom=False, direction='in', length=0)
fftplot(fftim, ax=axin, pcolormesh=False, 
        vmax=np.quantile(fftim, 0.998),
        vmin=np.quantile(fftim, 0.01), 
        cmap='cet_fire_r', d=NMPERPIXEL, interpolation='none')
axin.scatter(*(pks/NMPERPIXEL).T+0.5/NMPERPIXEL/s, edgecolor='black', alpha=0.7, s=70, marker='o', color='none')
axin.scatter(*(-pks/NMPERPIXEL).T+0.5/NMPERPIXEL/s, edgecolor='black', alpha=0.7, s=70, marker='o', color='none')
axin.scatter(pks[j, 0]/NMPERPIXEL+0.5/NMPERPIXEL/s, 
             pks[j, 1]/NMPERPIXEL+0.5/NMPERPIXEL/s, 
             color='blue', s=70, alpha=0.6)
#indicate_k(pks/NMPERPIXEL, j, ax=axin, inset=False, origin='lower', s=20)
rf = 0.04
axin.set_xlim([-rf,rf])
axin.set_ylim([-rf,rf])
axin.tick_params(labelleft=False, labelbottom=False, direction='in')

im = axs[1].imshow(trim_nans(np.where(mask, np.angle(phase), np.nan)).T, 
                   cmap='twilight', 
                   interpolation='none', origin='lower')
im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
im = axs[1].imshow(recon[dr:-dr,dr:-dr].T, origin='lower', 
                   cmap='gray', alpha=0.3, 
                   extent=im.get_extent())




im = axs[2].imshow(image[x, y].astype(float)[dr:-dr,dr:-dr],
                cmap='gray', origin='lower',
                vmin=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.002),
                vmax=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.999),
                ) #

a=20
axs[2].quiver((xx[dr:-dr:a,dr:-dr:a]-dr)*NMPERPIXEL/1000, 
              (yy[dr:-dr:a,dr:-dr:a]-dr)*NMPERPIXEL/1000, 
              *-uwi[:,dr:-dr:a,dr:-dr:a], 
              angles='xy', scale_units='xy', headwidth=5, ec=glasbey[2], linewidth=1, pivot='tip',
              scale=1000/NMPERPIXEL, color=glasbey[2])
extent = np.array(im.get_extent())*NMPERPIXEL / 1000
im.set_extent(extent)
for ax in axs:
    ax.set_xlabel('μm')
axs[0].set_ylabel('μm')
axs[0].set_title('Reference phase')
axs[1].set_title('Phase difference')
axs[2].set_title('Deformation field')

for ax in axs2:
    ax.set_xlabel('μm')
    ax.set_ylabel('μm')

props = pe.calc_props_from_phasegradient(pks, grads, weights, nmperpixel=3.7)    
props = np.stack([p[dr:-dr,dr:-dr] for p in props])
im = axs2[0].imshow(allprops[1][0], origin="lower",
                 vmax=np.nanquantile(allprops[1][0], 0.9999),
                 vmin=0.235 #np.nanquantile(props[2], 0.01), 
                )
cbar = plt.colorbar(im, ax=axs2[0], label = '$\\theta^*$ ($^\circ$)', shrink=0.94)
if contours:
    CF = axs2[0].contour(allprops[1][0], levels=np.linspace(0.225,0.275, 40),
                    origin="lower", 
                    extent=extent, colors='white', alpha=0.5, linewidth=0.5
                    )
im.set_extent(extent)
if contours: 
    cbar.add_lines(CF)


im = axs2[1].imshow(image[x, y].astype(float)[dr:-dr,dr:-dr],
                    cmap='gray', origin='lower',
                    vmin=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.002),
                    vmax=np.quantile(image[x, y].astype(float)[dr:-dr,dr:-dr], 0.999),
                    )

epscolor = allprops[1][2].T*100
angle = np.deg2rad(allprops[1][1]).T# % 180
magnitude = epscolor
#angle = np.deg2rad(props[1])
#magnitude = (props[3] - 1)
kx = magnitude*np.cos(angle)
ky = magnitude*np.sin(angle)
a = 20


cbar = plt.colorbar(Q, ax=axs2[1], label='$\epsilon$ (%)', shrink=0.94)#, ticks=[1, 1.1, 1.2])
if contours:
    CF = axs2[1].contour(allprops[1][2]*100, levels=np.linspace(0.0,0.13, 52),
                    origin="lower", 
                    extent=extent, colors='k', alpha=0.5, linewidth=0.5
                    )
Q = axs2[1].quiver((xx[dr:-dr:a,dr:-dr:a]-dr)*NMPERPIXEL/1000, 
               (yy[dr:-dr:a,dr:-dr:a]-dr)*NMPERPIXEL/1000, 
               kx[::a, ::a],
               ky[::a, ::a],
               np.clip(epscolor[::a,::a], 0, np.quantile(epscolor, 0.985)),
               cmap='magma',
               norm=mpl.colors.Normalize(vmin=0),
               angles='xy', scale_units='xy', 
               width=0.008, pivot='mid',
                       ec='black', linewidth=0.5,
               headlength=0., headwidth=0.5, headaxislength=0)
im.set_extent(np.array(im.get_extent()) * NMPERPIXEL/1000)
if contours: 
    cbar.add_lines(CF)
axs2[0].set_title('Local twist angle')
axs2[1].set_title('Heterostrain')
#axs2[1].set_title('Anisotropy')

for l, ax in zip('abc', axs):
    ax.set_title(l, loc='left', fontweight='bold')
for l, ax in zip('de', axs2):
    ax.set_title(l, loc='left', fontweight='bold')
    for axis in ['top','bottom','left','right']:
        #ax.spines[axis].set_linewidth(1)
        ax.spines[axis].set_color(glasbey[2])

fig2.savefig(os.path.join('figures', 'GPA_center_eps_contour.pdf'))

# %%
J = pe.phasegradient2J(pks, grads, weights, nmperpixel=NMPERPIXEL, iso_ref=False, sort=1)
stride = 1
res, refest = pe.Kerelsky_J(J[::stride,::stride], pks, sort=1, nmperpixel=NMPERPIXEL)
res = res.compute()
res[...,1] = res[...,1] % 180
res[...,3] = res[...,3] % 180
props = np.moveaxis(res, -1,0)
props.shape

# %%
histfig, histaxs = plt.subplots(ncols=2, figsize=[12,5])
allprops = []
ayes = [3, 5, 7, 8, 9, 10, 11]
for i in ayes:
    coord = coordsl[i]
    pks = kvec_lists[sindices[i]]
    kw = np.linalg.norm(pks, axis=1).mean()/4
    kstep = kw/3
    sigma = max(10, 0.9/np.linalg.norm(pks,axis=1).min())
    print(sigma, 1/np.linalg.norm(pks,axis=1).mean())
    dr = int(np.ceil(3*sigma))
    x = slice(int(coord[0])-sdp[i]-dr, int(coord[0])+sdp[i]+dr)
    y = slice(int(coord[1])-sdp[i]-dr, int(coord[1])+sdp[i]+dr)
    lim = image[x, y].astype(float)
    mask = np.zeros_like(lim, dtype=bool)
    drp = dr//3
    mask[drp:-drp, drp:-drp] = True
    mask2 = (all_dists[sindices[i]] > drp)[500-sdp[i]-dr:500+sdp[i]+dr, 500-sdp[i]-dr:500+sdp[i]+dr]
    mask = np.logical_and(mask, mask2)
    lim = gauss_homogenize2(lim, mask, sigma=sigma/1)
    lim -= np.nanmean(lim)
    lim = np.nan_to_num(lim)
    mask3 = np.zeros_like(lim, dtype=bool)
    mask3[dr:-dr,dr:-dr] = True
    mask = mask3 
    gs = []
    for pk in pks:
        g = GPA.wfr2_grad_opt(lim, sigma, pk[0]*1., pk[1]*1., kw=kw, kstep=kstep)
        gs.append(g)
    
    
    xx, yy = np.meshgrid(np.arange(lim.shape[0]), np.arange(lim.shape[1]), indexing='ij')
    
    phases = np.stack([np.angle(g['lockin']) for g in gs])
    maskzero = 0.000001
    weights = np.stack([np.abs(g['lockin']) for g in gs])*(mask+maskzero)
    grads = np.stack([g['grad'] for g in gs])
    
    #unew = GPA.reconstruct_u_inv_from_phases(pks, phases, weights)
    #uwi = GPA.invert_u_overlap(unew)
    J = pe.phasegradient2J(pks, grads, weights, nmperpixel=NMPERPIXEL, iso_ref=False, sort=1)
    stride = 1
    res, refest = pe.Kerelsky_J(J[::stride,::stride], pks, sort=1, nmperpixel=NMPERPIXEL)
    res = res.compute()
    res[...,1] = res[...,1] % 180
    res[...,3] = res[...,3] % 180
    props = np.moveaxis(res, -1,0)
    #props = pe.calc_props_from_phasegradient(pks, grads, weights, nmperpixel=3.7)
    #props[3] = extract.calc_eps_from_phasegradient(pks, grads, weights, nmperpixel=3.7)*100
    props = np.stack([p[dr//stride:-dr//stride,dr//stride:-dr//stride] for p in props])
    allprops.append(props)
    histaxs[0].hist(props[0].ravel(), bins=50//stride, alpha=0.7, color=glasbey[i],
                    label=f'{props[0].mean():.3f}° ($\pm ${props[0].std():.3f}°)',
                    weights=np.full_like(props[0].ravel(), (NMPERPIXEL/1e3)**2 / ((props[0].max()-props[2].min())/50*stride))
                   )
    histaxs[1].hist(props[2].ravel(), bins=100//stride, alpha=0.3, color=glasbey[i], 
                    label=f'{props[2].mean():.3f}  ($\pm ${props[2].std():.3f})',
                    weights=np.full_like(props[2].ravel(), (NMPERPIXEL/1e3)**2 / ((props[2].max()-props[2].min())/100*stride))
                   )

histaxs[0].legend()
histaxs[0].set_xlabel('twist angle ($^\circ$)')
histaxs[1].legend()
histaxs[1].set_xlabel('ϵ ()')
plt.tight_layout()
for l, ax in zip('fg', histaxs):
    ax.set_title(l, loc='left', fontweight='bold')

# %%
mpl

# %%
angles = [p[0].ravel() for p in allprops]
epsilons = [100*p[2].ravel() for p in allprops]
histfig, histaxs = plt.subplots(ncols=2, figsize=[12,3.5])
# Hacking colors together to make consistent with  Fig 1 todo: fix SI as well
colors = glasbey[ayes]
colors[1] = glasbey[2]
colors[0] = glasbey[1]
colors[4] = glasbey[3]
colors[5] = glasbey[4]
histaxs[0].hist(angles, bins=100, #alpha=0.5,
                color=colors, stacked=True,
                #histtype='step', fill=True,
                label=[f'{x.mean():.3f}° ($\pm ${x.std():.3f}°)' for x in angles],
                weights=[np.full_like(p, (NMPERPIXEL/1e3)**2) for p in angles]
               )
# histaxs[0].hist(angles, bins=100, #alpha=0.7, 
#                 color=colors, #stacked=True,
#                 histtype='step', fill=False,
#                 weights=[np.full_like(p, (NMPERPIXEL/1e3)**2) for p in angles]
#                )
histaxs[1].hist(epsilons[::-1], bins=100, #alpha=0.5, 
                color=colors[::-1], stacked=True, 
                label=[f'{x.mean():.3f}%  ($\pm ${x.std():.3f}%)' for x in epsilons[::-1]],
                weights=[np.full_like(p, (NMPERPIXEL/1e3)**2) for p in epsilons[::-1]]
               )
# histaxs[1].hist(epsilons[::-1], bins=100, #alpha=0.7, 
#                 color=colors[::-1], #stacked=True, 
#                 fill=False, histtype='step',
#                 weights=[np.full_like(p, (NMPERPIXEL/1e3)**2) for p in epsilons[::-1]]
#                )

lgd = histaxs[0].legend(ncol=2,
                  )
lgd.set_title('Mean twist angle ($\pm $s.d.)', prop=mpl.font_manager.FontProperties(weight='bold'))
histaxs[0].set_xlabel('twist angle $\\theta^*$ ($^\circ$)')
handles, labels = histaxs[1].get_legend_handles_labels()
lgd = histaxs[1].legend(handles[::-1], labels[::-1], 
                  ncol=2,
                 )
lgd.set_title('Mean strain ($\pm $s.d.)', prop=mpl.font_manager.FontProperties(weight='bold'))
histaxs[1].set_xlabel('Heterostrain $\\epsilon$ (%)')
histaxs[1].set_xlim(0, None)
for l, ax in zip('fg', histaxs):
    ax.set_title(l, loc='left', fontweight='bold')
    ax.set_ylabel('Area (μm$^2$)')
plt.tight_layout(pad=0.1)
histfig.savefig(os.path.join('figures', 'GPA_bottom_new.pdf'))

# %%
np.concatenate([[np.max(an[0])], np.max(np.diff(an, axis=0), axis=1)])

# %%
plt.plot(np.diff(an, axis=0).T)
plt.plot(an[0])

# %%
histfig, histaxs = plt.subplots(ncols=2, figsize=[12,5])
allprops = []
ayes = [3, 5, 7, 8, 9, 10, 11]
for i in ayes:
    coord = coordsl[i]
    pks = kvec_lists[sindices[i]]
    kw = np.linalg.norm(pks, axis=1).mean()/4
    kstep = kw/3
    sigma = max(10, 0.9/np.linalg.norm(pks,axis=1).min())
    print(sigma, 1/np.linalg.norm(pks,axis=1).mean())
    dr = int(np.ceil(3*sigma))
    x = slice(int(coord[0])-sdp[i]-dr, int(coord[0])+sdp[i]+dr)
    y = slice(int(coord[1])-sdp[i]-dr, int(coord[1])+sdp[i]+dr)
    lim = image[x, y].astype(np.float)
    mask = np.zeros_like(lim, dtype=bool)
    drp = dr//3
    mask[drp:-drp, drp:-drp] = True
    mask2 = (all_dists[sindices[i]] > drp)[500-sdp[i]-dr:500+sdp[i]+dr, 500-sdp[i]-dr:500+sdp[i]+dr]
    mask = np.logical_and(mask, mask2)
    lim = gauss_homogenize2(lim, mask, sigma=sigma/1)
    lim -= np.nanmean(lim)
    lim = np.nan_to_num(lim)
    mask3 = np.zeros_like(lim, dtype=bool)
    mask3[dr:-dr,dr:-dr] = True
    mask = mask3 
    gs = []
    for pk in pks:
        g = GPA.wfr2_grad_opt(lim, sigma, pk[0]*1., pk[1]*1., kw=kw, kstep=kstep)
        gs.append(g)
    
    
    xx, yy = np.meshgrid(np.arange(lim.shape[0]), np.arange(lim.shape[1]), indexing='ij')
    
    phases = np.stack([np.angle(g['lockin']) for g in gs])
    maskzero = 0.000001
    weights = np.stack([np.abs(g['lockin']) for g in gs])*(mask+maskzero)
    grads = np.stack([g['grad'] for g in gs])
    
    #unew = GPA.reconstruct_u_inv_from_phases(pks, phases, weights)
    #uwi = GPA.invert_u_overlap(unew)

    props = pe.calc_props_from_phasegradient(pks, grads, weights, nmperpixel=3.7)
    #props[3] = extract.calc_eps_from_phasegradient(pks, grads, weights, nmperpixel=3.7)*100
    props = np.stack([p[dr:-dr,dr:-dr] for p in props])
    allprops.append(props)
    histaxs[0].hist(props[2].ravel(), bins=50, alpha=0.7, color=glasbey[i],
                    label=f'{props[2].mean():.3f}° ($\pm ${props[2].std():.3f}°)',
                    weights=np.full_like(props[2].ravel(), (NMPERPIXEL/1e3)**2 / ((props[2].max()-props[2].min())/50))
                   )
    histaxs[1].hist(props[3].ravel(), bins=100, alpha=0.3, color=glasbey[i], 
                    label=f'{props[3].mean():.3f}  ($\pm ${props[3].std():.3f})',
                    weights=np.full_like(props[3].ravel(), (NMPERPIXEL/1e3)**2 / ((props[3].max()-props[3].min())/200))
                   )

histaxs[0].legend()
histaxs[0].set_xlabel('twist angle ($^\circ$)')
histaxs[1].legend()
histaxs[1].set_xlabel('κ ()')
plt.tight_layout()
for l, ax in zip('fg', histaxs):
    ax.set_title(l, loc='left', fontweight='bold')

# %%
angles = [p[2].ravel() for p in allprops]
kappas = [p[3].ravel() for p in allprops]

# %%
histfig, histaxs = plt.subplots(ncols=2, figsize=[12,3.5])
# Hacking colors together to make consistent with  Fig 1 todo: fix SI as well
colors = glasbey[ayes]
colors[1] = glasbey[2]
colors[0] = glasbey[1]
colors[4] = glasbey[3]
colors[5] = glasbey[4]
histaxs[0].hist(angles, bins=100, alpha=0.7, color=colors, stacked=True,
                label=[f'{x.mean():.3f}° ($\pm ${x.std():.3f}°)' for x in angles],
                weights=[np.full_like(p, (NMPERPIXEL/1e3)**2) for p in angles]
               )
histaxs[1].hist(kappas[::-1], bins=100, alpha=0.7, color=colors[::-1], stacked=True,
                label=[f'{x.mean():.3f}  ($\pm ${x.std():.3f})' for x in kappas[::-1]],
                #weights=[np.full_like(p, (NMPERPIXEL/1e3)**2 / ((p.max()-p.min())/100)) for p in kappas[::-1]]
                weights=[np.full_like(p, (NMPERPIXEL/1e3)**2) for p in kappas[::-1]]
               )

histaxs[0].legend(fontsize='small', ncol=2)
histaxs[0].set_xlabel('twist angle ($^\circ$)')
handles, labels = histaxs[1].get_legend_handles_labels()
#axs.legend(handles[::-1], labels[::-1], title='graphene\nlayers', loc='upper right', numpoints=3, handlelength=0.1)
histaxs[1].legend(handles[::-1], labels[::-1], fontsize='small', ncol=2)
histaxs[1].set_xlabel('κ ()')
histaxs[1].set_xlim(1., 1.25)
for l, ax in zip('fg', histaxs):
    ax.set_title(l, loc='left', fontweight='bold')
    ax.set_ylabel('Area (μm$^2$)')
plt.tight_layout()
histfig.savefig(os.path.join('figures', 'GPA_bottom.pdf'))
