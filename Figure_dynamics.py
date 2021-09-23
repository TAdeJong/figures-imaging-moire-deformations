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
#     display_name: Python [conda env:pyL5]
#     language: python
#     name: conda-env-pyL5-py
# ---

# %%
import numpy as np
import scipy.ndimage as ndi
import matplotlib
import matplotlib.pyplot as plt
import dask.array as da
#from dask.distributed import Client, LocalCluster
import time
import os

from pyL5.lib.analysis.container import Container
from ipywidgets import interact, interactive, fixed, interact_manual
import ipywidgets as widgets
#import dask_image

import colorcet
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from dateutil.parser import parse

# %%
import pyGPA.geometric_phase_analysis as GPA
import pyGPA


# %%
folder = r'/mnt/storage-linux/speeldata/20200716-XTBLG02'
name = '20200717_104948_3.5um_484.1_movement'

container = Container(os.path.join(folder,name+'.nlp'))

data = container.getStack('driftcorrected').getDaskArray().astype(np.float)


# %%
def plot_stack2(images, n):
    """Plot the n-th image from a stack of n images.
    For interactive use with ipython widgets"""
    im = images[n, :, :]
    plt.figure(figsize=[12,12])
    plt.imshow(im.T, cmap='gray')#, vmax=im.mean()*2)
    plt.show()

ndatag = da.from_zarr(os.path.join(folder, name, f'gausshomogenize50.zarr'))
ndata = ndi.filters.gaussian_filter1d(ndatag, sigma=1,
                                      axis=0, order=0)
interactive(lambda n: plot_stack2(ndata, n), 
            n=widgets.IntSlider(1, 0, ndata.shape[0]-1, 1, continuous_update=False)
           )

# %%
cropped = pyGPA.imagetools.trim_nans2(ndata[0])
pks,_ = GPA.extract_primary_ks(cropped, plot=True, pix_norm_range=(5,100))

# %%
from skimage.morphology import binary_erosion, selem, disk

# %%
1/(kw*4)

# %%
kw = np.linalg.norm(pks,axis=1).mean()/4
sigma = 20
dr = 2*sigma
kstep = kw/6
mask = ~np.isnan(ndata[0])
mask = binary_erosion(np.pad(mask,1), selem=disk(dr))[1:-1,1:-1]
maskzero = 0.000001
uws = []
uws_old = []
ks = [0,70,178]
#ks = [0,120,146]
for i in ks:
    print(i, end=', ')
    deformed = ndata[i]
    deformed = deformed - np.nanmean(deformed)
    deformed = np.nan_to_num(deformed)
    plt.figure()
    plt.imshow(deformed.T)
    gs = []
    for pk in pks:
        g = GPA.wfr2_grad(deformed, sigma, 
                          pk[0], pk[1], kw=kw, 
                          kstep=kstep, grad='diff')
        gs.append(g)
        plt.figure()
        plt.imshow(np.angle(g['lockin']).T)
    #phases = np.angle(gs)
    phases = np.stack([np.angle(g['lockin']) for g in gs])
    grads = np.stack([g['grad'] for g in gs])
    weights = np.abs([g['lockin'] for g in gs])*(mask+maskzero)
    u = GPA.reconstruct_u_inv_from_phases(pks, phases, weights, weighted_unwrap=True)
    u_grad = GPA.reconstruct_u_inv_from_phases(pks, -2*np.pi*grads, weights, pre_diff=True)
    #u_inv = GPA.invert_u_overlap(unew, iters=3)
    uws.append(u_grad)
    uws_old.append(u)

# %%
uws = np.array(uws)
uws.mean(axis=(-1,-2))

# %%
meanshifts = np.nanmean(np.where(mask, uws, np.nan), axis=(-1,-2))
meanshifts - meanshifts[0]

# %%
import pyGPA.property_extract as pe
from dask.distributed import Client, LocalCluster
cluster = LocalCluster(n_workers=16, threads_per_worker=2, memory_limit='1GB')  
client = Client(cluster)
client

# %%
from importlib import reload
reload(pe)

# %%
NMPERPIXEL=2.23
props = []
for u in uws:
    J = pe.u2J(u, NMPERPIXEL)
    prop = pe.Kerelsky_J(J, pks, nmperpixel=NMPERPIXEL, sort=1)[0]
    props.append(prop)

# %%
import dask.array as da

# %%
props = da.compute(props)[0]

# %%
props = np.array(props)
props.shape

# %%
xslice, yslice = slice(250, 655), slice(550,910)
plt.figure(figsize=[15,12])
plt.imshow((props-props[0])[2][xslice, yslice,2].T*100, vmax=0.01, vmin=-0.01, cmap='PiYG')#, vmax=0.3, vmin=-0.3,cmap='Spectral')
plt.colorbar()

# %%
xslice, yslice = slice(250, 655), slice(550,910)
plt.figure(figsize=[15,12])
plt.imshow((props-props[0])[2][200:1000,200:800,2].T*100, vmax=0.02, vmin=-0.02, cmap='bwr')#, vmax=0.3, vmin=-0.3,cmap='Spectral')
plt.colorbar()

# %%
(props-props[0])[2][200:800,200:800,0].std()*100

# %%
plt.hist((props-props[0])[2][200:1000,200:800,0].ravel(), bins=100, range=(-0.005,0.005));

# %%
plt.figure(figsize=[15,12])
plt.imshow((props-props[0])[2][xslice,yslice,0].T, vmax=0.005, vmin=-0.005, cmap='bwr')
plt.colorbar()

# %%
plt.figure(figsize=[15,12])
plt.imshow((props-props[0])[2][200:1000,200:800,0].T, vmax=0.005, vmin=-0.005, cmap='bwr')
plt.colorbar()

# %%
xslice, yslice = slice(250, 655), slice(550,910)
plt.figure(figsize=[15,12])
plt.imshow((props)[1][...,2].T*100, 
           vmax=np.quantile((props)[1][...,2],0.99)*100,
          vmin=np.quantile((props)[1][...,2],0.01)*100
          )
plt.colorbar()

# %%
# acont = Container(os.path.join(folder, name))

dts = [parse(x) - parse(acont['TTT'][0]) for x in acont["TTT"]]
clim = np.nanquantile(ndata-ndata[0], [0.9999, 0.0001])
clim=[0.05,-0.05]

# %%
np.mgrid[0:(uws[0].shape[0]), 0:(uws[0].shape[1])].shape

# %%
clim2 = np.nanquantile(ndatag, [0.001, 0.999])

# %%
import matplotlib.animation as animation

fig, ax = plt.subplots(ncols=3, figsize=[12,5], constrained_layout=True, sharex=True)
im = [None]*3
#k = 123
k = 178
xx,yy = np.mgrid[0:ndata[k].shape[0],0:ndata[k].shape[1]]*NMPERPIXEL
a = 20
im[0] = ax[1].imshow(ndatag[k].T, cmap='gray', vmax=clim2[1], vmin=clim2[0])
cbar = plt.colorbar(im[0], ax=ax[1], orientation='horizontal', shrink=1)
cbar.ax.set_xlabel('normalized intensity\n (arb.units.)')
im[1] = ax[0].imshow(100*(ndata[k] - ndata[0]).T, cmap='RdBu_r', vmax=100*clim[0], vmin=100*clim[1])
ax[1].set_title(f"t= {dts[k]}")
cbar1 = plt.colorbar(im[1], ax=ax[0], orientation='horizontal', shrink=1)
cbar1.ax.set_xlabel('difference (%)')
uwdiff = (uws[2]-uws[0])*NMPERPIXEL
uwdiff = np.where(mask, uwdiff, np.nan)
im[2] = ax[2].imshow(np.linalg.norm(uwdiff, axis=0).T, cmap='inferno', vmax=7, vmin=0)
a = 30
arrow_scale=12
Q = ax[2].quiver(xx[::a,::a], yy[::a,::a], *uwdiff[:,::a,::a], 
                  color='white',angles='xy', scale_units='xy', scale=1/arrow_scale)
cbar2 = plt.colorbar(im[2], ax=ax[2], orientation='horizontal', shrink=1)
cbar2.ax.set_xlabel(f'moiré displacement (nm)\n (Arrows scaled by factor {arrow_scale})')
for imi in im:
    imi.set_extent(np.array(imi.get_extent())*NMPERPIXEL)
for axi in ax[1:]:
    axi.tick_params(axis='y', which='both', labelleft=False, labelright=False)


# %%
def update(i):
    im[0].set_data(ndatag[i].T)
    im[1].set_data(100*(ndata[i] - ndata[0]).T)
    ax[1].set_title(f"t= {dts[i]}")
    
    deformed = ndata[i]
    deformed = deformed - np.nanmean(deformed)
    deformed = np.nan_to_num(deformed)
    gs = []
    for pk in pks:
        g = GPA.wfr2_only_lockin_vec(deformed, sigma, pk[0]*1., pk[1]*1., kw=kw, kstep=kstep)
        gs.append(g)
    gs = np.stack(gs)
    phases = np.angle(gs)
    weights = np.abs(gs)*(mask+maskzero)
    unew = GPA.reconstruct_u_inv_from_phases(pks, phases, weights, weighted_unwrap=True)
    uwdiff = (unew-uws[0])*NMPERPIXEL
    uwdiff = np.where(mask, uwdiff, np.nan)
    im[2].set_data(np.linalg.norm(uwdiff, axis=0).T)
    Q.set_UVC(*uwdiff[:,::a,::a])
    print(i)
    return im + [Q]

ani = animation.FuncAnimation(fig, update, interval=80, 
                              blit=True, save_count=ndata.shape[0])
ani.save(os.path.join(f'deformationcum_t_sigma=1.mp4'), 
         dpi=300)

# %%
900/NMPERPIXEL, 800/NMPERPIXEL

# %%
import matplotlib.ticker as ticker
transpose = True
NMPERPIXEL=2.23
#xslice, yslice = slice(250,600), slice(550,900)
xslice, yslice = slice(250, 655), slice(550,910)
xx,yy = np.mgrid[xslice, yslice]
xx -=250
yy -=550
xx = xx*NMPERPIXEL
yy = yy*NMPERPIXEL

if transpose:
    fig = plt.figure(figsize=[8.5,6.5], constrained_layout=True)
    gs = fig.add_gridspec(1,2, width_ratios=[1.2,3])
    gs0 = gs[0].subgridspec(3, 1)
    gs1 = gs[1].subgridspec(2, 2)
    axs = gs1.subplots(sharex=True, sharey=True).T
else:
    fig = plt.figure(figsize=[6.5,8], constrained_layout=True)
    gs = fig.add_gridspec(2,1, height_ratios=[1.3,3])
    gs0 = gs[0].subgridspec(1, 3)
    gs1 = gs[1].subgridspec(2, 2)
    axs = gs1.subplots(sharex=True, sharey=True)

clim = np.nanquantile(ndata-ndata[0], [0.9999, 0.0001])
clim=[0.05,-0.05]

oaxs = gs0.subplots(sharey=True)
for a in oaxs:
    a.set_xticklabels([])
#axs = gs1.subplots(sharex=True, sharey=True)
for i, k in enumerate(ks):
    im = oaxs[i].imshow(ndatag[k, xslice, yslice].T, cmap='gray')
    im.set_extent(np.array(im.get_extent())*NMPERPIXEL)
    oaxs[i].xaxis.set_minor_locator(ticker.MultipleLocator(100))
    oaxs[i].yaxis.set_minor_locator(ticker.MultipleLocator(100))
    oaxs[i].yaxis.set_major_locator(ticker.MultipleLocator(400))
    oaxs[i].xaxis.set_major_locator(ticker.MultipleLocator(200))
    oaxs[i].grid(color='green', alpha=0.3, which='both')
    oaxs[i].set_title(f"t = {dts[k].total_seconds():.0f}s", fontsize='medium')
    oaxs[i].set_ylabel('nm')
    if i==2:
        cbar = plt.colorbar(im, ax=oaxs, orientation='horizontal', shrink=0.85)
        cbar.ax.set_xlabel('normalized intensity\n (arb.units.)')
    if i > 0:
        im = axs[0,i-1].imshow(100*(ndata-ndata[0])[k, xslice, yslice].T, cmap='RdBu_r', vmax=100*clim[0], vmin=100*clim[1])
        im.set_extent(np.array(im.get_extent())*NMPERPIXEL)
        axs[0, i-1].xaxis.set_minor_locator(ticker.MultipleLocator(100))
        axs[0, i-1].yaxis.set_minor_locator(ticker.MultipleLocator(100))
        axs[0, i-1].grid(color='green', alpha=0.5, which='both')
        if i==2:
            cbar = plt.colorbar(im, ax=axs[0,:], orientation='horizontal', shrink=0.97)
            cbar.ax.set_xlabel('difference (%)')
            axs[0, i-1].set_xlabel('nm')
            axs[1, i-1].set_xlabel('nm')
        uwdiff = (uws[i]-uws[0])[:, xslice, yslice]*NMPERPIXEL
        im = axs[1,i-1].imshow(np.linalg.norm(uwdiff, axis=0).T, cmap='inferno', vmax=7., vmin=0)
        im.set_extent(np.array(im.get_extent()) * NMPERPIXEL)
        a = 20
        axs[1,i-1].quiver(xx[::a,::a], yy[::a,::a], *uwdiff[:,::a,::a], 
                          color='white',angles='xy', scale_units='xy', scale=1/8)
        if i==2:
            cbar = plt.colorbar(im, ax=axs[1,:], orientation='horizontal', shrink=0.97)
            cbar.ax.set_xlabel('moiré displacement (nm)')

title_kwargs=dict(fontsize=14, fontweight='bold', loc='left')
for ax, label in zip(axs.flat, 'defg'):
    ax.set_title(label, **title_kwargs)
for ax, label in zip(oaxs.flat, 'abc'):
    ax.text(0.1, 1.02, label, transform=ax.transAxes,
            fontsize=14, fontweight='bold', va='bottom', ha='right')
#axs[0,0].set_title('Image difference')
#axs[1,0].set_title('GPA magnitude')
#plt.savefig(os.path.join(folder, name, 'dynamicsdraftT_t_sigma=1.pdf'), dpi=300)

# %%
matplotlib.__version__

# %%
plt.hist(np.linalg.norm(uwdiff, axis=0).ravel(), bins=100, range=(0,10))

# %%
if transpose:
    fig = plt.figure(figsize=[8,6.2], constrained_layout=True)
    gs = fig.add_gridspec(1,2, width_ratios=[1.,3])
    gs0 = gs[0].subgridspec(3, 1)
    gs1 = gs[1].subgridspec(2, 2)
    axs = gs1.subplots(sharex=True, sharey=True).T
else:
    fig = plt.figure(figsize=[6.5,8], constrained_layout=True)
    gs = fig.add_gridspec(2,1, height_ratios=[1.3,3])
    gs0 = gs[0].subgridspec(1, 3)
    gs1 = gs[1].subgridspec(2, 2)
    axs = gs1.subplots(sharex=True, sharey=True)

xx,yy = np.mgrid[250:400,550+175:550+325]
xx -=250
yy -=550 + 175
xx = xx*NMPERPIXEL
yy = yy*NMPERPIXEL
    
oaxs = gs0.subplots(sharey=True)
for a in oaxs:
    a.set_xticklabels([])
#axs = gs1.subplots(sharex=True, sharey=True)
for i, k in enumerate(ks):
    im = oaxs[i].imshow(ndatag[k, 250:400,550+175:550+325].T, cmap='gray')
    im.set_extent(np.array(im.get_extent())*NMPERPIXEL)
    oaxs[i].xaxis.set_minor_locator(ticker.MultipleLocator(100))
    oaxs[i].yaxis.set_minor_locator(ticker.MultipleLocator(100))
    oaxs[i].grid(color='green', alpha=0.8, which='both')
    oaxs[i].set_title(f"t = {dts[k].total_seconds():.0f}s")
    if i==2:
        cbar = plt.colorbar(im, ax=oaxs, orientation='horizontal', shrink=0.85)
        cbar.ax.set_xlabel('normalized intensity\n (arb.units.)')
    if i > 0:
        im = axs[0,i-1].imshow(100*(ndata-ndata[0])[k, 250:400,550+175:550+325].T, cmap='RdBu_r', vmax=100*clim[0], vmin=100*clim[1])
        im.set_extent(np.array(im.get_extent())*NMPERPIXEL)
        axs[0, i-1].xaxis.set_minor_locator(ticker.MultipleLocator(100))
        axs[0, i-1].yaxis.set_minor_locator(ticker.MultipleLocator(100))
        axs[0, i-1].grid(color='green', alpha=0.8, which='both')
        if i==2:
            cbar = plt.colorbar(im, ax=axs[0,:], orientation='horizontal', shrink=0.85)
            cbar.ax.set_xlabel('difference (%)')
        uwdiff = (uws[i]-uws[0])[:,250:400,550+175:550+325]*NMPERPIXEL
        im = axs[1,i-1].imshow(np.linalg.norm(uwdiff, axis=0).T, cmap='inferno', vmax=6.7, vmin=0)
        im.set_extent(np.array(im.get_extent()) * NMPERPIXEL)
        a = 20
        axs[1,i-1].quiver(xx[::a,::a], yy[::a,::a], *uwdiff[:,::a,::a], 
                          color='white',angles='xy', scale_units='xy', scale=1)
        if i==2:
            cbar = plt.colorbar(im, ax=axs[1,:], orientation='horizontal', shrink=0.85)
            cbar.ax.set_xlabel('moiré displacement (nm)\n')

# %%
uwdiff.max()

# %%
