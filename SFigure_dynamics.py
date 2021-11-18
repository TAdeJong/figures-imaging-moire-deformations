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
#     display_name: Python [conda env:pyL5cupy]
#     language: python
#     name: conda-env-pyL5cupy-py
# ---

# %%
import matplotlib as mpl
import matplotlib.ticker as ticker
import matplotlib.animation as animation
import numpy as np
import scipy.ndimage as ndi
import matplotlib
import matplotlib.pyplot as plt
import dask.array as da
from dask.distributed import Client, LocalCluster
import os

from pyL5.lib.analysis.container import Container
from ipywidgets import interactive
import ipywidgets as widgets
from skimage.morphology import binary_erosion, disk

import colorcet
from dateutil.parser import parse

# %%
import pyGPA.geometric_phase_analysis as GPA
import pyGPA
import pyGPA.property_extract as pe
from pyGPA.imagetools import gauss_homogenize2


# %%
import pyGPA.cuGPA as cuGPA

# %%
# %matplotlib inline

# %%
folder = r'/mnt/storage-linux/speeldata/20200716-XTBLG02'
name = '20200717_133946_2.3um_480.6_movement'

container = Container(os.path.join(folder, name+'.nlp'))

data = container.getStack('driftcorrected').getDaskArray().astype(float)
NMPERPIXEL = 1.36

# %%
dims = data.shape
xx, yy = np.meshgrid(np.arange(-dims[2]//2, dims[2]//2),
                     np.arange(-dims[1]//2, dims[1]//2))

outer_radius = 640
mask = (xx)**2 + (yy)**2 < outer_radius**2

edge = 20

# %%
ndata = np.where(mask, data, np.nan)

ndata = ndata[:, edge:-edge, edge:-edge]
ndata = ndata / np.nanmean(ndata, axis=(1, 2), keepdims=True)
ndatag = gauss_homogenize2(ndata, mask[None, edge:-edge, edge:-edge], sigma=[0, 50, 50])

# %% tags=[]
da.to_zarr(ndatag, os.path.join(folder, name, f'gausshomogenize50.zarr'), overwrite=True)

# %%
cluster = LocalCluster(n_workers=16, threads_per_worker=2, memory_limit='1GB')
client = Client(cluster)
client


# %%
def plot_stack2(images, n):
    """Plot the n-th image from a stack of n images.
    For interactive use with ipython widgets"""
    im = images[n, :, :]
    plt.figure(figsize=[12, 12])
    plt.imshow(im.T, cmap='gray')  # , vmax=im.mean()*2)
    plt.show()


ndatag = da.from_zarr(os.path.join(folder, name, f'gausshomogenize50.zarr'))
ndata = ndi.filters.gaussian_filter1d(ndatag, sigma=1,
                                      axis=0, order=0)
interactive(lambda n: plot_stack2(ndatag, n),
            n=widgets.IntSlider(1, 0, ndata.shape[0]-1, 1, continuous_update=False)
            )

# %%
cropped = pyGPA.imagetools.trim_nans2(ndata[0])
pks, _ = GPA.extract_primary_ks(cropped, plot=True, pix_norm_range=(5, 100), sigma=3)

# %%
kw = np.linalg.norm(pks, axis=1).mean()/4
sigma = 20
dr = 2*sigma
kstep = kw/3  # 6
mask = ~np.isnan(ndata[0])
mask = binary_erosion(np.pad(mask, 1), selem=disk(dr))[1:-1, 1:-1]
maskzero = 0.000001
uws = []
uws_old = []
ks = [0, 55, 100]
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
        g = cuGPA.wfr2_grad_opt(deformed, sigma,
                                pk[0], pk[1], kw=kw,
                                kstep=kstep, grad='diff')
        gs.append(g)
        plt.figure()
        plt.imshow(np.angle(g['lockin']).T)
    #phases = np.angle(gs)
    phases = np.stack([np.angle(g['lockin']) for g in gs])
    grads = np.stack([g['grad'] for g in gs])
    weights = np.abs([g['lockin'] for g in gs])*(mask+maskzero)
    #u = GPA.reconstruct_u_inv_from_phases(pks, phases, weights, weighted_unwrap=True)
    u_grad = GPA.reconstruct_u_inv_from_phases(pks, grads[..., ::-1],
                                               # grads,
                                               # -2*np.pi*grads,
                                               weights, pre_diff=True)
    #u_inv = GPA.invert_u_overlap(unew, iters=3)
    uws.append(u_grad)
    # uws_old.append(u)

# %%
# Check that average displacement is 0, i.e. no global movement.
uws = np.array(uws)
uws.mean(axis=(-1, -2))

# %%
meanshifts = np.nanmean(np.where(mask, uws, np.nan), axis=(-1, -2))
meanshifts - meanshifts[0]

# %%

props = []
for u in uws:
    J = pe.u2J(u, NMPERPIXEL)
    prop = pe.Kerelsky_J(J, pks, nmperpixel=NMPERPIXEL, sort=1)[0]
    props.append(prop)

# %%
props = da.compute(props)[0]

# %%
acont = Container(os.path.join(folder, name))

dts = [parse(x) - parse(acont['TTT'][0]) for x in acont["TTT"]]
clim = np.nanquantile(ndata-ndata[0], [0.9999, 0.0001])
clim = [0.05, -0.05]

# %%
clim2 = da.percentile(ndatag[~np.isnan(ndatag)].flatten(), [0.1, 99.9]).compute()

# %%

fig, ax = plt.subplots(ncols=3, figsize=[12, 5], constrained_layout=True, sharex=True)
im = [None]*3
#k = 123
k = 100
xx, yy = np.mgrid[0:ndata[k].shape[0], 0:ndata[k].shape[1]]*NMPERPIXEL
a = 20
im[0] = ax[0].imshow(ndatag[k].T, cmap='gray', vmax=clim2[1], vmin=clim2[0])
cbar = plt.colorbar(im[0], ax=ax[0], orientation='horizontal', shrink=1)
cbar.ax.set_xlabel('normalized intensity\n (arb.units.)')
im[1] = ax[1].imshow(100*(ndata[k] - ndata[0]).T, cmap='RdBu_r', vmax=100*clim[0], vmin=100*clim[1])
ax[1].set_title(f"t= {dts[k]}")
cbar1 = plt.colorbar(im[1], ax=ax[1], orientation='horizontal', shrink=1)
cbar1.ax.set_xlabel('difference (%)')
uwdiff = (uws[2]-uws[0])*NMPERPIXEL
#uwdiff = (unew-uws_old[0])*NMPERPIXEL
uwdiff = np.where(mask, uwdiff, np.nan)
im[2] = ax[2].imshow(np.linalg.norm(uwdiff, axis=0).T, cmap='inferno', vmax=7, vmin=0)
a = 30
arrow_scale = 12
Q = ax[2].quiver(xx[::a, ::a], yy[::a, ::a], *uwdiff[:, ::a, ::a],
                 color='white', angles='xy', scale_units='xy', scale=1/arrow_scale)
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
    gs = [cuGPA.wfr2_grad_opt(deformed, sigma,
                              pk[0], pk[1], kw=kw,
                              kstep=kstep, grad='diff')
          for pk in pks]
    grads = np.stack([g['grad'] for g in gs])
    weights = np.abs([g['lockin'] for g in gs])*(mask+maskzero)
    unew = GPA.reconstruct_u_inv_from_phases(pks, grads[..., ::-1], weights, pre_diff=True)

    #gs = []
    # for pk in pks:
    #    g = GPA.wfr2_only_lockin_vec(deformed, sigma, pk[0]*1., pk[1]*1., kw=kw, kstep=kstep)
    #    gs.append(g)
    #gs = np.stack(gs)
#     phases = np.angle(gs)
#     weights = np.abs(gs)*(mask+maskzero)
#     unew = GPA.reconstruct_u_inv_from_phases(pks, phases, weights, weighted_unwrap=True)
    uwdiff = (unew-uws[0])*NMPERPIXEL
    uwdiff = np.where(mask, uwdiff, np.nan)
    im[2].set_data(np.linalg.norm(uwdiff, axis=0).T)
    Q.set_UVC(*uwdiff[:, ::a, ::a])
    if (i % 10) == 0:
        print(i)
    else:
        print(i, end=' ')
    return im + [Q]


ani = animation.FuncAnimation(fig, update, interval=200,
                              blit=True, save_count=ndata.shape[0])
ani.save(os.path.join('figures', f'Sdeformationcum_133946_2.3um_t_sigma=1.mp4'),
         dpi=300)

# %%
glasbey = plt.get_cmap('cet_glasbey_dark')(np.linspace(0, 1, 255))

transpose = True
# NMPERPIXEL=2.23
#xslice, yslice = slice(250,600), slice(550,900)
xa, ya = 350, 200
xslice, yslice = slice(xa, xa+655-250), slice(ya, ya+910-550)
xx, yy = np.mgrid[xslice, yslice]
xx -= xa
yy -= ya
xx = xx*NMPERPIXEL
yy = yy*NMPERPIXEL

loc_im = ndatag[ks[0]]
plt.imshow(loc_im.T, cmap='gray')
rect = mpl.patches.Rectangle([xslice.start, yslice.start],
                             xslice.stop-xslice.start,
                             yslice.stop-yslice.start, facecolor='none', edgecolor=glasbey[2])
plt.gca().add_patch(rect)


# %%
if transpose:
    fig = plt.figure(figsize=[8.5, 6.5], constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 3])
    gs0 = gs[0].subgridspec(3, 1)
    gs1 = gs[1].subgridspec(2, 2)
    axs = gs1.subplots(sharex=True, sharey=True).T
else:
    fig = plt.figure(figsize=[6.5, 8], constrained_layout=True)
    gs = fig.add_gridspec(2, 1, height_ratios=[1.3, 3])
    gs0 = gs[0].subgridspec(1, 3)
    gs1 = gs[1].subgridspec(2, 2)
    axs = gs1.subplots(sharex=True, sharey=True)

clim = np.nanquantile(ndata-ndata[0], [0.9999, 0.0001])
clim = [0.05, -0.05]

oaxs = gs0.subplots(sharey=True, sharex=True)
for a in oaxs:
    a.set_xticklabels([])
#axs = gs1.subplots(sharex=True, sharey=True)
for i, k in enumerate(ks):
    im = oaxs[i].imshow(ndatag[k, xslice, yslice].T, cmap='gray')
    im.set_extent(np.array(im.get_extent())*NMPERPIXEL)
    oaxs[i].xaxis.set_minor_locator(ticker.MultipleLocator(100))
    # oaxs[i].yaxis.set_minor_locator(ticker.MultipleLocator(100))
    # oaxs[i].yaxis.set_major_locator(ticker.MultipleLocator(400))
    oaxs[i].xaxis.set_major_locator(ticker.MultipleLocator(200))
    oaxs[i].grid(color=glasbey[2], alpha=0.5, which='both')
    oaxs[i].set_title(f"t = {dts[k].total_seconds():.0f}s", fontsize='medium')
    oaxs[i].set_ylabel('nm')
    if i == 2:
        cbar = plt.colorbar(im, ax=oaxs, orientation='horizontal', shrink=0.81, pad=0.01)
        cbar.ax.set_xlabel('normalized intensity\n (arb.units.)')
    if i > 0:
        im = axs[0, i-1].imshow(100*(ndata-ndata[0])[k, xslice, yslice].T,
                                cmap='RdBu_r', vmax=100*clim[0], vmin=100*clim[1])
        im.set_extent(np.array(im.get_extent())*NMPERPIXEL)
        axs[0, i-1].xaxis.set_minor_locator(ticker.MultipleLocator(100))
        axs[0, i-1].yaxis.set_minor_locator(ticker.MultipleLocator(100))
        axs[0, i-1].grid(color=glasbey[2], alpha=0.7, which='both')
        if i == 2:
            cbar = plt.colorbar(im, ax=axs[0, :], orientation='horizontal', shrink=0.98, pad=0.02)
            cbar.ax.set_xlabel('difference (%)')
            axs[0, i-1].set_xlabel('nm')
            axs[1, i-1].set_xlabel('nm')
        uwdiff = -1*(uws[i]-uws[0])[:, xslice, yslice]*NMPERPIXEL
        im = axs[1, i-1].imshow(np.linalg.norm(uwdiff, axis=0).T, cmap='inferno', vmax=7.,
                                vmin=0)
        im.set_extent(np.array(im.get_extent()) * NMPERPIXEL)
        a = 20
        axs[1, i-1].quiver(xx[::a, ::a], yy[::a, ::a], *uwdiff[:, ::a, ::a],
                           color='white', angles='xy', scale_units='xy', scale=1/8)
        if i == 2:
            cbar = plt.colorbar(im, ax=axs[1, :], orientation='horizontal', shrink=0.98, pad=0.02)
            cbar.ax.set_xlabel('moiré displacement (nm)')

# for ax in axs[0, :]:
#     ax.set_yticklabels([])
title_kwargs = dict(fontsize=14, fontweight='bold', loc='left')
for ax, label in zip(axs.flat, 'defg'):
    ax.set_title(label, **title_kwargs)
for ax, label in zip(oaxs.flat, 'abc'):
    ax.set_title(label, **title_kwargs)
    # ax.text(0.1, 1.02, label, transform=ax.transAxes,
    #        fontsize=14, fontweight='bold', va='bottom', ha='right')
#axs[0,0].set_title('Image difference')
#axs[1,0].set_title('GPA magnitude')
plt.savefig(os.path.join('figures', 'Sdynamics_t_sigma=1.pdf'), dpi=300)

# %%
props0 = da.compute(props[0])[0]

# %%
plt.imshow(props0[..., 1].T % 180, cmap='twilight')
plt.colorbar()

# %%
plt.imshow(props0[..., 0].T)
plt.colorbar()

# %%
fig, axs = plt.subplots(ncols=2, figsize=[12, 4])

im = axs[0].imshow(np.where(mask, props0[..., 0], np.nan).T,
                   vmin=np.quantile(props0[..., 0], 0.001),
                   vmax=np.quantile(props0[..., 0], 0.97)
                   )

im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
plt.colorbar(im, ax=axs[0], label='$\\theta^*$ ($^\\circ$)')

im = axs[1].imshow(loc_im.T, cmap='gray')

epscolor = props0[..., 2]*100
angle = np.deg2rad(props0[..., 1])  # % 180
magnitude = epscolor
#angle = np.deg2rad(props0[...,3])
#magnitude = (props0[...,2] - 1)
kx = magnitude*np.cos(angle)
ky = magnitude*np.sin(angle)
a = 48
xxl, yyl = np.meshgrid(np.arange(loc_im.shape[0]),
                       np.arange(loc_im.shape[1]),
                       indexing='ij')

kx = np.where(mask, kx, np.nan)
ky = np.where(mask, ky, np.nan)
# axs[1].quiver((xxl[dr:-dr:a, dr:-dr:a])*NMPERPIXEL/1000,
#               (yyl[dr:-dr:a, dr:-dr:a])*NMPERPIXEL/1000,
#               *uws[0,..., dr:-dr:a, dr:-dr:a],
#               angles='xy', scale_units='xy', headwidth=5, ec=glasbey[1], linewidth=1, pivot='tip',
#               scale=1000/NMPERPIXEL, color=glasbey[1])
Q = axs[1].quiver((xxl[dr:-dr:a, dr:-dr:a])*NMPERPIXEL/1000,
                  (yyl[dr:-dr:a, dr:-dr:a])*NMPERPIXEL/1000,
                  ky[dr:-dr:a, dr:-dr:a],
                  kx[dr:-dr:a, dr:-dr:a],
                  np.clip(magnitude[dr:-dr:a, dr:-dr:a], 0, np.quantile(magnitude, 0.98)),
                  cmap='magma',
                  norm=mpl.colors.Normalize(vmin=0),
                  angles='xy', scale_units='xy',
                  width=0.006, pivot='mid',
                  ec='black', linewidth=0.5,
                  headlength=0., headwidth=0.5, headaxislength=0)
plt.colorbar(Q, ax=axs[1], label='$\\epsilon$ (%)')  # , ticks=[1, 1.1, 1.2])
for ax in axs:
    rect = mpl.patches.Rectangle(np.array([xslice.start, yslice.start]) * NMPERPIXEL/1000,
                                 (xslice.stop-xslice.start) * NMPERPIXEL/1000,
                                 (yslice.stop-yslice.start) * NMPERPIXEL/1000,
                                 facecolor='none', edgecolor=glasbey[2], linewidth=2)
    ax.add_patch(rect)
im.set_extent(np.array(im.get_extent())*NMPERPIXEL/1000)
axs[0].set_title('Local twist angle')
axs[1].set_title('Anisotropy')
for ax, l in zip(axs, 'ab'):
    ax.set_title(l, loc='left', fontweight='bold')
    ax.set_xlabel('y (μm)')
axs[0].set_ylabel('x (μm)')
plt.tight_layout()
plt.savefig(os.path.join('figures', 'SFig_dynamics_theta_and_aniso2.pdf'))

# %%
