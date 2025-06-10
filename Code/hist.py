import matplotlib.pyplot as plt
import numpy as np
import torch
# Force matplotlib to not use any Xwindows backend.
# matplotlib.use('Agg')

import multiprocessing
from muller_potential import MullerPotential
from utils import read_COLVAR


import time


def calculateFES(df, grid, sigma2=np.pi / 360):
    torsion = np.array([df[1], df[2]]).T

    l = grid.shape[0]
    fes = np.zeros(l)
    N = len(df.index)
    for i in range(l):
        t = grid[i]
        diff = np.abs(torsion - t)
        diff[diff > np.pi] = 2 * np.pi - diff[diff > np.pi]
        fes[i] = np.sum(np.exp(-np.sum(diff**2, axis=1) / 2 / sigma2)) / N

    return fes


def calculateFES_step(df, grid, sigma2, q, i):
    fes = calculateFES(df, grid, sigma2)
    print(fes)
    q[i] = fes


def calculateFES_multi(df, grid, nump, sigma2=np.pi / 360):
    l = grid.shape[0]
    n = int(l / nump)
    p = []
    q = multiprocessing.Manager().dict()
    fes = np.zeros(l)
    for i in range(nump):
        p.append(multiprocessing.Process(target=calculateFES_step,
                 args=(df, grid[i * n:i * n + n, :], sigma2, q, i)))
        p[-1].start()

    for i in range(nump):
        p[i].join()
        print(f"{i} process complete")

    for i in range(nump):
        fes[i * n:i * n + n] = q[i]
    return fes


def hist(data, xmin, xmax, ymin, ymax, Nbins, hist=None):
    dx = (xmax - xmin) / Nbins
    dy = (ymax - ymin) / Nbins
    dd = data - np.array([xmin, ymin])
    if hist is None:
        bins = np.zeros((Nbins, Nbins))
    else:
        bins = hist
    for i in range(data.shape[0]):
        x, y = int(np.floor(dd[i, 0] / dx)), int(np.floor(dd[i, 1] / dy))
        bins[x, y] = bins[x, y] + 1
    return bins


def hist_reweight(data, value, xmin, xmax, ymin, ymax, Nbins, hist=None):
    dx = (xmax - xmin) / Nbins
    dy = (ymax - ymin) / Nbins
    dd = data - np.array([xmin, ymin])
    l = np.any([(data[:,
                      0] <= xmin), (data[:,
                                         0] >= xmax), (data[:,
                                                            1] <= ymin), (data[:,
                                                                               1] >= ymax)], axis=0)
    if np.any(l):
        print('warning: out of range')
    dd = dd[~l, :]
    v = value[~l]
    if hist is None:
        bins = np.zeros((Nbins, Nbins))
    else:
        bins = hist

    for i in range(dd.shape[0]):
        x, y = int(np.floor(dd[i, 0] / dx)), int(np.floor(dd[i, 1] / dy))
        bins[x, y] = bins[x, y] + v[i]
    return bins


def grid(xmin, xmax, ymin, ymax, xbins, ybins):
    x, y = np.linspace(xmin, xmax, xbins), np.linspace(ymin, ymax, ybins)
    x, y = np.meshgrid(x, y)
    return np.column_stack((x.flatten(), y.flatten()))


if __name__ == '__main__':

    DEBUG = True
    samples = 20

    # first, run a long simulation
    # run_macro_simulation(debug=DEBUG)
    '''
    muller = MullerPotential()
    kbt = 1 / 0.15
    n = 100

    x_values = np.random.uniform(size=(n * n, 2)) * \
        np.array([2.7, 2.2]) - np.array([1.5, 0.2])
    x_values = x_values[~muller.in_a(x_values)]
    x_values = x_values[~muller.in_b(x_values)]
    print(x_values.shape)
    center = [-0.75, 0.6]
    l = np.sum((x_values - center)**2, axis=1) < 0.5**2

    U = muller.potential(x_values)

    w = np.exp(- (U - np.min(U)) / kbt)
    w = w
    '''

    ngrid = 900
    grid = np.linspace(-3, 3, ngrid)
    y, x = np.meshgrid(grid, grid)
    y = y.flatten()
    x = x.flatten()
    g = np.array([x, y]).T
    fig, axs = plt.subplots(2, 3, figsize=(15, 10))
    temps = [1, 5, 10, 15, 20, 25]
    for itr in range(6):
        i = int(np.floor(itr / 3))
        j = itr % 3
        print(i, j)
        temp = temps[itr]
        print(f'reading from simulation/A/COLVAR_{temp:.2f}')
        data = read_COLVAR(f'simulation/A/COLVAR_{temp:.2f}')
        data = data[:, 1:3]
        w = np.ones_like(data[:, 0])
        h = hist_reweight(data, w, -3, 3, -3, 3, ngrid)

        h = h.flatten()
        cc = np.log(h[h > 0])
        thread = -20
        cc[cc < thread] = thread
        plt.scatter(g[:, 0][h > 0], g[:, 1][h > 0],
                    cmap='turbo', c=cc, s=1)

        sc = axs[i, j].scatter(g[:, 0][h > 0], g[:, 1][h > 0],
                               cmap='turbo', c=cc, s=1)
        axs[i, j].set_title(f'FES, Temperature = {temp:.2f}')
        axs[i, j].set_xlabel('x')
        axs[i, j].set_ylabel('y')
        fig.colorbar(sc, ax=axs[i, j])

    plt.savefig('simulation/A/tempering.png', dpi=300)
    plt.clf()

    '''
    data = np.loadtxt('COLVAR')
    data = data[:, 1:3]
    print(data)
    filename = 'muller_0.15.txt'
    w = np.ones_like(data[:, 0])
    # filename='simulation/long/COLVAR'
    # data = np.loadtxt(filename)
    # data = x_values

    # fes=calculateFES_multi(df,grid,16)
    nstart = 0
    h = hist_reweight(data, w, -2, 2, -2, 2, ngrid)

    h = h.flatten()
    cc = np.log(h[h > 0])
    thread = -20
    cc[cc < thread] = thread
    plt.clf()
    plt.scatter(g[:, 0][h > 0], g[:, 1][h > 0],
                cmap='turbo', c=cc, s=1)
    plt.xlabel('$\\phi$')
    plt.ylabel('$\\psi$')
    plt.colorbar(label='FES')
    plt.savefig('test.png', dpi=300)
    plt.clf()
    '''
    # draw point density
    '''
    pi=3.14159265
    plt.hist2d(df[1], df[2],range=[[-pi,pi], [-pi, pi]],bins=360,norm=LogNorm())
    plt.colorbar()

    # "improved" x/y axis ticks


    # axis labels (right order?)
    plt.xlabel('$\\phi$')
    plt.ylabel('$\\psi$')

    # 1:1 aspect ratio

    # remove grid lines
    print("Saving figure 'Ramachandran-Plot'")
    plt.savefig('Ramachandran-Plot.png', dpi=300)
    plt.clf()
    '''
