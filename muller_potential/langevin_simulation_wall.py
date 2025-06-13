import numpy as np
from triple_well_potential import TripleWellPotential, TWP_grad
from muller_potential import MullerPotential, Muller_grad
import matplotlib.pyplot as plt
import numba
import time


@numba.njit()
def normal(Num: int) -> np.ndarray:
    r = np.zeros(Num)
    for i in range(Num):
        r[i] = np.random.normal()
    return r


@numba.njit()
def ul_simulation(grad_func, xdim, Nx, kbt, xinit=None, vinit=None,
                  gamma=100, tstep=5e-3, nstep=10**6, stride=10, random_seed=None, stride_print=False):

    if xinit is None:
        x0 = np.zeros((xdim, Nx), dtype=np.float64)
    else:
        x0 = xinit

    # x0 = x0.astype(np.float64)
    if vinit is None:
        v = np.zeros((xdim, Nx), dtype=np.float64)
    else:
        v = vinit

    if random_seed is not None:
        np.random.seed(random_seed)

    k = int(nstep / stride)
    xs = np.zeros(shape=(k, Nx, xdim))
    vs = np.zeros(shape=(k, Nx, xdim))
    sigma = np.sqrt(2 * gamma * kbt * tstep)
    for i in range(nstep):
        noise = normal(xdim * Nx).reshape(x0.shape)

        xt = x0 + v * tstep

        v = v - (grad_func(x0) + gamma * v) * \
            tstep + sigma * noise

        x0 = xt
        if i % stride == 0:
            idx = int(i / stride)
            xs[idx, :, :] = x0
            vs[idx, :, :] = v
            if stride_print:
                print(i)

    return xs, vs


# @numba.njit()
def ul_simulation_target(grad_func, xdim, Nx, kbt, c_a, c_b, xinit=None, vinit=None,
                         gamma=100, tstep=5e-3, nstep=10**6, random_seed=None):

    if xinit is None:
        x0 = np.zeros((Nx, xdim), dtype=np.float64)
    else:
        x0 = xinit

    # x0 = x0.astype(np.float64)
    if vinit is None:
        v = np.zeros((Nx, xdim), dtype=np.float64)
    else:
        v = vinit

    if random_seed is not None:
        np.random.seed(random_seed)

    sigma = np.sqrt(2 * gamma * kbt * tstep)
    mask = np.zeros(shape=Nx, dtype=np.bool_)
    arrival = np.zeros(shape=Nx)
    f = 1 - np.exp(-gamma * tstep)
    for i in range(nstep):
        noise = normal(xdim * Nx).reshape(x0.shape)

        v = v - grad_func(x0) * tstep
        dv = -f * v + np.sqrt(f * (2 - f) * kbt) * noise
        xt = x0 + (v + dv / 2) * tstep
        v = v + dv

        xt[xt[:, 0] < -2, 0] = -2
        xt[xt[:, 1] < -1, 1] = -1
        xt[xt[:, 0] > 2, 0] = 2
        xt[xt[:, 1] > 2.5, 1] = 2.5

        x0 = xt
        la = np.sum((x0 - c_a)**2, axis=1) < 0.2**2
        lb = np.sum((x0 - c_b)**2, axis=1) < 0.2**2
        arrival[np.logical_and(np.logical_not(mask), lb)] = 1
        mask = np.logical_or(np.logical_or(mask, la), lb)
        if np.all(mask):
            print(f'yeah:{i}')
            break

    if not np.all(mask):
        print('Warning: some points heve not arrived!')
    return arrival


if __name__ == '__main__':
    '''
    muller = MullerPotential()
    grad_func = muller.gradient
    kbt = 25
    print(muller.c_b())
    xs, vs = ul_simulation(grad_func, 2, kbt, nstep=5 * 10 **
                           5, stride=1, xinit=muller.c_a())

    # plt.scatter(xs[:, 0], xs[:, 1], alpha=0.5)

    ngrid = 400
    grid = np.linspace(-2, 2, ngrid)
    y, x = np.meshgrid(grid, grid)
    y = y.flatten()
    x = x.flatten()
    g = np.array([x, y]).T
    # filename='simulation/long/COLVAR'
    data = xs

    # fes=calculateFES_multi(df,grid,16)
    nstart = 0
    h = hist_reweight(data, np.ones_like(data[:, 0]), -2, 2, -2, 2, ngrid)

    h = h.flatten()
    cc = np.log(h[h > 0])
    thread = -20
    cc[cc < thread] = thread
    plt.scatter(g[:, 0][h > 0], g[:, 1][h > 0],
                cmap='turbo', c=cc, s=1)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.colorbar(label='FES')
    plt.show()

    np.savetxt('muller_25_A_5e5.txt', xs)
    '''
    muller = MullerPotential()
    grad_fn = Muller_grad

    '''
    N = 100
    N_simulation = 10**6
    x0 = np.linspace(-1, 1, num=N + 1)
    count = np.zeros((N + 1,))
    for t in range(N_simulation):
        xs, _ = ul_simulation(grad_fn, x0.shape[0], kbt=0.1, xinit=x0, vinit=np.zeros_like(
            x0), gamma=.1, tstep=0.005, nstep=10**3 * 2, stride=1)
        arrival = np.ones_like(xs[0]) * 0.5
        mask = np.ones_like(xs[0], dtype=bool)
        for i in range(len(xs)):

            # print(xi >= 1)

            arrival[np.logical_and(mask, xs[i] >= 1)] = 1
            arrival[np.logical_and(mask, xs[i] <= -1)] = 0

            mask = np.logical_and(mask, xs[i] > -1)
            mask = np.logical_and(mask, xs[i] < 1)

        if np.any(mask):
            print('Warning! Some points have not arrived!')
        if t % 100 == 0:
            print(t)
        count = count + arrival

    print(count / N_simulation)
    np.savetxt('q_s_1d.txt', count / N_simulation)
    '''
    xmin, xmax = -1.5, 1.2
    ymin, ymax = -.2, 2
    dx = 0.05
    dy = 0.05
    Nx = int((xmax - xmin) / dx)
    Ny = int((ymax - ymin) / dy)
    kbt = 5
    gamma = 1

    Ncol = Nx + 1
    Nrow = Ny + 1
    x = np.linspace(xmin, xmax, Nx + 1)
    y = np.linspace(ymin, ymax, Ny + 1)

    if Ncol == Nx - 1:
        xcal = x[1:-1]
    else:
        xcal = x

    if Nrow == Ny - 1:
        ycal = y[1:-1]
    else:
        ycal = y

    X, Y = np.meshgrid(xcal, ycal)
    # print(X.shape, V)

    points = np.array([X.reshape(-1), Y.reshape(-1)]).T.astype(np.float64)
    U = muller.potential(points)
    print(np.max(U))
    print(gamma)
    points = points[U < 100]
    v_sample = 25

    x0 = points
    c_a = muller.c_a()
    c_b = muller.c_b()
    T = 10**6
    N = 100
    stride = 10
    arr = np.zeros(x0.shape[0])
    # vs = np.random.normal(size=(v_sample, 2)) * np.sqrt(kbt)
    # np.savetxt(f'muller_potential/model/simulation_{kbt}/simulation_vconfig_kbt{kbt}.txt', vs)
    vs = np.loadtxt(
        f'muller_potential/model/simulation_{kbt}/simulation_vconfig_kbt{kbt}.txt')

    for i in range(v_sample):
        st = time.time()
        arr = np.zeros(x0.shape[0])
        x0 = points
        v = np.random.normal(size=(2)) * np.sqrt(kbt)
        vs[i, :] = v
        vinit = np.tile(v, (x0.shape[0], 1))
        for t in range(N):
            print(t)
            arr += ul_simulation_target(grad_fn,
                                        c_a=c_a,
                                        c_b=c_b,
                                        xdim=x0.shape[1],
                                        Nx=x0.shape[0],
                                        kbt=kbt,
                                        xinit=x0,
                                        vinit=vinit,
                                        gamma=gamma,
                                        tstep=2e-4,
                                        nstep=T,
                                        random_seed=None)
            '''
            xs, _ = ul_simulation(grad_fn, xdim=x0.shape[1],Nx = x0.shape[0], kbt=kbt, xinit=x0, vinit=np.zeros_like(
                x0), gamma=gamma, tstep=0.005, nstep=T, stride=stride)



            mask = np.zeros(x0.shape[0],dtype=bool)
            for i in range(int(T/stride)):
                mask = np.logical_or(mask,TWP.in_a(xs[i,:,:],r=0.2))
                mask = np.logical_or(mask,TWP.in_b(xs[i,:,:],r=0.2))
            print(np.all(mask))
            '''
        arr = arr / N
        arr = arr.reshape((points.shape[0], 1))
        result = np.concatenate((points, arr), axis=1)
        np.savetxt(
            f'muller_potential/model/simulation_{kbt}/simulation_kbt{kbt}_gamma{gamma}_{i}_2.txt',
            result)
        tt = time.time()
        print(f'Using time {tt-st}!')
    # np.savetxt('model/simulation_kbt.1_gamma10.txt', xs)
    # plt.scatter(xs[:, 0], xs[:, 1], alpha=0.05)
    # plt.show()
    np.savetxt(
        f'muller_potential/model/simulation_{kbt}/simulation_vconfig_kbt{kbt}.txt',
        vs)
