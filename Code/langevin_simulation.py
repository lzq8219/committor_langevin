import numpy as np
from muller_potential import MullerPotential
import matplotlib.pyplot as plt
from committor_langevin.src.hist import hist_reweight
import numba


def ul_simulation(grad_func, xdim, kbt, xinit=None, vinit=None,
                  gamma=100, tstep=5e-3, nstep=10**6, stride=10, random_seed=None, stride_print=False):

    if xinit is None:
        x0 = np.zeros((xdim,))
    else:
        x0 = xinit

    x0 = x0.astype(np.float32)
    if vinit is None:
        v = np.zeros((xdim,))
    else:
        v = vinit

    if random_seed is not None:
        np.random.seed(random_seed)

    xs = []
    vs = []
    sigma = np.sqrt(2 * gamma * kbt * tstep)
    for i in range(nstep):
        noise = np.random.normal(size=x0.shape)
        dx = v * tstep

        xt = x0 + dx
        xt = xt.astype(np.float32)

        v = v - (grad_func(x0) + gamma * v) * \
            tstep + sigma * noise

        x0 = xt
        if i % stride == 0:
            xs.append(xt)
            vs.append(v)
            if stride_print:
                print(i)

    xs = np.array(xs)
    vs = np.array(vs)
    return xs, vs


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

    def grad_fn(x):
        return 4 * x * (x**2 - 1)

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
