import numpy as np
from triple_well_potential import TripleWellPotential, TWP_grad
from muller_potential import MullerPotential, Muller_grad
import matplotlib.pyplot as plt
import numba
import time
from multiprocessing import Pool
from functools import partial


@numba.njit()
def normal(Num: int) -> np.ndarray:
    r = np.zeros(Num)
    for i in range(Num):
        r[i] = np.random.normal()
    return r


#@numba.njit()
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


#@numba.njit()
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
        print(i,end='\r')
        noise = normal(xdim * Nx).reshape(x0.shape)

        v = v - grad_func(x0) * tstep
        dv = -f * v + np.sqrt(f * (2 - f) * kbt) * noise
        xt = x0 + (v + dv / 2) * tstep
        v = v + dv

        v[xt[:, 0] < -1.5, 0] = 0
        v[xt[:, 1] < -0.2, 1] = 0
        v[xt[:, 0] > 1.2, 0] = 0
        v[xt[:, 1] > 2, 1] = 0
        xt[xt[:, 0] < -1.5, 0] = -1.5
        xt[xt[:, 1] < -0.2, 1] = -0.2
        xt[xt[:, 0] > 1.2, 0] = 1.2
        xt[xt[:, 1] > 2, 1] = 2

        x0 = xt
        la = np.sum((x0 - c_a)**2, axis=1) < 0.2**2
        lb = np.sum((x0 - c_b)**2, axis=1) < 0.2**2
        arrival[np.logical_and(np.logical_not(mask), lb)] = 1
        mask = np.logical_or(np.logical_or(mask, la), lb)
        if np.all(mask):
            print(f'yeah:{i}')
            break

    if not np.all(mask):
        not_arrived = np.sum(np.logical_not(mask))
        print(f'Warning: {not_arrived} points have not arrived!')
    return arrival

def ul_simulation_target_1(grad_func, xdim, Nx, kbt, c_a, c_b, xinit=None, vinit=None,
                         gamma=100, tstep=5e-3, nstep=10**6, random_seed=None):

    if xinit is None:
        x0 = np.zeros((Nx, xdim), dtype=np.float64)
    else:
        x0 = xinit.copy()

    if vinit is None:
        v = np.zeros((Nx, xdim), dtype=np.float64)
    else:
        v = vinit.copy()

    if random_seed is not None:
        np.random.seed(random_seed)

    sigma = np.sqrt(2 * gamma * kbt * tstep)
    mask = np.zeros(shape=Nx, dtype=np.bool_)
    arrival = np.zeros(shape=Nx)
    f = 1 - np.exp(-gamma * tstep)
    
    for i in range(nstep):
        print(i, end='\r')
        
        # Only update points that haven't arrived
        active = ~mask
        n_active = np.sum(active)
        
        if n_active == 0:
            print(f'All points arrived at step {i}')
            break
        
        # Generate noise only for active points
        noise = normal(xdim * n_active).reshape(n_active, xdim)

        # Update only active points
        v[active] = v[active] - grad_func(x0[active]) * tstep
        dv = -f * v[active] + np.sqrt(f * (2 - f) * kbt) * noise
        xt = x0[active] + (v[active] + dv / 2) * tstep
        v[active] = v[active] + dv

        # Apply boundary conditions for active points
        '''
        v[active, 0][xt[:, 0] < -1.5] = 0
        v[active, 1][xt[:, 1] < -0.2] = 0
        v[active, 0][xt[:, 0] > 1.2] = 0
        v[active, 1][xt[:, 1] > 2] = 0
        
        xt[xt[:, 0] < -1.5, 0] = -1.5
        xt[xt[:, 1] < -0.2, 1] = -0.2
        xt[xt[:, 0] > 1.2, 0] = 1.2
        xt[xt[:, 1] > 2, 1] = 2
        '''

        # Update positions for active points
        x0[active] = xt
        
        # Check arrival for active points only
        la = np.sum((x0[active] - c_a)**2, axis=1) < 0.2**2
        lb = np.sum((x0[active] - c_b)**2, axis=1) < 0.2**2
        
        # Update mask and arrival for active points
        active_indices = np.where(active)[0]
        arrival[active_indices[lb]] = 1
        mask[active_indices[np.logical_or(la, lb)]] = True

    if not np.all(mask):
        not_arrived = np.sum(~mask)
        print(f'Warning: {not_arrived} points have not arrived!')
    
    return arrival

def ul_simulation_target_2(grad_func, xdim, Nx, kbt, c_a, c_b, xinit=None, vinit=None,
                         gamma=100, tstep=5e-3, nstep=10**6, random_seed=None):

    if xinit is None:
        x0 = np.zeros((Nx, xdim), dtype=np.float64)
    else:
        x0 = xinit.copy()

    if vinit is None:
        v = np.zeros((Nx, xdim), dtype=np.float64)
    else:
        v = vinit.copy()

    if random_seed is not None:
        np.random.seed(random_seed)

    
    mask = np.zeros(shape=Nx, dtype=np.bool_)
    arrival = np.zeros(shape=Nx)
    c1 =  np.exp(-gamma * tstep/2)
    sigma = np.sqrt(2*gamma*kbt*(1-c1**2))
    
    for i in range(nstep):
        print(i, end='\r')
        
        # Only update points that haven't arrived
        active = ~mask
        n_active = np.sum(active)
        
        if n_active == 0:
            print(f'All points arrived at step {i}')
            break
        
        # Generate noise only for active points
        noise = normal(xdim * n_active).reshape(n_active, xdim)

        # Update only active points
        v[active] = c1*v[active] + sigma*noise - grad_func(x0[active]) * tstep/2
        
        
        xt = x0[active] + (v[active]) * tstep
        v[active] = v[active]- grad_func(xt) * tstep/2
        noise = normal(xdim * n_active).reshape(n_active, xdim)
        v[active] = c1*v[active] + sigma*noise

        # Apply boundary conditions for active points
        '''
        v[active, 0][xt[:, 0] < -1.5] = 0
        v[active, 1][xt[:, 1] < -0.2] = 0
        v[active, 0][xt[:, 0] > 1.2] = 0
        v[active, 1][xt[:, 1] > 2] = 0
        
        xt[xt[:, 0] < -1.5, 0] = -1.5
        xt[xt[:, 1] < -0.2, 1] = -0.2
        xt[xt[:, 0] > 1.2, 0] = 1.2
        xt[xt[:, 1] > 2, 1] = 2
        '''

        # Update positions for active points
        x0[active] = xt
        
        # Check arrival for active points only
        la = np.sum((x0[active] - c_a)**2, axis=1) < 0.2**2
        lb = np.sum((x0[active] - c_b)**2, axis=1) < 0.2**2
        
        # Update mask and arrival for active points
        active_indices = np.where(active)[0]
        arrival[active_indices[lb]] = 1
        mask[active_indices[np.logical_or(la, lb)]] = True

    if not np.all(mask):
        not_arrived = np.sum(~mask)
        print(f'Warning: {not_arrived} points have not arrived!')
    
    return arrival

# Define this OUTSIDE of ul_simulation_batch, at module level
def _single_batch_worker(args):
    """Worker function for multiprocessing - must be at module level"""
    batch_idx, grad_fn, xdim, nx_per_batch, kbt, c_a, c_b, xinit, vinit, gamma, tstep, nstep, random_seed = args
    
    # Your original single_batch logic here
    seed = random_seed + batch_idx if random_seed is not None else None
    result = ul_simulation_target_1(grad_fn, xdim, nx_per_batch, kbt, c_a, c_b,
                                  xinit=xinit, vinit=vinit,
                                  gamma=gamma, tstep=tstep, nstep=nstep,
                                  random_seed=seed)
    return result


def ul_simulation_batch(grad_fn, xdim, Nx, kbt, c_a, c_b, xinit=None, vinit=None,
                        gamma=100, tstep=5e-3, nstep=10**6, random_seed=None, n_batches=4):
    
    
    nx_per_batch = Nx // n_batches
    
    # Prepare arguments for each batch
    task_args = []
    for batch_idx in range(n_batches):
        task_args.append((
            batch_idx, grad_fn, xdim, nx_per_batch, kbt, c_a, c_b,
            xinit[batch_idx * nx_per_batch:(batch_idx + 1) * nx_per_batch,:], 
            vinit[batch_idx * nx_per_batch:(batch_idx + 1) * nx_per_batch,:], 
            gamma, tstep, nstep, random_seed
        ))
    
    with Pool(n_batches) as pool:
        results = pool.map(_single_batch_worker, task_args)
    
    return np.concatenate(results)

if False:
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
    T = 10**10
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
                                        tstep=2e-5,
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

if __name__ == '__main__':
    muller = MullerPotential()
    grad_fn = Muller_grad
    gamma = 5
    kbt = 5
    q0 = np.loadtxt(f'muller_potential/model/fd_kbt{kbt}.txt')
    q = np.loadtxt(f'muller_potential/model/ave_qqq_kbt{kbt}_gamma{gamma}.txt')
    mask_q_tran = abs(q-0.5)<0.05
    print(mask_q_tran.sum())
    points = q0[:, :2]

    xmin, xmax = -1.5, 1.2
    ymin, ymax = -0.2, 2
    dx = 0.01
    dy = 0.01
    Nx = int((xmax - xmin) / dx)
    Ny = int((ymax - ymin) / dy)


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

    ppoints = np.array([X.reshape(-1), Y.reshape(-1)]).T.astype(np.float32)
    UU = muller.potential(ppoints).reshape(X.shape)
    UU[UU>0] = 0

    points = points[mask_q_tran]
    #points = np.loadtxt(f'./muller_potential/model/fd_kbt{kbt}_masked.txt')[:, :2]
    points = points[points[:,0]>-1,:]
    points = points[points[:,0]<-0.5,:]
    points = points[points[:,1]>0.3,:]
    points = points[points[:,1]<0.8,:]
    points = points[::100,:]
    mask_55 = [14,17,20,21,22,23,25,26,28]
    points = points[mask_55,:]
    
    
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.contour(X, Y, UU, levels=50, cmap='viridis')
    plt.scatter(points[:, 0], points[:, 1])
    plt.colorbar(label='Potential') 
    #plt.show()
    
    
    
    U = muller.potential(points)
    print(np.max(U))
    
    U_thr = 10
    points = points[U < U_thr]
    v_sample = 100

    x0 = points
    c_a = muller.c_a()
    c_b = muller.c_b()
    T = 10**6
    N = 1000
    stride = 10
    tstep = 1e-4
    arr = np.zeros(x0.shape[0])
    vs = np.random.normal(size=(v_sample, 2)) * np.sqrt(kbt)
    # np.savetxt(f'muller_potential/model/simulation_{kbt}/simulation_vconfig_kbt{kbt}.txt', vs)
    result = np.ones((points.shape[0], 3))
    result[:, :2] = points
    print(f'gamma: {gamma}, kbt: {kbt}, points num: {points.shape[0]}')
    v_sd = np.array([[0.01,-2.88],[3.05,0.74],[0.02,-4.78],[3.02,-0.94],[-2,0],[4,-4]])  # velocity at the saddle point
    sd = np.array([[-0.822, 0.624]])
    result_sd = np.ones((v_sd.shape[0], 3))
    result_sd[:, :2] = v_sd

    for i in range(v_sd.shape[0]):
        st = time.time()
        
        x0 = points[i, :]
        x0 = np.tile(x0, (vs.shape[0], 1))
        x0 = x0.repeat(N, axis=0)
        v0 = vs.repeat(N, axis=0)
        print(x0.shape, v0.shape)
        
        '''
        x0 = np.tile(sd, (N, 1))
        v0 = np.tile(v_sd[i], (N, 1))
        '''
        
        arr = ul_simulation_target_1(grad_fn,
                                        c_a=c_a,
                                        c_b=c_b,
                                        xdim=x0.shape[1],
                                        Nx=x0.shape[0],
                                        kbt=kbt,
                                        xinit=x0,
                                        vinit=v0,
                                        gamma=gamma,
                                        tstep=tstep,
                                        nstep=T,
                                        random_seed=None)
        
        
        '''
        arr =ul_simulation_batch(grad_fn, x0.shape[1], x0.shape[0], kbt, c_a, c_b,
                       gamma=5, tstep=tstep, nstep=T, n_batches=4,xinit=x0, vinit=v0, random_seed=None)
        '''
        result[i, 2] = np.sum(arr)/N/v_sample

        tt = time.time()
        print(f'point {i}: committor {result[i, 2]:.4f}, using time {tt-st}!')
    # np.savetxt('model/simulation_kbt.1_gamma10.txt', xs)
    # plt.scatter(xs[:, 0], xs[:, 1], alpha=0.05)
    # plt.show()
    np.savetxt(
            f'muller_potential/model/simulation_{kbt}/simulation_kbt{kbt}_gamma{gamma}_saddle.txt',
            result)
    