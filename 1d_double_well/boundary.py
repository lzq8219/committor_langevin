import numpy as np

N = 1000000


def sample_x1r(N, threshold):
    x1 = np.random.rand(N) * 3 - 1.5
    R = np.random.rand(N)
    l = ((x1**2 - 1)**2 + 0.3 * R**2) < threshold
    return x1[l].copy(), R[l].copy()


def sample_x(N, threshold, Nm1, ndim):
    x1, r = sample_x1r(N, threshold)
    print(x1.shape)
    x = np.zeros((len(x1) * Nm1, ndim), dtype=np.float32)
    for i in range(len(x1)):
        x[i * Nm1:(i + 1) * Nm1, 0] = x1[i]
        x[i * Nm1:(i + 1) * Nm1, 1:] = r[i] * \
            np.random.normal(size=(Nm1, ndim - 1))
    return x


if __name__ == "__main__":
    x = sample_x(N, 0.2, 10, 10)
    print(x.shape)
