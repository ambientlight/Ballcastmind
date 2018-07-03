import numpy as np


def hough_transform(img_bin, theta_res=1, rho_res=1):
    n_r, n_c = img_bin.shape
    theta = np.linspace(-90.0, 0.0, np.ceil(90.0 / theta_res) + 1.0)
    theta = np.concatenate((theta, -theta[len(theta) - 2::-1]))

    D = np.sqrt((n_r - 1) ** 2 + (n_c - 1) ** 2)
    q = np.ceil(D / rho_res)
    nrho = 2 * q + 1
    rho = np.linspace(-q * rho_res, q * rho_res, nrho)
    h = np.zeros((len(rho), len(theta)))
    for row_idx in range(n_r):
        for col_idx in range(n_c):
            if img_bin[row_idx, col_idx]:
                for thIdx in range(len(theta)):
                    rho_val = col_idx * np.cos(theta[thIdx] * np.pi / 180.0) + \
                             row_idx * np.sin(theta[thIdx] * np.pi / 180)
                    rho_idx = np.nonzero(np.abs(rho - rho_val) == np.min(np.abs(rho - rho_val)))[0]
                    h[rho_idx[0], thIdx] += 1

    print(theta)
    print(f'D: {D}')
    print(f'q: {q}')
    print(f'nrho: {nrho}')
    print(f'rho: {rho}, {len(rho)}')
    print(h)
    return rho, theta, h
