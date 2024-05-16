# -*- coding: utf-8 -*-
# @Time    : 2024/4/22 16:23
# @Author  : 纪冠州
# @File    : haze_remove.py
# @Software: PyCharm 
# @Comment : haze remove

import os
import cv2
import numpy as np
from tqdm import tqdm
from numpy.lib import stride_tricks
from scipy.ndimage import minimum_filter
from scipy.sparse import coo_matrix, identity, lil_matrix
from scipy.sparse.linalg import cg, spsolve

patch_size = 15
brightest_ratio = 0.001
w = 0.95
eps = 1e-7
win_len = 3
lbd = 1e-4
t0 = 0.1


def dark_channel(img):
    """
    calculate the dark channel of the input image
    :param img: 3-dim numpy matrix with input image
    :return: 2-dim numpy matrix with input image's dark channel
    """
    return minimum_filter(np.min(img, axis=2), patch_size)


def estimate_A(imgO, imgD):
    """
    estimate the atmospheric light
    :param imgO: 3-dim numpy matrix with input image
    :param imgD: 2-dim numpy matrix with input image's dark channel
    :return: atmospheric light
    """
    brightest_num = int(brightest_ratio * imgD.size)
    max_indices = np.argpartition(imgD.flatten(), -brightest_num)[-brightest_num:]
    max_indices = np.unravel_index(max_indices, imgD.shape)
    max_indices = list(zip(*max_indices))

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    max_area = [gray[each] for each in max_indices]
    brightest_pixel_index = max_indices[np.argmax(max_area)]
    brightest_pixel = imgO[brightest_pixel_index]

    return brightest_pixel


def estimate_t(imgO, A):
    """
    estimate the transmission map
    :param imgO: 3-dim numpy matrix with input image
    :param A: atmospheric light
    :return: transmission map
    """
    t = 1 - w * dark_channel(imgO / A)
    return t


def _rolling_block(A, block=(3, 3)):
    """Applies sliding window to given matrix."""
    shape = (A.shape[0] - block[0] + 1, A.shape[1] - block[1] + 1) + block
    strides = (A.strides[0], A.strides[1]) + A.strides
    return stride_tricks.as_strided(A, shape=shape, strides=strides)


def compute_laplacian(img: np.ndarray, mask=None, eps: float = 10 ** (-7), win_rad: int = 1):
    """
    Computes Matting Laplacian for a given image.
    img: 3-dim numpy matrix with input image
    mask: mask of pixels for which Laplacian will be computed.
        If not set Laplacian will be computed for all pixels.
    eps: regularization parameter controlling alpha smoothness
        from Eq. 12 of the original paper. Defaults to 1e-7.
    win_rad: radius of window used to build Matting Laplacian (i.e.
        radius of omega_k in Eq. 12).
    Returns: sparse matrix holding Matting Laplacian.
    """
    win_size = (win_rad * 2 + 1) ** 2
    h, w, d = img.shape
    win_diam = win_rad * 2 + 1

    indsM = np.arange(h * w).reshape((h, w))
    ravelImg = img.reshape(h * w, d)
    win_inds = _rolling_block(indsM, block=(win_diam, win_diam)).reshape(-1, win_size)

    winI = ravelImg[win_inds]

    win_mu = np.mean(winI, axis=1, keepdims=True)
    win_var = np.einsum('...ji,...jk ->...ik', winI, winI) / win_size - np.einsum('...ji,...jk ->...ik', win_mu, win_mu)

    A = win_var + (eps / win_size) * np.eye(3)
    B = (winI - win_mu).transpose(0, 2, 1)
    X = np.linalg.solve(A, B).transpose(0, 2, 1)
    vals = np.eye(win_size) - (1.0 / win_size) * (1 + X @ B)

    nz_indsCol = np.tile(win_inds, win_size).ravel()
    nz_indsRow = np.repeat(win_inds, win_size).ravel()
    nz_indsVal = vals.ravel()
    L = coo_matrix((nz_indsVal, (nz_indsRow, nz_indsCol)), shape=(h * w, h * w))

    # L = scipy.sparse.csr_matrix((nz_indsVal, nz_indsCol, np.arange(0, nz_indsVal.shape[0] + 1, win_size)), shape=(h * w, h * w))
    return L


def estimate_t_with_soft_matting(imgO, t, atmosphere):
    """
    estimate the transmission map with soft matting
    :param imgO: 3-dim numpy matrix with input image
    :param t: transmission map before soft matting
    :return: transmission map after soft matting
    """
    print("calc Laplacian")
    L = compute_laplacian(imgO)
    print("Complete!")

    A = L + lbd * identity(L.shape[0])
    B = lbd * t.flatten()

    # 使用对角预条件器
    # diagonal = A.diagonal()
    # M = LinearOperator(A.shape, matvec=lambda x: x / diagonal)

    maxiters = 1000

    with tqdm(total=maxiters) as pbar:
        pbar.set_description('calc t')

        def calc_t_callback(x):
            pbar.update(1)

        # 未使用预处理器，求解缓慢。尝试使用预处理器但未成功，预处理器的使用方法有待进一步研究
        t_new, info = cg(A, B, x0=t.flatten(), maxiter=maxiters, tol=1e-6, callback=calc_t_callback)
        cv2.destroyAllWindows()

    t_new = t_new.reshape(imgO.shape[:2])

    return t_new


def Guidedfilter(img_gray, p, r):
    mean_I = cv2.boxFilter(img_gray, cv2.CV_64F, (r, r))
    mean_p = cv2.boxFilter(p, cv2.CV_64F, (r, r))
    mean_Ip = cv2.boxFilter(img_gray * p, cv2.CV_64F, (r, r))
    cov_Ip = mean_Ip - mean_I * mean_p

    mean_II = cv2.boxFilter(img_gray * img_gray, cv2.CV_64F, (r, r))
    var_I = mean_II - mean_I * mean_I

    a = cov_Ip / (var_I + eps)
    b = mean_p - a * mean_I

    mean_a = cv2.boxFilter(a, cv2.CV_64F, (r, r))
    mean_b = cv2.boxFilter(b, cv2.CV_64F, (r, r))

    q = mean_a * img_gray + mean_b
    return q


def TransmissionRefine(im, et):
    gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)
    gray = np.float64(gray) / 255
    r = 60
    t = Guidedfilter(gray, et, r)

    return t


def recovering_scene(imgO, t, A):
    """
    recover the scene
    :param imgO: 3-dim numpy matrix with input image
    :param t: transmission map
    :param A: atmospheric light
    :return: recovered scene
    """
    t_new = np.repeat(cv2.max(t, t0)[:, :, np.newaxis], 3, axis=2)
    A_new = np.ones(imgO.shape) * A.reshape((1, 1, 3))
    J = (imgO - A_new) / t_new + A_new
    return J


if __name__ == '__main__':
    img_path = './images/img_raw'
    t_path = './images/img_t'
    dehaze_path = './images/img_dehaze'
    imgs = os.listdir(img_path)
    for each in imgs:
        print(each)
        img = cv2.imread(img_path + "/" + each)

        img_dark = dark_channel(img)

        A = estimate_A(img, img_dark)

        t1 = estimate_t(img, A)
        cv2.imwrite(t_path + '/' + each.split('.')[0] + '_t1.jpg', t1 * 255)
        J1 = recovering_scene(img, t1, A)
        cv2.imwrite(dehaze_path + '/' + each.split('.')[0] + '_t1_dehaze.jpg', J1)

        # t2 = estimate_t_with_soft_matting(img, t1, A)
        t2 = TransmissionRefine(img, t1)
        cv2.imwrite(t_path + '/' + each.split('.')[0] + '_t2.jpg', t2 * 255)
        J2 = recovering_scene(img, t2, A)
        cv2.imwrite(dehaze_path + '/' + each.split('.')[0] + '_t2_dehaze.jpg', J2)

        # cv2.imshow('origin1', J1.astype(np.uint8))
        # cv2.imshow('origin2', J2.astype(np.uint8))
        # cv2.waitKey(0)

"""
def calc_coordinates(x, w):
    return x // w, x % w


def calc_Laplacian(img):
    eps = 1e-7
    r = int(win_len / 2)

    img_h, img_w, img_d = img.shape
    img_N = img_h * img_w

    img_w_mean = np.zeros((img_h, img_w, 3))
    img_w_cov = np.zeros((img_h, img_w, 3, 3))
    with tqdm(total=(img_h - 2) * (img_w - 2)) as pbar:
        pbar.set_description('mean and cov calc')
        for i in range(r, img_h - r):
            for j in range(r, img_w - r):
                w = img[i - r:i + r + 1, j - r:j + r + 1]
                img_w_mean[i, j] = np.mean(w, axis=(0, 1))
                img_w_cov[i, j] = np.cov(w.reshape(9, 3).T)
                pbar.update(1)

    L_row = []
    L_col = []
    L_val = []

    with tqdm(total=img_N ** 2) as pbar:
        pbar.set_description('Laplacian calc')

        for L_i in range(img_N):
            for L_j in range(img_N):
                i = calc_coordinates(L_i, img_w)
                j = calc_coordinates(L_j, img_w)

                Kronecker_delta = 1 if L_i == L_j else 0

                if abs(i[0] - j[0]) < win_len and abs(i[1] - j[1]) < win_len:
                    r_i = win_len - abs(i[0] - j[0]) - 1
                    r_j = win_len - abs(i[1] - j[1]) - 1
                    # print(i[0], j[0])

                    w_is = [t for t in range(max(min(i[0], j[0]) - r_i, 1), min(max(i[0], j[0]) + r_i + 1, img_h - 1))]
                    w_js = [t for t in range(max(min(i[1], j[1]) - r_j, 1), min(max(i[1], j[1]) + r_j + 1, img_w - 1))]
                    # print(w_is, w_js)

                    L_row.append(L_i)
                    L_col.append(L_j)
                    L_val_tmp = 0

                    for w_i in w_is:
                        for w_j in w_js:
                            L_val_tmp += Kronecker_delta - (
                                    1 + (img[i[0], i[1]] - img_w_mean[w_i, w_j]).T @ np.linalg.inv(
                                img_w_cov[w_i, w_j] - np.eye(3) * eps / win_len ** 2) @ (
                                            img[j[0], j[1]] - img_w_mean[w_i, w_j])) / win_len ** 2

                    L_val.append(L_val_tmp)

                pbar.update(1)

        Laplacian = coo_matrix((L_val, (L_row, L_col)), shape=(img_N, img_N), dtype=np.float32)

    return Laplacian
"""
