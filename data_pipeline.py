import os
from typing import Tuple, Dict

import numpy as np
import healpy as hp
import pysm3
from astropy import units as u

# Global settings
nside = 512
fsky = 0.1
freqs = [27, 39, 93, 145, 225, 280]
use_goal_depth = True
sat_depth = {
    27:  (30.0, 20.0),
    39:  (18.0, 12.0),
    93:  (2.6,  1.9),
    145: (3.3,  2.1),
    225: (7.0,  5.0),
    280: (16.0, 10.0),
}


def create_mask(nside: int, fsky: float) -> np.ndarray:
    """Random full-sky mask with fraction fsky."""
    npix = hp.nside2npix(nside)
    mask = np.zeros(npix, bool)
    keep = int(fsky * npix)
    idx = np.random.default_rng(0).choice(npix, keep, replace=False)
    mask[idx] = True
    return mask


def generate_cmb_map(nside: int, r: float) -> np.ndarray:
    """
    Returns a (3, npix) I,Q,U map made of:
      1) PySM3 'c1' scalar CMB (unlensed, B=0)
      2) plus a primordial BB only realization at tensor-to-scalar r
    """
    # 1) unlensed scalar CMB
    sky0 = pysm3.Sky(nside=nside, preset_strings=["c1"])
    cmb0 = sky0.get_emission(150 * u.GHz).value   # shape (3, npix)

    # 2) build BB-only component
    ellmax = 3 * nside - 1
    ells = np.arange(ellmax + 1)
    p0 = 1e-9 * (ells / 80) ** (-2.5)
    cls = np.zeros((4, ellmax + 1))
    cls[2] = r * p0  # only BB
    np.random.seed(0)
    alm = hp.synalm(cls, lmax=ellmax, new=True)  # returns (lmax+1)^2 coeffs
    bb_map = hp.alm2map(alm, nside, pixwin=True)
    cmb1 = np.stack([np.zeros_like(bb_map), np.zeros_like(bb_map), bb_map], axis=0)

    return cmb0 + cmb1


def generate_fg_maps(nside: int, freqs: list[int], seed: int = None) -> Dict[int, np.ndarray]:
    """
    Draws foreground IQU maps at each frequency.
    Returns {freq: array(shape=(3, npix))}.
    """
    if seed is not None:
        np.random.seed(seed)
    sky = pysm3.Sky(nside=nside, preset_strings=["d1", "s1", "a2"])
    out = {}
    for f in freqs:
        qty = sky.get_emission(f * u.GHz)
        out[f] = qty.value
    return out


def generate_white_noise(nside: int, depth: float, seed: int = None) -> np.ndarray:
    """
    White Gaussian noise maps at each freq band.
    """
    if seed is not None:
        np.random.seed(seed)
    npix = hp.nside2npix(nside)
    omega_sr = 4 * np.pi / npix
    omega_arc2 = omega_sr * (180 * 60 / np.pi) ** 2
    sigma_pix = depth / np.sqrt(omega_arc2)
    return np.random.normal(0, sigma_pix, (3, npix))


def simulate_patch(i: int, r: float) -> Tuple[np.ndarray, np.ndarray]:
    """
    Simulates one patch: returns (maps, true_cmb_b_alm).
      - maps: ndarray shape (n_freq, 3, npix)
      - true_cmb_b_alm: ndarray shape (#alm,) the B-mode alm
    """
    fg_seed, noise_seed = 1000 + i, 2000 + i

    # 1) CMB + BB
    cmb_IQU = generate_cmb_map(nside, r=r)          # (3, npix)
    true_b_alm = hp.map2alm(cmb_IQU, pol=True)[2]   # extract the B alm

    # 2) Foregrounds + stacking
    fg = generate_fg_maps(nside, freqs, seed=fg_seed)
    maps = []
    for f in freqs:
        depth = sat_depth[f][1] if use_goal_depth else sat_depth[f][0]
        noise = generate_white_noise(nside, depth, seed=noise_seed + f)
        maps.append(fg[f] + cmb_IQU + noise)

    return np.stack(maps, axis=0), true_b_alm


def compute_analytic_ilc(bmA: Dict[int, np.ndarray],
                         bmB: Dict[int, np.ndarray],
                         freqs: list[int],
                         mask: np.ndarray) -> np.ndarray:
    """
    Performs analytic ILC on two sets of splits (bmA, bmB) over given mask.
    Returns the gnomonic-projected cleaned patch (128×128).
    """
    D_A = np.stack([bmA[f][mask] for f in freqs], axis=0)
    D_B = np.stack([bmB[f][mask] for f in freqs], axis=0)
    C = np.cov(D_A)
    ridge = 1e-6 * np.trace(C) / C.shape[0]
    iC = np.linalg.inv(C + np.eye(C.shape[0]) * ridge)
    w = iC.dot(np.ones(C.shape[0], dtype=np.float32))
    w /= w.sum()
    hp_map = w @ D_B  # weighted sum
    return hp.gnomview(
        hp_map, rot=(0, -57),
        xsize=128, ysize=128,
        reso=10 * 60 / 128,
        return_projected_map=True,
        notext=True, cbar=False, no_plot=True
    )


def stamp2healpix(stamp: np.ndarray, nside: int) -> np.ndarray:
    """Inverse gnomonic: 128×128 stamp → full-sky Healpix map."""
    return hp.gnom2map(
        stamp, nside,
        rot=(0, -57),
        xsize=128, ysize=128,
        reso=10 * 60 / 128,
        pixwin=True,
        verbose=False
    )
