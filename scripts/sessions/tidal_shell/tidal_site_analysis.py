"""Where and how a frictional stripe would slip asymmetrically: from the locked
shell's global stress (gstress_*.npz, last period), the shear and normal traction
on a plane of given azimuth at every grid point, their amplitudes, the phase of
the clamping relative to the shear, and an asymmetry index from the paper's 1-D
Coulomb picture (ED Fig. 4): slip rate ~ max(0, |tau| - mu (T_b - sigma_n)) with
the sign of tau, so

    asym = (I_right - I_left) / (I_right + I_left),  I = integral of the positive
    Coulomb traction over the right- / left-lateral windows.

Azimuth: the plane's strike measured counter-clockwise from local east (viewed
from outside). Usage: <sim> <run> <mu> <T_b> <azimuth deg | 'best'>"""
import glob, os, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sim, run = sys.argv[1:3]
mu, Tb = float(sys.argv[3]), float(sys.argv[4])
az_arg = sys.argv[5] if len(sys.argv) > 5 else "120"
files = sorted(glob.glob(os.path.join(sim, run, "gstress_*.npz")))
d0 = np.load(files[-1]); LON, LAT = d0["lon"], d0["lat"]
t_all = np.array([float(np.load(f)["t"]) for f in files])
last = t_all > t_all.max() - 1.0 + 1e-9
S = np.array([np.load(f)["S"] for f, m in zip(files, last) if m])          # (nt, npts, 3, 3)
t = t_all[last]
lon, lat = LON.ravel(), LAT.ravel()
r = np.column_stack([np.cos(lat) * np.cos(lon), np.cos(lat) * np.sin(lon), np.sin(lat)])
east = np.column_stack([-np.sin(lon), np.cos(lon), np.zeros_like(lon)])
north = np.cross(r, east)


def tractions(az_deg):
    a = np.radians(az_deg)
    strike = np.cos(a) * east + np.sin(a) * north
    nrm = np.cross(r, strike)
    tau = np.einsum("ni,tnij,nj->tn", strike, S, nrm)
    sn = np.einsum("ni,tnij,nj->tn", nrm, S, nrm)
    return tau, sn


def analyse(tau, sn):
    M = np.column_stack([np.cos(2 * np.pi * t), np.sin(2 * np.pi * t)])
    ct = np.linalg.lstsq(M, tau, rcond=None)[0]; cn = np.linalg.lstsq(M, sn, rcond=None)[0]
    amp_t, amp_n = np.hypot(*ct), np.hypot(*cn)
    ph_t, ph_n = np.arctan2(ct[1], ct[0]), np.arctan2(cn[1], cn[0])
    dphi = np.degrees((ph_n - ph_t + np.pi) % (2 * np.pi) - np.pi)       # clamping phase relative to shear
    coul = np.abs(tau) - mu * (Tb - sn)
    pos = np.maximum(coul, 0.0)
    I_r = (pos * (tau > 0)).sum(axis=0); I_l = (pos * (tau < 0)).sum(axis=0)
    asym = (I_r - I_l) / np.maximum(I_r + I_l, 1e-30)
    frac = (pos > 0).mean(axis=0)                                        # fraction of the cycle failing
    # timing: the centroid of each failure window relative to the shear extreme of
    # its sign (degrees, positive = late); the difference between the two windows
    ph = 2 * np.pi * t
    def centroid_offset(sign):
        w = pos * (np.sign(tau) == sign)
        ex = ph_t if sign > 0 else ph_t + np.pi                           # phase of the shear extreme
        rel = (ph[:, None] - ex[None, :] + np.pi) % (2 * np.pi) - np.pi
        return np.degrees((w * rel).sum(axis=0) / np.maximum(w.sum(axis=0), 1e-30))
    timing = centroid_offset(+1) - centroid_offset(-1)
    timing = np.where((I_r > 0) & (I_l > 0), timing, np.nan)
    return amp_t, amp_n, dphi, asym, frac, timing


amp_t, amp_n, dphi, asym, frac, timing = analyse(*tractions(float(az_arg)))
title = f"planes at azimuth {az_arg} deg from local east"
fields = [(amp_t, "shear traction amplitude", "viridis", None), (amp_n, "normal traction amplitude", "viridis", None),
          (dphi, "phase of the clamping relative to the shear (deg)", "twilight", (-180, 180)),
          (asym, "size asymmetry (I_right - I_left)/(I_right + I_left)", "RdBu_r", (-1, 1)),
          (timing, "timing asymmetry: onset offset, right minus left window (deg)", "RdBu_r", (-90, 90)),
          (frac, "fraction of the cycle at failure", "magma", (0, 1))]
n = len(fields)
fig, axes = plt.subplots(2, 3, figsize=(19, 8), subplot_kw={"projection": "mollweide"})
for ax, (fld, ttl, cmap, lim) in zip(axes.ravel(), fields):
    kw = {} if lim is None else {"vmin": lim[0], "vmax": lim[1]}
    pc = ax.pcolormesh(LON, LAT, fld.reshape(LAT.shape), cmap=cmap, shading="auto", **kw)
    ax.set_title(ttl, fontsize=9, loc="left"); ax.grid(alpha=0.3)
    fig.colorbar(pc, ax=ax, shrink=0.6, orientation="horizontal", pad=0.05)
for ax in axes.ravel()[n:]:
    ax.set_visible(False)
fig.suptitle(f"locked shell {run}, mu = {mu}, T_b = {Tb}: {title}", fontsize=11)
out = os.path.join(sim, f"sites_{run}_az{az_arg}.png")
fig.savefig(out, dpi=110, bbox_inches="tight"); print("wrote", out)
# the sites run frictionally
for name, la, lo, az in (("pole az30", -90, 0, 30), ("pole az90", -90, 0, 90), ("lat-45 lon0 az30", -45, 0, 30),
                         ("lat-45 lon90 az30", -45, 90, 30), ("equator lon0 az45", 0, 0, 45)):
    i = np.argmin((np.degrees(lat) - la) ** 2 + (np.cos(np.radians(la)) * (np.degrees(lon) - lo)) ** 2)
    # local azimuth from east: the driver's azimuth is from the sub-planet direction; at the
    # south pole that direction is +x, which is "north" in the local (east, north) frame
    # of the pole's lat-lon grid at lon 0 -> azimuth from east = az + 90 there; at other
    # sites the sub-planet direction is local west-ish (lon 0) or south-ish (lon 90)
    for az_e in (az + 90, az):
        a_t, a_n, dp, asy, fr, tm = analyse(*tractions(az_e % 180))
        print(f"  {name:20s} plane az_east {az_e % 180:5.0f}: |tau| {a_t[i]:.4f} |sn| {a_n[i]:.4f} clamp phase {dp[i]:+6.0f} "
              f"size asym {asy[i]:+.2f} timing asym {tm[i]:+6.0f} failing {fr[i]:.2f}")
