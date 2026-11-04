#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Simulasi Reaction-Diffusion-Advection (RDA) penyebaran karhutla gambut Kalimantan.

  dF/dt + Ue dF/dx + Ve dF/dy = div(D grad F) + r exp(-alpha m) F (1 - F/K) [* B] - mu F (1 - B)

Skema FDM: difusi = central difference (bentuk flux, D variabel), adveksi = upwind orde-1,
waktu = Euler eksplisit. Satuan internal: km dan jam.

Data masukan
  - data_1.grib          : ERA5 u100, v100 per jam (GRIB1, tanpa library eksternal)
  - Indonesia_peat_lands.csv : hanya atribut (TANPA geometri) -> mask gambut G harus
                              diganti shapefile/GeoTIFF asli (lihat load_peat_mask)
  - hotspot (opsional)   : CSV FIRMS (latitude, longitude, acq_date, acq_time)
  - soil moisture (opsional): belum ada di GRIB; lihat load_soil_moisture

Jalankan:  python rda_karhutla.py --grib data_1.grib --start 2026-09-01 --days 30 --out hasil
"""
import argparse, os, datetime as dt
import numpy as np
from scipy.ndimage import gaussian_filter, map_coordinates

# --------------------------------------------------------------------------
# 1. Pembaca GRIB1 (simple packing) tanpa eccodes/cfgrib
# --------------------------------------------------------------------------
def _s(b):
    v = int.from_bytes(b, "big"); n = len(b) * 8
    return -(v & ~(1 << (n - 1))) if v >> (n - 1) else v

def _ibm(b):
    v = int.from_bytes(b, "big")
    s = -1 if v >> 31 else 1
    return s * (v & 0xFFFFFF) / 16777216.0 * 16.0 ** (((v >> 24) & 0x7F) - 64)

def read_grib1(path):
    """Return dict param-> (times, lat(S->N), lon, array[t, lat, lon])."""
    b = open(path, "rb").read(); pos = 0; rec = []; grid = None
    while True:
        i = b.find(b"GRIB", pos)
        if i < 0: break
        L = int.from_bytes(b[i + 4:i + 7], "big"); m = b[i:i + L]; pos = i + L
        p = 8; pl = int.from_bytes(m[p:p + 3], "big"); pds = m[p:p + pl]; p += pl
        flag = pds[7]
        t = dt.datetime(100 * (pds[24] - 1) + pds[12], pds[13], pds[14], pds[15], pds[16])
        dsf = _s(pds[26:28]); prm = pds[8]
        if flag & 0x80:
            gl = int.from_bytes(m[p:p + 3], "big"); g = m[p:p + gl]; p += gl
            grid = dict(ni=int.from_bytes(g[6:8], "big"), nj=int.from_bytes(g[8:10], "big"),
                        lat1=_s(g[10:13]) / 1e3, lon1=_s(g[13:16]) / 1e3,
                        lat2=_s(g[17:20]) / 1e3, lon2=_s(g[20:23]) / 1e3)
        if flag & 0x40: p += int.from_bytes(m[p:p + 3], "big")
        bl = int.from_bytes(m[p:p + 3], "big"); bds = m[p:p + bl]
        E = _s(bds[4:6]); ref = _ibm(bds[6:10]); nb = bds[10]; data = bds[11:]
        bits = np.unpackbits(np.frombuffer(data, dtype=np.uint8))
        n = len(bits) // nb
        X = (bits[:n * nb].reshape(n, nb) * (1 << np.arange(nb - 1, -1, -1))).sum(1).astype(float)
        vals = (ref + X * 2.0 ** E) / 10.0 ** dsf
        rec.append((prm, t, vals[:grid["ni"] * grid["nj"]]))
    lat = np.linspace(grid["lat1"], grid["lat2"], grid["nj"])      # N -> S
    lon = np.linspace(grid["lon1"], grid["lon2"], grid["ni"])
    out = {}
    for prm in sorted(set(r[0] for r in rec)):
        rr = sorted([r for r in rec if r[0] == prm], key=lambda r: r[1])
        T = [r[1] for r in rr]
        A = np.stack([r[2].reshape(grid["nj"], grid["ni"]) for r in rr])[:, ::-1, :]  # S -> N
        out[prm] = (T, lat[::-1].copy(), lon, A)
    return out

PARAM = {165: "u10", 166: "v10", 167: "t2m", 246: "u100", 247: "v100"}   # ECMWF table 128/228

# --------------------------------------------------------------------------
# 2. Grid, interpolasi, input lingkungan
# --------------------------------------------------------------------------
def make_grid(lat0, lat1, lon0, lon1, dlon):
    lat = np.arange(lat0, lat1 + 1e-9, dlon); lon = np.arange(lon0, lon1 + 1e-9, dlon)
    LAT, LON = np.meshgrid(lat, lon, indexing="ij")
    dx = 111.32 * np.cos(np.deg2rad(LAT)) * dlon      # km (bervariasi tipis menurut lintang)
    dy = 111.32 * dlon
    return lat, lon, LAT, LON, dx, dy

def regrid(field2d, lat_s, lon_s, LAT, LON):
    """Interpolasi bilinear grid sumber -> grid model."""
    iy = (LAT - lat_s[0]) / (lat_s[1] - lat_s[0]); ix = (LON - lon_s[0]) / (lon_s[1] - lon_s[0])
    return map_coordinates(field2d, [iy, ix], order=1, mode="nearest")

def load_peat_mask(LAT, LON, path=None, seed=7, peat_fraction=0.45):
    """
    G(x,y) in {0,1}.
    Indonesia_peat_lands.csv TIDAK memuat koordinat, jadi mask asli belum bisa dibuat.
    Jika `path` (GeoTIFF/raster .npy) diberikan, dipakai. Jika tidak: mask SINTETIS
    (hanya untuk demonstrasi kode; BUKAN peta gambut nyata).
    Cara membuat mask asli: rasterisasi shapefile gambut (geopandas + rasterio.features.rasterize)
    ke grid ini, dengan nilai 1 untuk layer_revi == 'gambut'.
    """
    if path and path.endswith(".npy"):
        return np.load(path).astype(float)
    rng = np.random.default_rng(seed)
    z = gaussian_filter(rng.standard_normal(LAT.shape), sigma=LAT.shape[0] / 14)
    z = (z - z.mean()) / z.std()
    return (z > np.quantile(z, 1 - peat_fraction)).astype(float)

def load_soil_moisture(LAT, LON, m_const=0.3, path=None):
    """
    m(x,y) dalam [0,1] (kelembapan relatif). data.grib TIDAK berisi soil moisture
    (hanya u10, v10, t2m). Ganti dengan ERA5 'volumetric_soil_water_layer_1' (swvl1,
    param 39) lalu normalisasi: m = (swvl1 - swvl_min) / (swvl_max - swvl_min).
    Default: nilai konstan skenario `m_const`.
    """
    if path and path.endswith(".npy"):
        return np.load(path)
    return np.full(LAT.shape, float(m_const))

# --------------------------------------------------------------------------
# 3. Inti numerik FDM
# --------------------------------------------------------------------------
def step(F, B, U, V, D, r, alpha, m, K, dx, dy, dt_h, mu, use_fuel, beta=0.004):
    # --- difusi: div(D grad F), bentuk flux, D dirata-ratakan di muka sel, Neumann nol
    Fp = np.pad(F, 1, mode="edge"); Dp = np.pad(D, 1, mode="edge")
    De = 0.5 * (Dp[1:-1, 1:-1] + Dp[1:-1, 2:]);  Dw = 0.5 * (Dp[1:-1, 1:-1] + Dp[1:-1, :-2])
    Dn = 0.5 * (Dp[1:-1, 1:-1] + Dp[2:, 1:-1]);  Ds = 0.5 * (Dp[1:-1, 1:-1] + Dp[:-2, 1:-1])
    diff = (De * (Fp[1:-1, 2:] - F) - Dw * (F - Fp[1:-1, :-2])) / dx**2 \
         + (Dn * (Fp[2:, 1:-1] - F) - Ds * (F - Fp[:-2, 1:-1])) / dy**2
    # --- adveksi upwind: U>0 -> beda mundur, U<0 -> beda maju
    dFdx_b = (F - Fp[1:-1, :-2]) / dx; dFdx_f = (Fp[1:-1, 2:] - F) / dx
    dFdy_b = (F - Fp[:-2, 1:-1]) / dy; dFdy_f = (Fp[2:, 1:-1] - F) / dy
    adv = np.where(U > 0, U * dFdx_b, U * dFdx_f) + np.where(V > 0, V * dFdy_b, V * dFdy_f)
    # --- reaksi logistik dihambat kelembapan
    growth = r * np.exp(-alpha * m) * F * (1 - F / K)
    if use_fuel:
        growth = growth * B
        burnout = mu * F * (1 - B)
        B_new = np.clip(B - dt_h * (F / K) * B * beta, 0, 1)    # konsumsi bahan bakar
    else:
        burnout = 0.0; B_new = B
    F_new = F + dt_h * (-adv + diff + growth - burnout)
    return np.clip(F_new, 0.0, K), B_new

def cfl_check(D, U, V, dx, dy, dt_h):
    d = float(np.max(dt_h * np.max(D) * (1 / dx**2 + 1 / dy**2) * 2))   # <= 1 stabil (difusi)
    a = float(np.max(dt_h * (np.abs(U) / dx + np.abs(V) / dy)))        # <= 1 stabil (adveksi)
    return d, a

def run(params, grib_path, t_start, days, ignition, out_frames=None, verbose=True):
    """Satu simulasi. Return dict {F_final, frames, times, G, grid, dt}."""
    P = dict(dlon=0.02, dt_h=1.0, K=1.0, Dg=0.05, Dn=0.01, rg=0.05, rn=0.01,
             alpha=2.0, m_const=0.3, kappa=0.01, Fc=0.5, mu=0.01, beta=0.004, use_fuel=True,
             bbox=(-3.0, -1.0, 111.0, 116.0), peat_path=None, sm_path=None, uparam=246)
    P.update(params)
    lat, lon, LAT, LON, dx, dy = make_grid(*P["bbox"], P["dlon"])
    G = load_peat_mask(LAT, LON, P["peat_path"])
    m = load_soil_moisture(LAT, LON, P["m_const"], P["sm_path"])
    D = P["Dg"] * G + P["Dn"] * (1 - G); r = P["rg"] * G + P["rn"] * (1 - G)
    g = read_grib1(grib_path) if "_g" not in P else P["_g"]
    up = P["uparam"]; T, lat_s, lon_s, _ = g[up]
    t0 = int((t_start - T[0]).total_seconds() // 3600); n = int(days * 24 / P["dt_h"])\n    if t0 < 0 or t0 + n > len(T):\n        raise ValueError(f"Data angin u100/v100 tidak mencakup seluruh simulasi: diminta indeks jam {t0}..{t0+n-1}, tersedia 0..{len(T)-1}")
    F = np.zeros(LAT.shape); B = np.ones(LAT.shape)
    for (la, lo, amp, rad) in ignition:          # titik api awal (Gaussian kecil)
        if P.get("snap_peat", True):             # geser ke sel gambut terdekat
            dd = (LAT - la) ** 2 + (LON - lo) ** 2 + 1e6 * (G < 0.5)
            la, lo = LAT.flat[dd.argmin()], LON.flat[dd.argmin()]
        d2 = ((LAT - la) * 111.32) ** 2 + ((LON - lo) * 111.32 * np.cos(np.deg2rad(la))) ** 2
        F = np.maximum(F, amp * np.exp(-d2 / (2 * rad**2)))
    frames, times = [], []
    save_every = max(1, n // (out_frames or 6))
    for k in range(n):
        h = t0 + int(k * P["dt_h"])
        U = P["kappa"] * 3.6 * regrid(g[up][3][h], lat_s, lon_s, LAT, LON)   # m/s -> km/jam, x kappa
        V = P["kappa"] * 3.6 * regrid(g[up + 1][3][h], lat_s, lon_s, LAT, LON)
        if k == 0:
            cd, ca = cfl_check(D, U, V, dx, dy, P["dt_h"])
            if verbose: print(f"CFL difusi={cd:.3f} adveksi={ca:.3f} (harus <1)")
            assert cd < 1 and ca < 1, "dt terlalu besar: kecilkan dt_h atau perbesar dlon"
        F, B = step(F, B, U, V, D, r, P["alpha"], m, P["K"], dx, dy, P["dt_h"], P["mu"], P["use_fuel"], P["beta"])
        if k % save_every == 0 or k == n - 1:
            frames.append(F.copy()); times.append(T[0] + dt.timedelta(hours=h))
    return dict(F=F, frames=frames, times=times, G=G, m=m, LAT=LAT, LON=LON, dx=dx, dy=dy, P=P)

# --------------------------------------------------------------------------
# 4. Ekstraksi front, kecepatan, metrik validasi
# --------------------------------------------------------------------------
def burned(F, Fc): return F >= Fc

def area_km2(F, Fc, dx, dy): return float(np.sum(burned(F, Fc) * dx * dy))

def iou(A, B): 
    u = np.logical_or(A, B).sum(); return float(np.logical_and(A, B).sum() / u) if u else 1.0

def dice(A, B):
    s = A.sum() + B.sum(); return float(2 * np.logical_and(A, B).sum() / s) if s else 1.0

def rmse(a, b): return float(np.sqrt(np.mean((a - b) ** 2)))
def mae(a, b):  return float(np.mean(np.abs(a - b)))
def r2(obs, mod):
    ss = np.sum((obs - obs.mean()) ** 2); return float(1 - np.sum((obs - mod) ** 2) / ss) if ss else float("nan")

def front_speed(frames, times, Fc, dx, dy):
    """v_fire efektif (km/jam) dari akar luas: R = sqrt(A/pi), v = dR/dt."""
    R = np.array([np.sqrt(area_km2(f, Fc, dx, dy) / np.pi) for f in frames])
    th = np.array([(t - times[0]).total_seconds() / 3600 for t in times])
    return th, R, np.gradient(R, th)

def hotspots_to_grid(csv_path, LAT, LON, date, buffer_cells=1):
    """Rasterisasi hotspot FIRMS (kolom latitude, longitude, acq_date) menjadi mask biner."""
    import pandas as pd
    from scipy.ndimage import binary_dilation
    h = pd.read_csv(csv_path); h = h[h["acq_date"] == str(date)]
    M = np.zeros(LAT.shape, bool); la0, lo0 = LAT[0, 0], LON[0, 0]; d = LAT[1, 0] - LAT[0, 0]
    for a, o in zip(h["latitude"], h["longitude"]):
        i, j = int(round((a - la0) / d)), int(round((o - lo0) / d))
        if 0 <= i < M.shape[0] and 0 <= j < M.shape[1]: M[i, j] = True
    return binary_dilation(M, iterations=buffer_cells)

# --------------------------------------------------------------------------
# 5. Eksperimen: baseline, sensitivitas, twin experiment
# --------------------------------------------------------------------------
def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt, json
    ap = argparse.ArgumentParser()
    ap.add_argument("--grib", default="data_1.grib"); ap.add_argument("--out", default="hasil")
    ap.add_argument("--start", default="2026-09-01"); ap.add_argument("--days", type=float, default=30)
    ap.add_argument("--dlon", type=float, default=0.02)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)
    g = read_grib1(a.grib); t_start = dt.datetime.fromisoformat(a.start)
    ign = [(-2.6, 113.4, 1.0, 5.0), (-1.9, 114.3, 1.0, 5.0)]     # titik api awal ilustratif
    base = dict(dlon=a.dlon, _g=g)
    res = {}

    # (a) baseline
    R0 = run(base, a.grib, t_start, a.days, ign, out_frames=4)
    dx, dy, Fc = R0["dx"], R0["dy"], R0["P"]["Fc"]
    res["baseline_area_km2"] = area_km2(R0["F"], Fc, dx, dy)

    # (b) sensitivitas kelembapan m
    sm = {}
    for m_ in [0.0, 0.2, 0.4, 0.6, 0.8]:
        R = run({**base, "m_const": m_}, a.grib, t_start, a.days, ign, verbose=False)
        sm[m_] = area_km2(R["F"], Fc, dx, dy)
    res["area_vs_m"] = sm

    # (c) sensitivitas angin (kappa=0: tanpa adveksi)
    sk = {}
    for k_ in [0.0, 0.002, 0.004, 0.008]:
        R = run({**base, "kappa": k_}, a.grib, t_start, a.days, ign, verbose=False)
        la, lo = R["LAT"], R["LON"]; w = R["F"]
        cm = (np.sum(w * la) / w.sum(), np.sum(w * lo) / w.sum()) if w.sum() > 0 else (np.nan, np.nan)
        sk[k_] = dict(area=area_km2(w, Fc, dx, dy), centroid=cm)
    res["wind"] = {str(k): v for k, v in sk.items()}

    # (d) sensitivitas parameter lain (satu per satu, +-50%)
    sp = {}
    for nm in ["Dg", "rg", "alpha", "kappa"]:
        for f_ in [0.5, 1.5]:
            R = run({**base, nm: R0["P"][nm] * f_}, a.grib, t_start, a.days, ign, verbose=False)
            sp[f"{nm}x{f_}"] = area_km2(R["F"], Fc, dx, dy)
    res["oat"] = sp

    # (e) twin experiment: "observasi" = simulasi theta_true; model memakai theta salah
    truth = run({**base, "m_const": 0.3, "rg": 0.06}, a.grib, t_start, a.days, ign, verbose=False)
    obs = burned(truth["F"], Fc); tw = {}
    for lab, pr in {"theta_benar": {"rg": 0.06}, "rg_salah": {"rg": 0.03}, "m_salah": {"rg": 0.06, "m_const": 0.7}}.items():
        R = run({**base, **pr}, a.grib, t_start, a.days, ign, verbose=False)
        mod = burned(R["F"], Fc)
        tw[lab] = dict(IoU=iou(obs, mod), Dice=dice(obs, mod), RMSE=rmse(truth["F"], R["F"]), MAE=mae(truth["F"], R["F"]))
    res["twin"] = tw

    # (f) kecepatan front
    th, Rr, v = front_speed(R0["frames"], R0["times"], Fc, dx, dy)
    res["v_fire_km_per_day_mean"] = float(np.mean(v[1:]) * 24)

    json.dump(res, open(os.path.join(a.out, "hasil.json"), "w"), indent=1, default=float)
    print(json.dumps(res, indent=1, default=float))

    # ---- gambar
    ext = [R0["LON"].min(), R0["LON"].max(), R0["LAT"].min(), R0["LAT"].max()]
    fig, ax = plt.subplots(1, len(R0["frames"]), figsize=(4 * len(R0["frames"]), 4), sharey=True)
    for k, (f, t) in enumerate(zip(R0["frames"], R0["times"])):
        ax[k].imshow(R0["G"], origin="lower", extent=ext, cmap="Greens", alpha=0.35)
        ax[k].imshow(np.ma.masked_less(f, 0.05), origin="lower", extent=ext, cmap="YlOrRd", vmin=0, vmax=1)
        ax[k].contour(R0["LON"], R0["LAT"], f, levels=[Fc], colors="k", linewidths=0.8)
        ax[k].set_title(t.strftime("%d %b %H:%M")); ax[k].set_xlabel("Bujur")
    ax[0].set_ylabel("Lintang"); plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig1_front.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    ax[0].plot(list(sm), list(sm.values()), "o-"); ax[0].set_xlabel("m"); ax[0].set_ylabel("Luas terbakar (km²)")
    ax[1].plot([float(k) for k in sk], [v["area"] for v in sk.values()], "s-"); ax[1].set_xlabel("κ (koef. angin)")
    ax[2].plot(th / 24, Rr, "^-"); ax[2].set_xlabel("Hari"); ax[2].set_ylabel("R ekuivalen (km)")
    for x in ax: x.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig2_sensitivitas.png"), dpi=150); plt.close()

    fig, ax = plt.subplots(figsize=(6, 5)); step_ = 20
    ax.quiver(R0["LON"][::step_, ::step_], R0["LAT"][::step_, ::step_],
              regrid(g[246][3][int((t_start - g[246][0][0]).total_seconds() // 3600)], g[246][1], g[246][2], R0["LAT"], R0["LON"])[::step_, ::step_],
              regrid(g[247][3][int((t_start - g[246][0][0]).total_seconds() // 3600)], g[246][1], g[246][2], R0["LAT"], R0["LON"])[::step_, ::step_])
    ax.set_title("Angin 100 m (ERA5) pada " + a.start); plt.savefig(os.path.join(a.out, "fig3_angin.png"), dpi=150); plt.close()

if __name__ == "__main__":
    main()
