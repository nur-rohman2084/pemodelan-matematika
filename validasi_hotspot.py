#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Validasi dasar model RDA terhadap hotspot VIIRS/NOAA-20 (1-30 Sep 2026) dengan angin ERA5 100 m.

Alur: hotspot hari-1 -> kondisi awal F0; simulasi 11 hari; bandingkan Omega_f(t) dengan
hotspot kumulatif (di-dilasi 1 sel). Kalibrasi (r_eff, D, kappa) pada hari 2-6, uji pada hari 7-11.
Pembanding wajib: model "persistensi" (tanpa penyebaran, F0 saja).

Catatan jujur: mask gambut belum ada (G=1 seragam), kelembapan tanah belum ada
(r_eff = r*exp(-alpha*m) dikalibrasi sebagai satu parameter), hotspot != area terbakar.
"""
import argparse, os, sys, json, itertools, datetime as dt
import numpy as np, pandas as pd
from scipy.ndimage import binary_dilation

# Allow this script, located in the data/ directory, to import the project
# module from its parent directory when run directly or inspected by an IDE.
_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from rda_karhutla import (read_grib1, make_grid, regrid, step, cfl_check,
                          iou, dice, rmse, mae, r2)

BBOX = (-3.0, -1.0, 111.0, 116.0)  # domain validasi pada data ERA5 100 m
T0 = dt.datetime(2026, 9, 1, 0, 0)   # UTC
NDAYS = 30

def load_hotspots(path, keep=("Medium", "High")):
    d = pd.read_csv(path, sep=";", decimal=",", encoding="utf-8-sig")
    n_all = len(d)
    d = d[d["Confidence"].isin(keep)].copy()
    wib = pd.to_datetime(d["Tanggal"] + " " + d["Waktu"].str.replace(" WIB", ""), format="%d-%m-%Y %H:%M")
    d["t_utc"] = wib - pd.Timedelta(hours=7)
    d["hour"] = (d["t_utc"] - pd.Timestamp(T0)).dt.total_seconds() / 3600
    d = d[(d.hour >= 0) & (d.hour < NDAYS * 24)]
    inb = (d.Latitude.between(BBOX[0], BBOX[1])) & (d.Longitude.between(BBOX[2], BBOX[3]))
    info = dict(total=n_all, setelah_confidence_waktu=len(d), di_domain=int(inb.sum()))
    return d[inb], info

def to_mask(df, LAT, LON, dlon, buf=1):
    M = np.zeros(LAT.shape, bool)
    i = np.round((df.Latitude.values - LAT[0, 0]) / dlon).astype(int)
    j = np.round((df.Longitude.values - LON[0, 0]) / dlon).astype(int)
    ok = (i >= 0) & (i < M.shape[0]) & (j >= 0) & (j < M.shape[1])
    M[i[ok], j[ok]] = True
    return binary_dilation(M, iterations=buf) if buf else M

def simulate(g, F0, LAT, LON, dx, dy, reff, D, kappa, dt_h=1.0, Fc=0.5):
    """Euler-upwind-central; snapshot Omega_f di akhir tiap hari."""
    T, lat_s, lon_s, _ = g[246]; h0 = int((T0 - T[0]).total_seconds() // 3600)
    Dm = np.full(LAT.shape, D); rm = np.full(LAT.shape, reff); m = np.zeros(LAT.shape)
    F = F0.copy(); B = np.ones(LAT.shape); snaps = {}
    for k in range(NDAYS * 24):
        U = kappa * 3.6 * regrid(g[246][3][h0 + k], lat_s, lon_s, LAT, LON)
        V = kappa * 3.6 * regrid(g[247][3][h0 + k], lat_s, lon_s, LAT, LON)
        if k == 0:
            cd, ca = cfl_check(Dm, U, V, dx, dy, dt_h); assert cd < 1 and ca < 1
        F, B = step(F, B, U, V, Dm, rm, 0.0, m, 1.0, dx, dy, dt_h, 0.0, False)
        if (k + 1) % 24 == 0: snaps[(k + 1) // 24] = F >= Fc
    return snaps

def score(snaps, obs_cum, days, dx, dy):
    r = {}
    for k in days:
        A, B = obs_cum[k], snaps[k]
        r[k] = dict(IoU=iou(A, B), Dice=dice(A, B),
                    recall=float((A & B).sum() / max(A.sum(), 1)),
                    prec=float((A & B).sum() / max(B.sum(), 1)),
                    area_obs=float((A * dx * dy).sum()), area_mod=float((B * dx * dy).sum()))
    return r

def main():
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    ap = argparse.ArgumentParser()
    ap.add_argument("--grib", default="data_1.grib"); ap.add_argument("--hot", default="sebaran-hotspot.csv")
    ap.add_argument("--out", default="hasil_validasi"); ap.add_argument("--dlon", type=float, default=0.03)
    a = ap.parse_args(); os.makedirs(a.out, exist_ok=True)

    g = read_grib1(a.grib)
    df, info = load_hotspots(a.hot); print("hotspot:", info)
    lat, lon, LAT, LON, dx, dy = make_grid(*BBOX, a.dlon)
    # observasi kumulatif sampai akhir hari k (k=1..30)
    obs_cum = {k: to_mask(df[df.hour < 24 * k], LAT, LON, a.dlon) for k in range(1, NDAYS + 1)}
    F0 = obs_cum[1].astype(float)                       # kondisi awal = hotspot hari-1
    cal_days, test_days = list(range(2, 16)), list(range(16, 31))

    # pembanding persistensi (tanpa penyebaran)
    persist = {k: F0 >= 0.5 for k in range(1, NDAYS + 1)}
    sp = score(persist, obs_cum, cal_days + test_days, dx, dy)

    # kalibrasi grid-search pada hari 2-6 (maks. rata-rata Dice)
    grid = list(itertools.product([0.002, 0.005, 0.01, 0.02, 0.04], [0.02, 0.05, 0.1, 0.2], [0.0, 0.01, 0.03]))
    best, rows = None, []
    for reff, D, kap in grid:
        sc = score(simulate(g, F0, LAT, LON, dx, dy, reff, D, kap), obs_cum, cal_days, dx, dy)
        md = np.mean([v["Dice"] for v in sc.values()]); rows.append((reff, D, kap, md))
        if best is None or md > best[3]: best = (reff, D, kap, md)
    print("terbaik (r_eff, D, kappa, Dice_cal):", best)
    pd.DataFrame(rows, columns=["r_eff", "D", "kappa", "Dice_kalibrasi"]).to_csv(os.path.join(a.out, "kalibrasi.csv"), index=False)

    reff, D, kap, _ = best
    snaps = simulate(g, F0, LAT, LON, dx, dy, reff, D, kap)
    s_mod = score(snaps, obs_cum, cal_days + test_days, dx, dy)
    # kontrol: tanpa angin pada parameter terbaik
    s_nw = score(simulate(g, F0, LAT, LON, dx, dy, reff, D, 0.0), obs_cum, test_days, dx, dy)

    def avg(s, days, key): return float(np.mean([s[k][key] for k in days]))
    res = dict(info=info, params=dict(r_eff=reff, D=D, kappa=kap), grid_dlon=a.dlon,
               model_cal={k: avg(s_mod, cal_days, k) for k in ["IoU", "Dice", "recall", "prec"]},
               model_test={k: avg(s_mod, test_days, k) for k in ["IoU", "Dice", "recall", "prec"]},
               persist_cal={k: avg(sp, cal_days, k) for k in ["IoU", "Dice", "recall", "prec"]},
               persist_test={k: avg(sp, test_days, k) for k in ["IoU", "Dice", "recall", "prec"]},
               nowind_test={k: avg(s_nw, test_days, k) for k in ["IoU", "Dice", "recall", "prec"]})
    # temporal: luas model vs luas hotspot kumulatif pada hari uji
    ao = np.array([s_mod[k]["area_obs"] for k in test_days]); am = np.array([s_mod[k]["area_mod"] for k in test_days])
    res["temporal_test"] = dict(RMSE_km2=rmse(ao, am), MAE_km2=mae(ao, am), R2=r2(ao, am))
    # kecepatan: radius ekuivalen R=sqrt(A/pi), v = dR/dt rata-rata hari 2-11
    days_all = np.array(cal_days + test_days)
    Ro = np.sqrt(np.array([s_mod[k]["area_obs"] for k in days_all]) / np.pi)
    Rm = np.sqrt(np.array([s_mod[k]["area_mod"] for k in days_all]) / np.pi)
    res["v_fire_km_per_day"] = dict(obs=float(np.polyfit(days_all, Ro, 1)[0]), model=float(np.polyfit(days_all, Rm, 1)[0]))
    res["per_hari"] = {int(k): {kk: float(vv) for kk, vv in s_mod[k].items()} for k in cal_days + test_days}
    json.dump(res, open(os.path.join(a.out, "hasil_validasi.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "per_hari"}, indent=1))

    # ---- gambar
    ext = [LON[0, 0], LON[0, -1], LAT[0, 0], LAT[-1, 0]]
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.6), sharey=True)
    for x, k in zip(ax, [1, 15, 30]):
        x.imshow(np.where(obs_cum[k], 1, np.nan), origin="lower", extent=ext, cmap="Blues", vmin=0, vmax=1.6, alpha=.8)
        x.contour(LON, LAT, (snaps[k] if k > 1 else persist[1]).astype(float), levels=[.5], colors="red", linewidths=.8)
        x.set_title(f"Hari ke-{k}: biru=hotspot kumulatif, merah=front model"); x.set_xlabel("Bujur")
    ax[0].set_ylabel("Lintang"); plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig4_validasi_peta.png"), dpi=140); plt.close()

    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8)); ds = cal_days + test_days
    ax[0].plot(ds, [s_mod[k]["Dice"] for k in ds], "o-", label="Model RDA")
    ax[0].plot(ds, [sp[k]["Dice"] for k in ds], "s--", label="Persistensi")
    ax[0].axvline(15.5, color="gray", ls=":"); ax[0].set_xlabel("Hari"); ax[0].set_ylabel("Dice"); ax[0].legend()
    ax[1].plot(ds, [s_mod[k]["area_obs"] for k in ds], "o-", label="Hotspot kumulatif")
    ax[1].plot(ds, [s_mod[k]["area_mod"] for k in ds], "s-", label="Model")
    ax[1].axvline(15.5, color="gray", ls=":"); ax[1].set_xlabel("Hari"); ax[1].set_ylabel("Luas (km²)"); ax[1].legend()
    for x in ax: x.grid(alpha=.3)
    plt.tight_layout(); plt.savefig(os.path.join(a.out, "fig5_metrik.png"), dpi=140); plt.close()

if __name__ == "__main__":
    main()
