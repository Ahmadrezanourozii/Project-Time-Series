"""Clinically motivated gait features for one VGRF recording.

The amplitude-statistic tokens used by the Mamba model describe each second of
signal; they do not describe *gait cycles*. This module segments each foot's
total force into stance phases and measures what the PD gait literature
reports as discriminative:

  timing        stride/stance/swing time, cadence, double support, and their
                stride-to-stride variability (robust CV) and serial correlation
  force shape   the "M" profile of stance: heel-strike peak, mid-stance valley,
                push-off peak, loading/unloading rates -- PD flattens it
  pressure path centre-of-pressure (COP) from the 8 sensor coordinates in
                format.txt: heel-to-toe roll-over excursion and speed, medio-
                lateral sway, heel vs forefoot loading at initial contact
  bilateral     left/right asymmetry, gait asymmetry (Plotnik 2005) and the
                phase coordination index (Plotnik 2007), both impaired in PD

Every quantity is computed from the recording itself. Forces are divided by a
body-weight proxy -- the mean of left+right total force, which over whole gait
cycles equals body weight -- so insole calibration and subject mass cancel.
No statistic is shared across recordings, which keeps inference on a new
patient independent of any cohort.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FS_HZ = 100.0
SKIP_INIT_SEC = 5.0
MIN_STRIDE_SEC, MAX_STRIDE_SEC = 0.6, 2.5
MIN_STANCE_SEC = 0.2

# Sensor positions from Data/format.txt (Y < 0 is the heel, Y > 0 the toes).
SENSOR_X = np.array([-500, -700, -300, -700, -300, -700, -300, -500], dtype=float)
SENSOR_Y = np.array([-800, -400, -400, 0, 0, 400, 400, 800], dtype=float)
HEEL = np.array([0, 1, 2])
FOREFOOT = np.array([5, 6, 7])

COLUMNS = ["time"] + [f"L{i}" for i in range(1, 9)] + [f"R{i}" for i in range(1, 9)] + ["TotalL", "TotalR"]


def load_recording(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="c")
    return df.loc[df["time"] > SKIP_INIT_SEC].reset_index(drop=True)


def stance_intervals(force: np.ndarray, threshold: float) -> list[tuple[int, int]]:
    """Contiguous loaded stretches [start, end) longer than MIN_STANCE_SEC."""
    loaded = force > threshold
    edges = np.diff(loaded.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    # Partial stances at the borders are dropped: their timing is unknown.
    if loaded[0] and ends:
        ends = ends[1:]
    if loaded[-1] and starts:
        starts = starts[:-1]
    return [(s, e) for s, e in zip(starts, ends) if (e - s) >= MIN_STANCE_SEC * FS_HZ]


def _robust_cv(x: np.ndarray) -> float:
    med = np.median(x)
    return float(1.4826 * np.median(np.abs(x - med)) / med) if len(x) > 2 and med > 0 else np.nan


def _sd_cv(x: np.ndarray) -> float:
    """Classical CV after discarding values outside median +/- 3 MAD (turns, missteps)."""
    if len(x) < 3:
        return np.nan
    med = np.median(x)
    mad = 1.4826 * np.median(np.abs(x - med)) + 1e-9
    keep = x[np.abs(x - med) <= 3 * mad]
    return float(np.std(keep, ddof=1) / np.mean(keep)) if len(keep) > 2 else np.nan


def _lag1(x: np.ndarray) -> float:
    if len(x) < 5:
        return np.nan
    x = x - x.mean()
    denom = np.sum(x * x)
    return float(np.sum(x[1:] * x[:-1]) / denom) if denom > 0 else np.nan


def _stance_shape(f: np.ndarray) -> dict[str, float]:
    """Heel-strike peak, valley, push-off peak of one stance (force in BW)."""
    grid = np.linspace(0, len(f) - 1, 101)
    g = np.interp(grid, np.arange(len(f)), f)
    i1 = int(np.argmax(g[:50]))
    i2 = 50 + int(np.argmax(g[50:]))
    iv = i1 + int(np.argmin(g[i1 : i2 + 1]))
    f1, f2, fv = g[i1], g[i2], g[iv]
    dur = len(f) / FS_HZ
    return {
        "f1": f1,
        "f2": f2,
        "fvalley": fv,
        "peak_ratio": f1 / f2 if f2 > 0 else np.nan,
        "valley_depth": fv / max(0.5 * (f1 + f2), 1e-9),
        "t_f1": i1 / 100.0,
        "t_f2": i2 / 100.0,
        "load_rate": f1 / max(i1 / 100.0 * dur, 1.0 / FS_HZ),
        "unload_rate": f2 / max((100 - i2) / 100.0 * dur, 1.0 / FS_HZ),
        "impulse": float(np.sum(f) / FS_HZ),
    }


def _stance_cop(sensors: np.ndarray, x_sign: float) -> dict[str, float]:
    """COP path of one stance; sensors is (n, 8) raw force."""
    total = sensors.sum(axis=1)
    ok = total > 1e-6
    w = sensors[ok] / total[ok, None]
    cop_y = w @ SENSOR_Y
    cop_x = x_sign * (w @ SENSOR_X)  # medial is negative for both feet
    n = len(cop_y)
    k = max(1, n // 10)
    excursion = float(np.percentile(cop_y, 95) - np.percentile(cop_y, 5))
    impulse = sensors.sum(axis=0)
    share = impulse / max(impulse.sum(), 1e-9)
    first = sensors[: max(1, len(sensors) // 10)].sum(axis=0)
    return {
        "cop_y_start": float(cop_y[:k].mean()),
        "cop_y_end": float(cop_y[-k:].mean()),
        "cop_ap_excursion": excursion,
        "cop_ap_speed": excursion / (len(sensors) / FS_HZ),
        "cop_ml_range": float(np.percentile(cop_x, 95) - np.percentile(cop_x, 5)),
        "cop_ml_mean": float(cop_x.mean()),
        "heel_share": float(share[HEEL].sum()),
        "forefoot_share": float(share[FOREFOOT].sum()),
        "heel_share_contact": float(first[HEEL].sum() / max(first.sum(), 1e-9)),
        **{f"sensor{i + 1}_share": float(share[i]) for i in range(8)},
    }


def foot_strides(df: pd.DataFrame, side: str, bw: float) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """One row per valid stride of one foot (stance i -> next contact)."""
    force = df[f"Total{side}"].to_numpy(dtype=float)
    sensors = df[[f"{side}{i}" for i in range(1, 9)]].to_numpy(dtype=float)
    threshold = max(0.02 * bw, 0.05 * np.percentile(force, 99))
    stances = stance_intervals(force, threshold)
    x_sign = -1.0 if side == "L" else 1.0  # mirror so +x is lateral on both feet

    rows = []
    for (s0, e0), (s1, _) in zip(stances[:-1], stances[1:]):
        stride = (s1 - s0) / FS_HZ
        if not (MIN_STRIDE_SEC <= stride <= MAX_STRIDE_SEC):
            continue
        stance = (e0 - s0) / FS_HZ
        row = {
            "start": s0,
            "end": e0,
            "next": s1,
            "stride_time": stride,
            "stance_time": stance,
            "swing_time": (s1 - e0) / FS_HZ,
            "stance_pct": stance / stride,
        }
        row.update(_stance_shape(force[s0:e0] / bw))
        row.update(_stance_cop(sensors[s0:e0], x_sign))
        rows.append(row)
    return pd.DataFrame(rows), stances


PER_STRIDE_MEDIAN = [
    "stride_time", "stance_time", "swing_time", "stance_pct", "f1", "f2", "fvalley", "peak_ratio",
    "valley_depth", "t_f1", "t_f2", "load_rate", "unload_rate", "impulse", "cop_y_start", "cop_y_end",
    "cop_ap_excursion", "cop_ap_speed", "cop_ml_range", "cop_ml_mean", "heel_share", "forefoot_share",
    "heel_share_contact",
] + [f"sensor{i}_share" for i in range(1, 9)]
PER_STRIDE_VARIABILITY = [
    "stride_time", "stance_time", "swing_time", "stance_pct", "f1", "f2", "impulse", "cop_ap_excursion",
]
# Positions (sign-changing coordinates) get a spread, not a coefficient of variation.
PER_STRIDE_SPREAD = ["cop_y_start", "cop_y_end", "cop_ml_mean"]


def _double_support(own: pd.DataFrame, other: list[tuple[int, int]]) -> np.ndarray:
    """Per stride: fraction of the stride both feet are loaded."""
    out = []
    for s0, e0, s1 in own[["start", "end", "next"]].to_numpy():
        overlap = sum(max(0, min(e0, oe) - max(s0, os_)) for os_, oe in other)
        out.append(overlap / (s1 - s0))
    return np.asarray(out, dtype=float)


def _phase_coordination(left: pd.DataFrame, right: pd.DataFrame) -> tuple[float, float, float]:
    """(PCI, phase CV %, mean |phase-180| deg) -- Plotnik et al. 2007.

    Phase of each right heel strike within the enclosing left stride, in degrees;
    perfectly anti-phased walking gives 180.
    """
    r_starts = right["start"].to_numpy()
    phases = []
    for s0, s1 in left[["start", "next"]].to_numpy():
        inside = r_starts[(r_starts > s0) & (r_starts < s1)]
        if len(inside) == 1:
            phases.append(360.0 * (inside[0] - s0) / (s1 - s0))
    phases = np.asarray(phases)
    if len(phases) < 5:
        return np.nan, np.nan, np.nan
    cv = 100.0 * np.std(phases, ddof=1) / np.mean(phases)
    abs_dev = np.mean(np.abs(phases - 180.0))
    return float(cv + 100.0 * abs_dev / 180.0), float(cv), float(abs_dev)


def _regularity(signal: np.ndarray, stride_samples: int) -> float:
    """Autocorrelation of the signal at the median stride lag (gait regularity)."""
    x = signal - signal.mean()
    if stride_samples <= 0 or stride_samples >= len(x) - 10:
        return np.nan
    a, b = x[:-stride_samples], x[stride_samples:]
    denom = np.sqrt(np.sum(a * a) * np.sum(b * b))
    return float(np.sum(a * b) / denom) if denom > 0 else np.nan


def recording_features(df: pd.DataFrame) -> dict[str, float]:
    """All features of one recording (after the initial 5 s are dropped)."""
    left_force = df["TotalL"].to_numpy(dtype=float)
    right_force = df["TotalR"].to_numpy(dtype=float)
    bw = float(np.mean(left_force + right_force))
    duration = len(df) / FS_HZ

    feats: dict[str, float] = {"duration_s": duration}
    strides = {}
    stances = {}
    for side in ("L", "R"):
        st, stance_list = foot_strides(df, side, bw)
        strides[side], stances[side] = st, stance_list
        force = left_force if side == "L" else right_force
        thr = max(0.02 * bw, 0.05 * np.percentile(force, 99))
        feats[f"{side}_loaded_fraction"] = float(np.mean(force > thr))
        feats[f"{side}_n_strides"] = float(len(st))
        if len(st) < 5:
            continue
        for col in PER_STRIDE_MEDIAN:
            feats[f"{side}_{col}"] = float(np.median(st[col]))
        for col in PER_STRIDE_VARIABILITY:
            feats[f"{side}_{col}_cv"] = _sd_cv(st[col].to_numpy())
            feats[f"{side}_{col}_rcv"] = _robust_cv(st[col].to_numpy())
        for col in PER_STRIDE_SPREAD:
            x = st[col].to_numpy()
            feats[f"{side}_{col}_sd"] = float(1.4826 * np.median(np.abs(x - np.median(x))))
        feats[f"{side}_stride_time_lag1"] = _lag1(st["stride_time"].to_numpy())
        feats[f"{side}_regularity"] = _regularity(force, int(round(np.median(st["stride_time"]) * FS_HZ)))

    left, right = strides["L"], strides["R"]
    if len(left) >= 5 and len(right) >= 5:
        for side, own, other in (("L", left, stances["R"]), ("R", right, stances["L"])):
            ds = _double_support(own, other)
            feats[f"{side}_double_support"] = float(np.median(ds))
            feats[f"{side}_double_support_cv"] = _sd_cv(ds + 1e-6)
        stride = 0.5 * (np.median(left["stride_time"]) + np.median(right["stride_time"]))
        feats["cadence"] = 120.0 / stride
        for col in ("stride_time", "stance_time", "swing_time", "stance_pct", "f1", "f2", "impulse",
                    "cop_ap_excursion", "heel_share"):
            a, b = np.median(left[col]), np.median(right[col])
            feats[f"asym_{col}"] = float(abs(a - b) / max(0.5 * abs(a + b), 1e-9))
        sw_l, sw_r = np.median(left["swing_time"]), np.median(right["swing_time"])
        feats["gait_asymmetry"] = float(100.0 * abs(np.log(sw_r / sw_l)))
        feats["pci"], feats["phase_cv"], feats["phase_abs_dev"] = _phase_coordination(left, right)
        feats["regularity_total"] = _regularity(
            left_force + right_force, int(round(stride * FS_HZ / 2))
        )
    feats["bw_proxy_N"] = bw  # kept for diagnostics, excluded from models
    return feats


def extract_directory(paths: list[Path]) -> pd.DataFrame:
    rows = []
    for path in paths:
        feats = recording_features(load_recording(path))
        feats["recording"] = path.name
        rows.append(feats)
    return pd.DataFrame(rows).set_index("recording")
