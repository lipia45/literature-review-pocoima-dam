#!/usr/bin/env python3
"""
Parse and plot CSMIP-style strong-motion files:
- V2: corrected accelerogram (time histories of acceleration, velocity, displacement)
- V3: response spectra (Sd, Sv, Sa, PSV at multiple damping ratios) + Fourier amplitude spectrum

Tested with: CHAN08.V2, CHAN08.V3 (Northridge, Pacoima Dam, CHAN 8)

Outputs:
- acc_time.png, vel_time.png, disp_time.png
- rs_Sa_5pct.png, rs_Sv_5pct.png, rs_Sd_5pct.png
- fourier_velocity.png (computed from V2 velocity)

Usage:
  python plot_v2_v3.py /path/to/CHAN08.V2 /path/to/CHAN08.V3

If paths are omitted, defaults to files in the current directory named *.V2 and *.V3
"""
from __future__ import annotations
import re
import sys
from pathlib import Path
from typing import Dict, Tuple
import numpy as np
import matplotlib.pyplot as plt

# -------------------------
# Parsers
# -------------------------

def parse_v2(path: Path) -> Tuple[float, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Parse a CSMIP-style V2 file (corrected accelerogram).

    Returns:
        dt (s), t (s), acc (cm/s^2), vel (cm/s), disp (cm)
    """
    txt = path.read_text(errors="ignore")
    lines = txt.splitlines()

    # Time step and number of points
    m_dt = re.search(r"INTERVALS OF\s+([.\d]+)\s+SEC", txt)
    if not m_dt:
        raise RuntimeError("Could not find sampling interval in V2 file.")
    dt = float(m_dt.group(1))

    mN = re.search(r"(\d+)\s+POINTS OF ACCEL DATA", txt)
    if not mN:
        raise RuntimeError("Could not find number of points in V2 file.")
    N = int(mN.group(1))

    def grab_series(header_regex: str, count: int) -> np.ndarray:
        # find header line index
        try:
            idx = next(i for i, l in enumerate(lines) if re.search(header_regex, l))
        except StopIteration:
            raise RuntimeError(f"Header not found: {header_regex}")
        vals: list[float] = []
        i = idx + 1
        while i < len(lines) and len(vals) < count:
            # floats with possible leading dot and scientific notation
            nums = re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:\.\d+)?", lines[i])
            vals.extend(float(x) for x in nums)
            i += 1
        arr = np.array(vals[:count], dtype=float)
        if arr.size != count:
            raise RuntimeError(f"Expected {count} values after '{header_regex}', found {arr.size}")
        return arr

    acc = grab_series(r"POINTS OF ACCEL DATA", N)  # cm/s^2
    vel = grab_series(r"POINTS OF VELOC DATA", N)  # cm/s
    disp = grab_series(r"POINTS OF DISPL DATA", N)  # cm

    t = np.arange(N, dtype=float) * dt
    return dt, t, acc, vel, disp


def parse_v3(path: Path) -> Tuple[np.ndarray, Dict[float, Dict[str, np.ndarray]], np.ndarray]:
    """Parse a CSMIP-style V3 file (response + Fourier amplitude spectra).

    Returns:
        T (s) periods array
        spectra: dict keyed by damping ratio (e.g., 0.05) -> dict with keys 'Sd','Sv','Sa','PSV'
        fourier_amp: np.ndarray of Fourier amplitude values (units per file header)
    """
    txt = path.read_text(errors="ignore")

    # Period list (starts at first line beginning with ".040" and runs until the Fourier header)
    m_start = re.search(r"\n\s*\.040\s", txt)
    if not m_start:
        raise RuntimeError("Could not locate the periods list start (.040) in V3 file.")
    i0 = m_start.start()
    i1 = txt.find("FOURIER AMPLITUDE SPECTRA", i0)
    if i1 < 0:
        raise RuntimeError("Could not locate the 'FOURIER AMPLITUDE SPECTRA' header in V3 file.")
    period_block = txt[i0:i1]
    T_vals = [float(x) for x in re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:\.\d+)?", period_block)]
    # Keep positive values, de-duplicate while preserving order
    seen = set()
    T: list[float] = []
    for p in T_vals:
        if p > 0 and p not in seen:
            T.append(p)
            seen.add(p)
    T = np.array(T, dtype=float)

    # Damping blocks
    dampings = [0.00, 0.02, 0.05, 0.10, 0.20]
    spectra: Dict[float, Dict[str, np.ndarray]] = {}
    for d in dampings:
        hdr = f"DAMPING =  .{int(d*100):02d}. DATA OF SD,SV,SA,PSSV,TTSD,TTSV,TTSA"
        idx = txt.find(hdr)
        if idx < 0:
            continue
        sub = txt[idx + len(hdr):]
        # end at next damping header or EOF
        m_next = re.search(r"\n\s*DAMPING\s*=", sub)
        sub = sub[: m_next.start()] if m_next else sub
        nums = [float(x.replace('E', 'e')) for x in re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:\.\d+)?", sub)]
        if len(nums) % 7 != 0:
            # still proceed by truncation
            pass
        chunk_len = len(nums) // 7
        chunks = [nums[i*chunk_len:(i+1)*chunk_len] for i in range(7)]
        nT = len(T)
        Sd = np.array(chunks[0][:nT], dtype=float)
        Sv = np.array(chunks[1][:nT], dtype=float)
        Sa = np.array(chunks[2][:nT], dtype=float)
        PSV = np.array(chunks[3][:nT], dtype=float)
        spectra[d] = {"Sd": Sd, "Sv": Sv, "Sa": Sa, "PSV": PSV}

    # Fourier amplitude values between the Fourier header and the first damping block
    j0 = txt.find("FOURIER AMPLITUDE SPECTRA IN")
    sub = txt[j0:]
    m_damp = re.search(r"\n\s*DAMPING\s*=", sub)
    # take lines after the header up to (not including) the damping header
    lines = sub.splitlines()
    # skip the header line itself
    upto = sub[: m_damp.start()].count("\n") if m_damp else len(lines)
    fou_text = "\n".join(lines[1:upto])
    fourier_amp = np.array([float(x.replace('E', 'e')) for x in re.findall(r"[-+]?\d*\.\d+(?:[Ee][-+]?\d+)?|[-+]?\d+(?:\.\d+)?", fou_text)], dtype=float)

    return T, spectra, fourier_amp


# -------------------------
# Plot helpers (one chart per figure as per common notebook guidelines)
# -------------------------

def plot_time_series(t: np.ndarray, y: np.ndarray, ylabel: str, title: str, out_png: Path) -> None:
    plt.figure()
    plt.plot(t, y)
    plt.xlabel("Time (s)")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


def plot_response_spectrum(T: np.ndarray, Y: np.ndarray, ylab: str, title: str, out_png: Path) -> None:
    plt.figure()
    plt.plot(T, Y)
    plt.xlabel("Period T (s)")
    plt.ylabel(ylab)
    plt.title(title)
    plt.xscale("log")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


def plot_fourier_velocity_from_v2(vel: np.ndarray, dt: float, out_png: Path, fmax: float | None = None) -> None:
    """Compute and plot single-sided Fourier amplitude spectrum of velocity (|FFT| scaled).
    Units: if vel is in cm/s, amplitude will be in cm.
    """
    n = len(vel)
    # Real FFT
    V = np.fft.rfft(vel)
    f = np.fft.rfftfreq(n, d=dt)
    # Amplitude spectrum (single-sided), 2/N scaling except DC & Nyquist
    amp = np.abs(V) / n * 2.0
    if n % 2 == 0:
        amp[-1] /= 2.0  # correct Nyquist if even length
    amp[0] /= 2.0      # correct DC

    if fmax is None:
        fmax = f.max()
    mask = f <= fmax

    plt.figure()
    plt.plot(f[mask], amp[mask])
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Fourier amplitude of velocity (cm)")
    plt.title("Fourier Amplitude Spectrum (from V2 velocity)")
    plt.grid(True, which="both", linestyle=":", alpha=0.6)
    plt.tight_layout()
    plt.savefig(out_png, dpi=150)
    print(f"Saved: {out_png}")


# -------------------------
# Main
# -------------------------

def main(v2_path: str | None = None, v3_path: str | None = None) -> None:
    # Resolve input files
    if v2_path is None:
        v2_candidates = list(Path.cwd().glob("*.V2"))
        if not v2_candidates:
            raise SystemExit("No .V2 file found in current directory; pass a path explicitly.")
        v2 = v2_candidates[0]
    else:
        v2 = Path(v2_path)
    if v3_path is None:
        v3_candidates = list(Path.cwd().glob("*.V3"))
        if not v3_candidates:
            raise SystemExit("No .V3 file found in current directory; pass a path explicitly.")
        v3 = v3_candidates[0]
    else:
        v3 = Path(v3_path)

    print(f"Reading V2: {v2}")
    dt, t, acc, vel, disp = parse_v2(v2)
    print(f"  dt = {dt} s, N = {len(t)}")

    print(f"Reading V3: {v3}")
    T, spectra, fourier_amp = parse_v3(v3)
    print(f"  periods parsed: {len(T)}; damping keys: {sorted(spectra.keys())}")
    print(f"  Fourier amplitude points (from file): {len(fourier_amp)}")

    outdir = Path.cwd()

    # Time-series plots
    plot_time_series(t, acc, "Acceleration (cm/s²)", "Acceleration vs Time", outdir / "acc_time.png")
    plot_time_series(t, vel, "Velocity (cm/s)", "Velocity vs Time", outdir / "vel_time.png")
    plot_time_series(t, disp, "Displacement (cm)", "Displacement vs Time", outdir / "disp_time.png")

    # Response spectra: plot 5% damping by default
    if 0.05 in spectra:
        Sa5 = spectra[0.05]["Sa"]
        Sv5 = spectra[0.05]["Sv"]
        Sd5 = spectra[0.05]["Sd"]
        plot_response_spectrum(T, Sa5, "Sa (cm/s²)", "Response Spectrum Sa (5% damping)", outdir / "rs_Sa_5pct.png")
        plot_response_spectrum(T, Sv5, "Sv (cm/s)", "Response Spectrum Sv (5% damping)", outdir / "rs_Sv_5pct.png")
        plot_response_spectrum(T, Sd5, "Sd (cm)", "Response Spectrum Sd (5% damping)", outdir / "rs_Sd_5pct.png")
    else:
        print("Warning: 5% damping block not found in V3; skipping response spectrum plots.")

    # Fourier amplitude: compute from V2 velocity for a clean f-axis
    plot_fourier_velocity_from_v2(vel, dt, outdir / "fourier_velocity.png", fmax=25.0)

    print("Done.")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        main(sys.argv[1], sys.argv[2])
    else:
        # Try defaults in CWD
        main()
