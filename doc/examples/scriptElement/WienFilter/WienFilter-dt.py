#!/usr/bin/env python3
"""
WienFilter-dt.py

Propagate an Elegant-style particle distribution through an idealized, uniform-field
Wien filter (crossed E and B), ignoring spin.

Particles are read from an SDDS file with columns:
  x, xp, y, yp, t, p

Conventions used here (per your correction):
  x, y [m]
  xp, yp = dx/dz, dy/dz (dimensionless slopes)
  t [s] (lab time coordinate; advanced by dt)
  p = beta*gamma  (dimensionless) = |P| / (m c)

Fields:
  Lorentz-force cancellation for a chosen reference speed v0 is |E| = v0 * |B|
  (with E oriented to cancel v×B for the reference direction).

Input/output SDDS are written using Soliday's SDDS Python bindings (pip).

Mostly written by ChatGPT 5.2.  May contain errors.
"""

import argparse
import math
import os
import sys
from dataclasses import dataclass

import numpy as np

import sdds

# Physical constants (SI)
C = 299_792_458.0
E_CHARGE = 1.602176634e-19
M_E = 9.1093837015e-31


@dataclass
class Fields:
    E: np.ndarray  # [V/m]
    B: np.ndarray  # [T]


def parse_vec3(s: str) -> np.ndarray:
    parts = [p.strip() for p in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"Expected 3 comma-separated components, got: {s}")
    return np.array([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)


def unit(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n == 0:
        raise ValueError("Zero vector cannot be normalized")
    return v / n


def ke_to_bg(ke_eV: float) -> float:
    """Convert kinetic energy (eV) to beta*gamma for an electron."""
    mc2_eV = 510_998.950  # electron rest energy in eV
    gamma = 1.0 + ke_eV / mc2_eV
    beta2 = 1.0 - 1.0 / (gamma * gamma)
    beta = math.sqrt(max(beta2, 0.0))
    return beta * gamma


def bg_to_beta_gamma(bg: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Given beta*gamma, return (beta, gamma)."""
    gamma = np.sqrt(1.0 + bg * bg)
    beta = bg / gamma
    return beta, gamma


def slopes_to_direction(xp: np.ndarray, yp: np.ndarray) -> np.ndarray:
    """
    Convert Elegant slopes (x', y') to a unit direction vector (x,y,z),
    assuming z is the reference longitudinal axis.
    """
    denom = np.sqrt(1.0 + xp * xp + yp * yp)
    return np.stack((xp / denom, yp / denom, 1.0 / denom), axis=1)


def boris_push(p_si: np.ndarray,
               q: float,
               m: float,
               dt: np.ndarray,
               fields: Fields) -> np.ndarray:
    """
    Relativistic Boris pusher with per-particle time steps.

    Parameters
    ----------
    p_si : (N,3) ndarray
        Particle momenta in SI units [kg m / s]
    q : float
        Particle charge [C]
    m : float
        Particle mass [kg]
    dt : (N,) ndarray
        Per-particle time step [s]
    fields : Fields
        Uniform electric and magnetic fields

    Returns
    -------
    p_new : (N,3) ndarray
        Updated particle momenta
    """
    E = fields.E            # (3,)
    B = fields.B            # (3,)
    dt = dt[:, None]        # (N,1) for broadcasting

    # Half electric kick
    p_minus = p_si + q * E * (0.5 * dt)

    # gamma from p_minus
    p2 = np.sum(p_minus * p_minus, axis=1)
    gamma_minus = np.sqrt(1.0 + p2 / ((m * C) ** 2))
    gamma_minus = gamma_minus[:, None]  # (N,1)

    # Rotation vectors
    t = (q * B) * (0.5 * dt) / (m * gamma_minus)
    t2 = np.sum(t * t, axis=1)[:, None]
    s = 2.0 * t / (1.0 + t2)

    # Magnetic rotation
    p_prime = p_minus + np.cross(p_minus, t)
    p_plus = p_minus + np.cross(p_prime, s)

    # Half electric kick
    p_new = p_plus + q * E * (0.5 * dt)

    return p_new


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Propagate an Elegant SDDS particle distribution through a uniform Wien filter."
    )
    ap.add_argument("input_sdds", help="Input SDDS file (Elegant particle distribution).")
    ap.add_argument("output_sdds", help="Output SDDS file (updated distribution).")

    ap.add_argument("--length", type=float, required=True, help="Wien filter length [m].")
    ap.add_argument("--steps", type=int, default=2000, help="Integration steps (default: 2000).")

    ap.add_argument("--B", type=float, default=1.0, help="Magnetic field magnitude [T] (default: 1.0).")
    ap.add_argument("--Bdir", type=str, default="0,1,0", help="B direction as 'x,y,z' (default: 0,1,0).")

    ap.add_argument("--E", type=float, default=None, help="Electric field magnitude [V/m] (default: 0.0.")
    ap.add_argument("--Edir", type=str, default="1,0,0", help="E direction as 'x,y,z' (default: 1,0,0).")

    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if it exists.")
    args = ap.parse_args()

    if (not args.overwrite) and os.path.exists(args.output_sdds):
        print(f"ERROR: output file exists: {args.output_sdds} (use --overwrite to replace)", file=sys.stderr)
        return 2

    if args.E is None:
        raise SystemExit("Give --E option.")
    
    if args.steps < 10:
        print("ERROR: --steps must be >= 10", file=sys.stderr)
        return 2

    # Load SDDS
    sddsInput = sdds.load(args.input_sdds)
    
    def getColumn(name: str) -> np.ndarray:
        if name not in sddsInput.columnName:
            raise SystemExit(f"Missing required SDDS column: {name}")
        i = sddsInput.columnName.index(name)
        if name == "particleID":
            return sddsInput.columnData[i][0]
        return np.array(sddsInput.columnData[i][0], dtype=float)
    def getParameter(name: str):
        if name not in sddsInput.parameterName:
            raise SystemExit(f"Missing required SDDS parameter: {name}")
        i = sddsInput.parameterName.index(name)
        return sddsInput.parameterData[i][0]
    
    x = getColumn("x")
    xp = getColumn("xp")
    y = getColumn("y")
    yp = getColumn("yp")
    t = getColumn("t")
    bg = getColumn("p")  # beta*gamma
    pid = getColumn("particleID")
    charge = getParameter("Charge")
    slotsPerBunch = getParameter("IDSlotsPerBunch")
    
    del sddsInput
    
    n = len(x)
    if not (len(xp) == len(y) == len(yp) == len(t) == len(bg) == n):
        raise SystemExit("Column lengths do not match.")

    # Initial direction from slopes
    nhat = slopes_to_direction(xp, yp)  # (N,3)

    # Initial SI momentum magnitude: |p| = (beta*gamma) * m c
    p_si = (bg[:, None] * (M_E * C)) * nhat

    # Uniform fields
    Bdir = unit(parse_vec3(args.Bdir))
    Edir = unit(parse_vec3(args.Edir))
    Bvec = args.B * Bdir

    Emag = float(args.E)
    Evec = Emag * Edir
    fields = Fields(E=Evec.astype(float), B=Bvec.astype(float))

    # Time step: set by reference speed v0 (computed above if needed),
    # otherwise approximate v0 from median bg when E is specified.
    if args.E is not None:
        bg_ref = float(np.median(bg))
        beta_ref, _ = bg_to_beta_gamma(np.array([bg_ref], dtype=float))
        v0 = float(beta_ref[0] * C)

    q = -E_CHARGE
    m = M_E

    # Track z internally; Elegant coordinates don't store it in this table.
    z = np.zeros(n, dtype=float)
    dz = args.length/args.steps
    
    for step in range(args.steps):
        # gamma from SI momentum
        p2 = np.sum(p_si * p_si, axis=1)
        gamma = np.sqrt(1.0 + p2 / ((m * C) ** 2))
        v = p_si / (m * gamma[:, None])  # m/s
        vz = v[:,2]
        dt = dz/vz
        
        x += v[:, 0] * dt
        y += v[:, 1] * dt
        z += v[:, 2] * dt
        t += dt

        p_si = boris_push(p_si, q=q, m=m, dt=dt, fields=fields)

    # Convert back to Elegant columns
    p2 = np.sum(p_si * p_si, axis=1)
    bg_new = np.sqrt(p2) / (m * C)  # beta*gamma

    gamma = np.sqrt(1.0 + p2 / ((m * C) ** 2))
    v = p_si / (m * gamma[:, None])
    vz = v[:, 2]
    eps = 1e-30
    vz_safe = np.where(np.abs(vz) < eps, np.sign(vz) * eps + eps, vz)

    xp_new = v[:, 0] / vz_safe
    yp_new = v[:, 1] / vz_safe

    sddsOutput = sdds.SDDS()
    sddsOutput.mode = sdds.SDDS_BINARY
    #sddsOutput.setDescription("elegant coordinates", "Output from WienFilter.py")
    sddsOutput.defineColumn(name="x", units="m", type=sdds.SDDS_DOUBLE, fieldLength=0)
    sddsOutput.defineColumn(name="xp", units="", type=sdds.SDDS_DOUBLE, fieldLength=0)
    sddsOutput.defineColumn(name="y", units="m", type=sdds.SDDS_DOUBLE, fieldLength=0)
    sddsOutput.defineColumn(name="yp", units="", type=sdds.SDDS_DOUBLE, fieldLength=0)
    sddsOutput.defineColumn(name="t", units="s", type=sdds.SDDS_DOUBLE, fieldLength=0)
    sddsOutput.defineColumn(name="p", units="m$be$nc", type=sdds.SDDS_DOUBLE, fieldLength=0)
    sddsOutput.defineColumn(name="particleID", units="", type=sdds.SDDS_ULONG64, fieldLength=0)
    sddsOutput.defineParameter(name="Charge", units="C", type=sdds.SDDS_DOUBLE)
    sddsOutput.defineParameter(name="IDSlotsPerBunch", units="", type=sdds.SDDS_LONG)

    sddsOutput.setColumnValueList(name="x", valueList=x.tolist(), page=1)
    sddsOutput.setColumnValueList(name="xp", valueList=xp_new.tolist(), page=1)
    sddsOutput.setColumnValueList(name="y", valueList=y.tolist(), page=1)
    sddsOutput.setColumnValueList(name="yp", valueList=yp_new.tolist(), page=1)
    sddsOutput.setColumnValueList(name="t", valueList=t.tolist(), page=1)
    sddsOutput.setColumnValueList(name="p", valueList=bg_new.tolist(), page=1)
    sddsOutput.setColumnValueList(name="particleID", valueList=pid, page=1)
    sddsOutput.setParameterValue(name="Charge", value=charge, page=1)
    sddsOutput.setParameterValue(name="IDSlotsPerBunch", value=slotsPerBunch, page=1)
    
    sddsOutput.save(args.output_sdds)
    del sddsOutput
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
