#!/usr/bin/env python3
"""
gro_utils.py

Utilities to parse a .gro text and to write a .gro file from a PyTorch tensor,
matching fixed-width formatting and supporting velocities.

Functions:
- parse_gro_text(gro_text) -> dict
- write_gro_from_torch(coords, output_path, ..., velocities=None, atom_names=None, residue_names=None, residue_numbers=None, input_in_angstrom=True, box=None)

Example usage:
    # Parse existing gro text
    with open("example.gro") as f:
        text = f.read()
    parsed = parse_gro_text(text)

    # Write from torch tensor (coords in angstroms by default)
    write_gro_from_torch(coords_tensor, "out.gro", atom_names=parsed["atom_names"],
                         residue_names=parsed["residue_names"],
                         residue_numbers=parsed["residue_numbers"],
                         velocities=torch.tensor(parsed["velocities"]) if parsed["velocities"] is not None else None,
                         input_in_angstrom=True, box=parsed["box"])
"""

from typing import Optional, Sequence, Tuple, List, Dict, Any
import numpy as np
import torch
import re


def parse_gro_text(gro_text: str) -> Dict[str, Any]:
    """
    Parse a .gro file text into components.

    Returns dict with keys:
    - title: str
    - n_atoms: int
    - atoms: list of dicts with keys: residue_number, residue_name, atom_name, atom_number, x, y, z, (optional) vx, vy, vz
    - atom_names: list of atom names (strings)
    - residue_names: list of residue names (strings)
    - residue_numbers: list of ints
    - coords: numpy array shape (n,3) in nm
    - velocities: list of (vx,vy,vz) or None
    - box: tuple of 3 floats (nm) or the raw box line string if non-orthogonal
    """
    lines = [ln.rstrip() for ln in gro_text.splitlines() if ln.strip() != "" or ln == ""]
    if len(lines) < 3:
        raise ValueError("Input gro text is too short")

    title = lines[0]
    n_atoms = int(lines[1].strip())

    atom_lines = lines[2:2 + n_atoms]
    box_line = lines[2 + n_atoms] if len(lines) > 2 + n_atoms else ""

    atoms = []
    coords = np.zeros((n_atoms, 3), dtype=float)
    velocities: List[Optional[Tuple[float, float, float]]] = []

    # Strict gro format uses fixed columns, but some files have variable whitespace.
    # We'll try to parse robustly:
    for i, ln in enumerate(atom_lines):
        # First try strict fixed-width parsing similar to GROMACS:
        # cols: 0:5 resnr, 5:10 resname, 10:15 atomname, 15:20 atomnr, 20:28 x, 28:36 y, 36:44 z, optionally 44:52 vx, 52:60 vy, 60:68 vz
        try:
            resno = int(ln[0:5].strip())
            resname = ln[5:10].strip()
            atomname = ln[10:15].strip()
            atomno = int(ln[15:20].strip())
            x = float(ln[20:28].strip())
            y = float(ln[28:36].strip())
            z = float(ln[36:44].strip())
            vx = vy = vz = None
            if len(ln) >= 52:
                # velocities may be present
                try:
                    vx = float(ln[44:52].strip())
                    vy = float(ln[52:60].strip())
                    vz = float(ln[60:68].strip())
                except Exception:
                    vx = vy = vz = None
        except Exception:
            # Fallback: split by whitespace (some gro files are space separated)
            parts = re.split(r"\s+", ln.strip())
            if len(parts) < 7:
                raise ValueError(f"Cannot parse atom line ({i+1}): {ln!r}")
            # Typical split order: resname atomname atomno x y z [vx vy vz] but your sample includes resno first
            # We'll attempt to locate numeric tokens for coords
            # Find the first token that can be integer (resno)
            try:
                resno = int(parts[0])
                resname = parts[1]
                atomname = parts[2]
                atomno = int(parts[3])
                x = float(parts[4])
                y = float(parts[5])
                z = float(parts[6])
                if len(parts) >= 10:
                    vx = float(parts[7]); vy = float(parts[8]); vz = float(parts[9])
                else:
                    vx = vy = vz = None
            except Exception as e:
                raise ValueError(f"Unable to parse atom line by fallback: {ln!r}") from e

        coords[i, 0] = x
        coords[i, 1] = y
        coords[i, 2] = z
        velocities.append((vx, vy, vz) if vx is not None else None)

        atoms.append({
            "residue_number": resno,
            "residue_name": resname,
            "atom_name": atomname,
            "atom_number": atomno,
            "x": x,
            "y": y,
            "z": z,
            "vx": vx,
            "vy": vy,
            "vz": vz
        })

    # Parse box: often three floats; could be more complex for triclinic boxes
    box_vals = None
    box_raw = box_line.strip()
    if box_raw != "":
        parts = re.split(r"\s+", box_raw)
        try:
            if len(parts) >= 3:
                box_vals = tuple(float(parts[i]) for i in range(3))
            else:
                box_vals = tuple(float(p) for p in parts)
        except Exception:
            box_vals = box_raw

    return {
        "title": title,
        "n_atoms": n_atoms,
        "atoms": atoms,
        "atom_names": [a["atom_name"] for a in atoms],
        "residue_names": [a["residue_name"] for a in atoms],
        "residue_numbers": [a["residue_number"] for a in atoms],
        "coords": coords,            # in nm
        "velocities": velocities,    # list of tuples or None
        "box": box_vals or box_raw
    }


def _format_gro_atom_line_fixed(residue_number: int,
                                residue_name: str,
                                atom_name: str,
                                atom_number: int,
                                x_nm: float,
                                y_nm: float,
                                z_nm: float,
                                vx: Optional[float] = None,
                                vy: Optional[float] = None,
                                vz: Optional[float] = None) -> str:
    """
    Strict fixed-width formatting for .gro atom lines.

    Fields and column widths (GROMACS convention):
    1-5    residue number   %5d
    6-10   residue name     %5s
    11-15  atom name        %5s
    16-20  atom number      %5d
    21-28  x (nm)           %8.3f
    29-36  y (nm)           %8.3f
    37-44  z (nm)           %8.3f
    45-52  vx (nm/ps)       %8.4f  (optional)
    53-60  vy (nm/ps)       %8.4f
    61-68  vz (nm/ps)       %8.4f
    """
    # Note: Python formatting will naturally pad/truncate strings.
    base = f"{residue_number:5d}{residue_name:>5s}{atom_name:>5s}{atom_number:5d}" \
           f"{x_nm:8.3f}{y_nm:8.3f}{z_nm:8.3f}"
    if vx is not None and vy is not None and vz is not None:
        # velocities often formatted with 4 decimals
        base += f"{vx:8.4f}{vy:8.4f}{vz:8.4f}"
    return base


def write_gro_from_torch(coords: torch.Tensor,
                          output_path: str,
                          title: str = "Generated by gro_utils.write_gro_from_torch",
                          atom_names: Optional[Sequence[str]] = None,
                          residue_names: Optional[Sequence[str]] = None,
                          residue_numbers: Optional[Sequence[int]] = None,
                          velocities: Optional[torch.Tensor] = None,
                          input_in_angstrom: bool = True,
                          box: Optional[Sequence[float]] = None,
                          box_margin_nm: float = 1.0) -> None:
    """
    Write coords (torch.Tensor shape (n,3)) to a .gro file with strict formatting.

    Parameters:
    - coords: torch.Tensor, shape (n,3)
    - output_path: path to write .gro
    - title: first line string
    - atom_names: optional list of length n
    - residue_names: optional list of length n
    - residue_numbers: optional list of length n
    - velocities: optional torch.Tensor shape (n,3) in same units as coords (will be converted if input_in_angstrom True? -- velocities are typically nm/ps so supply in nm/ps)
    - input_in_angstrom: if True, coords are in angstrom and converted to nm (1 Å = 0.1 nm). Velocities should be in nm/ps already.
    - box: optional (x,y,z) in nm. If None, box will be guessed around coords with box_margin_nm.
    """
    if coords.ndim != 2 or coords.shape[1] != 3:
        raise ValueError("coords must be shape (n,3)")
    n = coords.shape[0]

    coords_np = coords.detach().cpu().numpy().astype(float)
    if input_in_angstrom:
        coords_np = coords_np * 0.1  # Å -> nm

    if box is None:
        min_vals = coords_np.min(axis=0)
        coords_shifted = coords_np - min_vals  # put min at 0
        spans = coords_shifted.max(axis=0) - coords_shifted.min(axis=0)
        spans[spans == 0.0] = 0.1
        box_vec = (float(spans[0] + 2.0 * box_margin_nm),
                   float(spans[1] + 2.0 * box_margin_nm),
                   float(spans[2] + 2.0 * box_margin_nm))
        coords_final = coords_shifted + box_margin_nm
    else:
        if len(box) != 3:
            raise ValueError("box must be length-3 (x,y,z) in nm")
        box_vec = tuple(float(b) for b in box)
        coords_final = coords_np.copy()

    # Prepare names and numbers
    if atom_names is None:
        atom_names = [f"A{(i+1)}" for i in range(n)]
    if residue_names is None:
        residue_names = ["RES"] * n
    if residue_numbers is None:
        residue_numbers = [1] * n

    if len(atom_names) != n or len(residue_names) != n or len(residue_numbers) != n:
        raise ValueError("atom_names, residue_names, residue_numbers must match number of atoms")

    # Velocities handling
    vel_np = None
    if velocities is not None:
        if isinstance(velocities, torch.Tensor):
            vel_np = velocities.detach().cpu().numpy().astype(float)
        else:
            vel_np = np.asarray(velocities, dtype=float)
        if vel_np.shape != (n, 3):
            raise ValueError("velocities must be shape (n,3)")

    # Build lines
    lines = []
    lines.append(title)
    lines.append(f"{n:d}")
    for i in range(n):
        resno = int(residue_numbers[i])
        resname = str(residue_names[i])[:5]
        atname = str(atom_names[i])[:5]
        atno = int(i + 1)
        x, y, z = coords_final[i].tolist()
        if vel_np is not None:
            vx, vy, vz = vel_np[i].tolist()
            atom_line = _format_gro_atom_line_fixed(resno, resname, atname, atno, float(x), float(y), float(z),
                                                    float(vx), float(vy), float(vz))
        else:
            atom_line = _format_gro_atom_line_fixed(resno, resname, atname, atno, float(x), float(y), float(z))
        lines.append(atom_line)

    # Box: write three floats with 5 decimals like your sample used 5 decimals
    box_line = f"{box_vec[0]:.5f} {box_vec[1]:.5f} {box_vec[2]:.5f}"
    lines.append(box_line)

    with open(output_path, "w", newline="\n") as fh:
        fh.write("\n".join(lines) + "\n")

    print(f"Wrote {n} atoms to {output_path} (box {box_vec})")


sample = """Gromacs Runs One Microsecond At Cannonball Speeds
   22
    1ACE   HH31    1  50.293  50.221  50.442 -0.8560 -1.1923 -1.6256
    1ACE    CH3    2  50.302  50.215  50.550 -0.2091 -0.0619  0.0634
    1ACE   HH32    3  50.210  50.214  50.596 -0.5092  3.9413 -2.1491
    1ACE   HH33    4  50.350  50.122  50.577  1.1635  0.0465 -2.7708
    1ACE      C    5  50.394  50.329  50.586 -0.1071 -0.0663 -0.4658
    1ACE      O    6  50.379  50.389  50.691  0.3963  0.4675  0.1718
    2ALA      N    7  50.496  50.346  50.504  0.0854 -0.5069 -0.0122
    2ALA      H    8  50.495  50.288  50.425 -0.2631  2.0137 -0.3027
    2ALA     CA    9  50.596  50.450  50.522 -0.8228 -0.4737 -0.6762
    2ALA     HA   10  50.660  50.443  50.432 -0.5261 -0.7689 -2.2550
    2ALA     CB   11  50.539  50.585  50.518  0.2152 -0.0607 -0.0476
    2ALA    HB1   12  50.472  50.598  50.431 -1.6021 -0.3094 -0.3025
    2ALA    HB2   13  50.483  50.606  50.611  0.5361  1.9825  1.5197
    2ALA    HB3   14  50.622  50.657  50.533  1.2819 -0.6873 -0.1963
    2ALA      C   15  50.693  50.436  50.636 -0.4579 -0.7412  0.0257
    2ALA      O   16  50.813  50.460  50.627 -0.3661 -0.2246 -0.0872
    3NME      N   17  50.645  50.406  50.758  0.0637 -0.3267 -0.0331
    3NME      H   18  50.547  50.400  50.765 -0.9089 -1.8758 -0.5069
    3NME    CH3   19  50.716  50.390  50.888  0.0009  0.3465 -0.4543
    3NME   HH31   20  50.811  50.354  50.886 -1.4041  1.7988  1.4160
    3NME   HH32   21  50.649  50.322  50.939 -2.2681 -0.7791 -2.7750
    3NME   HH33   22  50.716  50.482  50.947 -0.8655 -0.6340  1.0786
 100.89732 100.89732 100.89732
"""
parsed = parse_gro_text(sample)
# If run as main, demonstrate parsing your sample and rewriting it
if __name__ == "__main__":
    
    
    # Re-write exactly (treating coords as nm and using parsed velocities)
    coords_t = torch.tensor(parsed["coords"], dtype=torch.float32)
    vel_list = None
    if any(parsed["velocities"]):
        # convert list of tuples with None to array, replace None with zeros if absent
        vel_list_tmp = []
        has_vel = False
        for v in parsed["velocities"]:
            if v is None:
                vel_list_tmp.append((0.0, 0.0, 0.0))
            else:
                vel_list_tmp.append(v)
                has_vel = True
        if has_vel:
            vel_list = torch.tensor(vel_list_tmp, dtype=torch.float32)
    write_gro_from_torch(coords_t, "rewritten_out.gro",
                         title=parsed["title"],
                         atom_names=parsed["atom_names"],
                         residue_names=parsed["residue_names"],
                         residue_numbers=parsed["residue_numbers"],
                         velocities=vel_list,
                         input_in_angstrom=False,
                         box=parsed["box"] if isinstance(parsed["box"], (list, tuple)) else None)