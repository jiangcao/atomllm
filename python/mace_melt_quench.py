"""
MACE Melt-Quench Script for Generating Amorphous Structure Training Data
=========================================================================
Generates amorphous HfO2, SiO2, and HfSiO4 structures using MACE-MP-0
universal potential via ASE molecular dynamics (melt-quench protocol).

Requirements:
    pip install mace-torch ase numpy tqdm

Usage:
    python mace_melt_quench.py

Output:
    data/structures/   -> individual .extxyz files
    data/dataset.extxyz -> combined dataset for LLM tokenizer
"""

import os
import json
import numpy as np
from tqdm import tqdm
from pathlib import Path

from ase import Atoms
from ase.io import write, read
from ase.md.langevin import Langevin
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.optimize import FIRE
from ase import units

from mace.calculators import mace_mp


# ============================================================
# CONFIGURATION
# ============================================================

CONFIG = {
    # Output
    "output_dir": "data",
    "n_structures_per_composition": 200,  # per composition
    "snapshot_interval": 50,              # save every N MD steps

    # System sizes
    "n_atoms_range": (48, 96),            # randomly sample system size

    # Compositions to generate
    "compositions": [
        {"name": "SiO2",   "elements": {"Si": 1, "O": 2},  "density": 2.2},  # g/cm^3
        {"name": "HfO2",   "elements": {"Hf": 1, "O": 2},  "density": 9.7},
        {"name": "HfSiO4", "elements": {"Hf": 1, "Si": 1, "O": 4}, "density": 6.5},
    ],

    # Melt-quench protocol temperatures (Kelvin)
    "T_melt":   5000,   # high T to randomize structure
    "T_anneal": 2000,   # intermediate anneal
    "T_final":  300,    # room temperature

    # MD timestep and steps
    "timestep_fs": 1.0,       # femtoseconds
    "steps_melt":   500,      # steps at T_melt
    "steps_anneal": 300,      # steps at T_anneal
    "steps_quench": 500,      # steps cooling to T_final
    "steps_final":  200,      # steps at T_final

    # MACE model
    "mace_model": "medium",   # "small", "medium", or "large"
    "device": "cpu",          # "cpu" or "cuda"
}


# ============================================================
# STRUCTURE BUILDER
# ============================================================

def build_random_cell(composition: dict, n_atoms_total: int, density: float) -> Atoms:
    """
    Build a random atom placement in a cubic box at given density.

    Args:
        composition: dict of {element: ratio}, e.g. {"Si": 1, "O": 2}
        n_atoms_total: total number of atoms
        density: target density in g/cm^3

    Returns:
        ASE Atoms object with random positions
    """
    # Determine atom counts from stoichiometry
    total_ratio = sum(composition.values())
    counts = {}
    remaining = n_atoms_total
    elements = list(composition.keys())

    for i, (elem, ratio) in enumerate(composition.items()):
        if i == len(elements) - 1:
            counts[elem] = remaining  # assign remainder to last element
        else:
            counts[elem] = int(round(n_atoms_total * ratio / total_ratio))
            remaining -= counts[elem]

    # Compute box size from density
    from ase.data import atomic_masses, atomic_numbers
    total_mass = sum(
        counts[elem] * atomic_masses[atomic_numbers[elem]]
        for elem in counts
    )
    # mass in amu -> grams, density in g/cm^3 -> volume in cm^3 -> Angstrom^3
    volume_A3 = (total_mass / 6.022e23) / (density / 1e24)
    box_length = volume_A3 ** (1 / 3)

    # Build atom list
    symbols = []
    for elem, count in counts.items():
        symbols.extend([elem] * count)

    # Random positions (avoid overlaps with minimum distance check)
    positions = _random_positions_no_overlap(len(symbols), box_length, min_dist=1.5)

    atoms = Atoms(
        symbols=symbols,
        positions=positions,
        cell=[box_length, box_length, box_length],
        pbc=True
    )

    return atoms


def _random_positions_no_overlap(n_atoms: int, box_length: float,
                                  min_dist: float = 1.5,
                                  max_attempts: int = 10000) -> np.ndarray:
    """Place atoms randomly with minimum distance constraint."""
    positions = []
    attempts = 0

    while len(positions) < n_atoms and attempts < max_attempts:
        candidate = np.random.uniform(0, box_length, 3)
        valid = True

        for existing in positions:
            diff = candidate - existing
            # Minimum image convention
            diff -= box_length * np.round(diff / box_length)
            if np.linalg.norm(diff) < min_dist:
                valid = False
                break

        if valid:
            positions.append(candidate)
        attempts += 1

    if len(positions) < n_atoms:
        # Fill remaining with fully random (overlap allowed as fallback)
        while len(positions) < n_atoms:
            positions.append(np.random.uniform(0, box_length, 3))

    return np.array(positions)


# ============================================================
# MELT-QUENCH PROTOCOL
# ============================================================

def run_melt_quench(atoms: Atoms, calculator, config: dict,
                    label: str = "") -> list[Atoms]:
    """
    Run full melt-quench MD protocol and collect structure snapshots.

    Protocol:
        1. Energy minimization (remove worst clashes)
        2. Heat to T_melt (randomize structure)
        3. Anneal at T_anneal
        4. Quench to T_final
        5. Equilibrate at T_final

    Returns:
        List of ASE Atoms snapshots
    """
    atoms.calc = calculator
    snapshots = []

    def collect_snapshot(step_label):
        """Save a copy of current structure with metadata."""
        snap = atoms.copy()
        snap.info["step_label"] = step_label
        snap.info["label"] = label
        snap.info["energy"] = atoms.get_potential_energy()
        snap.info["energy_per_atom"] = atoms.get_potential_energy() / len(atoms)
        snap.arrays["forces"] = atoms.get_forces().copy()
        snapshots.append(snap)

    # ---- Stage 1: Quick energy minimization ----
    optimizer = FIRE(atoms, logfile=None)
    optimizer.run(fmax=1.0, steps=50)   # loose convergence — just remove bad overlaps
    collect_snapshot("minimized")

    dt = config["timestep_fs"] * units.fs
    friction = 0.02   # Langevin friction coefficient

    # ---- Stage 2: Melt at T_melt ----
    MaxwellBoltzmannDistribution(atoms, temperature_K=config["T_melt"])
    dyn_melt = Langevin(atoms, dt, temperature_K=config["T_melt"],
                        friction=friction, logfile=None)

    interval = config["snapshot_interval"]
    for step in range(config["steps_melt"]):
        dyn_melt.run(1)
        if step % interval == 0:
            collect_snapshot(f"melt_{step}")

    # ---- Stage 3: Anneal at T_anneal ----
    dyn_anneal = Langevin(atoms, dt, temperature_K=config["T_anneal"],
                          friction=friction, logfile=None)
    for step in range(config["steps_anneal"]):
        dyn_anneal.run(1)
        if step % interval == 0:
            collect_snapshot(f"anneal_{step}")

    # ---- Stage 4: Quench to T_final (linear cooling) ----
    T_start = config["T_anneal"]
    T_end = config["T_final"]
    steps_q = config["steps_quench"]

    for step in range(steps_q):
        T_current = T_start + (T_end - T_start) * step / steps_q
        dyn_q = Langevin(atoms, dt, temperature_K=T_current,
                         friction=friction, logfile=None)
        dyn_q.run(1)
        if step % interval == 0:
            collect_snapshot(f"quench_{step}_T{int(T_current)}")

    # ---- Stage 5: Equilibrate at T_final ----
    dyn_final = Langevin(atoms, dt, temperature_K=config["T_final"],
                         friction=friction, logfile=None)
    for step in range(config["steps_final"]):
        dyn_final.run(1)
        if step % interval == 0:
            collect_snapshot(f"final_{step}")

    return snapshots


# ============================================================
# DATASET GENERATION
# ============================================================

def generate_dataset(config: dict):
    """
    Main function: generate amorphous structure dataset for all compositions.
    """
    # Setup output directories
    out_dir = Path(config["output_dir"])
    struct_dir = out_dir / "structures"
    struct_dir.mkdir(parents=True, exist_ok=True)

    # Load MACE-MP-0 calculator
    print("Loading MACE-MP-0 calculator...")
    calculator = mace_mp(
        model=config["mace_model"],
        device=config["device"],
        default_dtype="float32"
    )
    print(f"  -> Model loaded on {config['device']}")

    all_snapshots = []
    metadata = []

    for comp in config["compositions"]:
        comp_name = comp["name"]
        print(f"\n{'='*50}")
        print(f"Generating {config['n_structures_per_composition']} "
              f"runs for {comp_name}")
        print(f"{'='*50}")

        comp_dir = struct_dir / comp_name
        comp_dir.mkdir(exist_ok=True)

        for run_idx in tqdm(range(config["n_structures_per_composition"]),
                            desc=comp_name):

            # Randomize system size
            n_atoms = np.random.randint(*config["n_atoms_range"])

            # Build random starting structure
            atoms = build_random_cell(
                composition=comp["elements"],
                n_atoms_total=n_atoms,
                density=comp["density"]
            )

            label = f"{comp_name}_run{run_idx:04d}_N{n_atoms}"

            try:
                # Run melt-quench
                snapshots = run_melt_quench(
                    atoms=atoms,
                    calculator=calculator,
                    config=config,
                    label=label
                )

                # Save individual run snapshots
                run_file = comp_dir / f"{label}.extxyz"
                write(str(run_file), snapshots)

                all_snapshots.extend(snapshots)

                metadata.append({
                    "label": label,
                    "composition": comp_name,
                    "n_atoms": n_atoms,
                    "n_snapshots": len(snapshots),
                    "file": str(run_file),
                })

            except Exception as e:
                print(f"\n  Warning: run {label} failed: {e}")
                continue

    # ---- Save combined dataset ----
    print(f"\nSaving combined dataset ({len(all_snapshots)} structures)...")
    dataset_file = out_dir / "dataset.extxyz"
    write(str(dataset_file), all_snapshots)

    # ---- Save metadata ----
    meta_file = out_dir / "metadata.json"
    with open(meta_file, "w") as f:
        json.dump({
            "config": config,
            "total_structures": len(all_snapshots),
            "runs": metadata,
            "compositions": {
                comp["name"]: sum(
                    m["n_snapshots"] for m in metadata
                    if m["composition"] == comp["name"]
                )
                for comp in config["compositions"]
            }
        }, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Dataset generation complete!")
    print(f"  Total structures : {len(all_snapshots)}")
    print(f"  Dataset file     : {dataset_file}")
    print(f"  Metadata         : {meta_file}")
    print(f"  Structure files  : {struct_dir}")
    print(f"{'='*50}")

    return all_snapshots, metadata


# ============================================================
# ANALYSIS / VALIDATION UTILITIES
# ============================================================

def compute_rdf(atoms: Atoms, element_pair: tuple,
                r_max: float = 6.0, n_bins: int = 200) -> tuple:
    """
    Compute radial distribution function g(r) for a given element pair.

    Args:
        atoms: ASE Atoms object
        element_pair: e.g. ("Si", "O")
        r_max: maximum distance in Angstroms
        n_bins: number of histogram bins

    Returns:
        (r_values, g_r) arrays
    """
    from ase.geometry.analysis import Analysis

    ana = Analysis(atoms)
    elem_a, elem_b = element_pair

    r_bins = np.linspace(0, r_max, n_bins)
    dr = r_bins[1] - r_bins[0]
    rdf = np.zeros(n_bins - 1)

    positions = atoms.get_positions()
    cell = atoms.get_cell()
    symbols = atoms.get_chemical_symbols()

    idx_a = [i for i, s in enumerate(symbols) if s == elem_a]
    idx_b = [i for i, s in enumerate(symbols) if s == elem_b]

    count = 0
    for i in idx_a:
        for j in idx_b:
            if i == j:
                continue
            diff = positions[i] - positions[j]
            # Minimum image convention
            diff -= np.dot(np.round(np.dot(diff, np.linalg.inv(cell))), cell)
            r = np.linalg.norm(diff)
            if r < r_max:
                bin_idx = int(r / dr)
                if bin_idx < len(rdf):
                    rdf[bin_idx] += 1
            count += 1

    # Normalize to g(r)
    volume = atoms.get_volume()
    n_b = len(idx_b)
    rdf_normalized = rdf / (
        len(idx_a) * n_b / volume * 4 * np.pi * r_bins[:-1]**2 * dr
    )

    return r_bins[:-1] + dr / 2, rdf_normalized


def validate_dataset_sample(dataset_file: str, n_sample: int = 5):
    """
    Quick validation: print bond length stats for a sample of structures.

    Expected values:
        Si-O: ~1.62 Å (first peak)
        Hf-O: ~2.10 Å (first peak)
    """
    print("\nValidation: bond length statistics")
    print("-" * 40)

    structures = read(dataset_file, index=f":{n_sample}")

    for atoms in structures:
        label = atoms.info.get("label", "unknown")
        energy_pa = atoms.info.get("energy_per_atom", float("nan"))
        symbols = set(atoms.get_chemical_symbols())

        print(f"\n{label}")
        print(f"  N_atoms={len(atoms)}, E/atom={energy_pa:.3f} eV")

        pairs_to_check = []
        if "Si" in symbols and "O" in symbols:
            pairs_to_check.append(("Si", "O"))
        if "Hf" in symbols and "O" in symbols:
            pairs_to_check.append(("Hf", "O"))

        for pair in pairs_to_check:
            r, g = compute_rdf(atoms, pair, r_max=4.0, n_bins=100)
            peak_r = r[np.argmax(g)]
            print(f"  {pair[0]}-{pair[1]} first peak: {peak_r:.2f} Å")


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate amorphous structure dataset using MACE melt-quench"
    )
    parser.add_argument("--n_runs", type=int, default=None,
                        help="Override n_structures_per_composition")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cpu/cuda)")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation on existing dataset")
    args = parser.parse_args()

    if args.n_runs:
        CONFIG["n_structures_per_composition"] = args.n_runs
    if args.device:
        CONFIG["device"] = args.device

    if args.validate:
        validate_dataset_sample("data/dataset.extxyz")
    else:
        snapshots, metadata = generate_dataset(CONFIG)
        validate_dataset_sample("data/dataset.extxyz")
