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
import zlib
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

from structure_defects import defect_summary


# ============================================================
# CONFIGURATION
# ============================================================
CONFIG = {
    # Output
    "output_dir": "data",
    "n_structures_per_composition": 100,  # melt-quench runs per composition

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

    # MD timestep and steps.
    # The quench must be slow: a 0.5 ps quench (~3e15 K/s) left every SiO2
    # structure with 3- and 5-fold Si and non-bridging O. 20 ps ≈ 8.5e13 K/s.
    "timestep_fs": 1.0,
    "steps_melt":    3000,
    "steps_anneal":  3000,
    "steps_quench": 20000,
    "steps_final":   5000,

    # Training frames (all relaxed at fixed cell with MACE; each is labelled
    # clean/defective by structure_defects.py and the label becomes a
    # conditioning token, so defective frames are kept as training data):
    #   - every quench_sample_every steps once the quench is below
    #     quench_sample_below_T (no liquid / high-T frames)
    #   - n_final_samples evenly spaced frames during the final 300 K stage
    "quench_sample_below_T": 1000,
    "quench_sample_every":   1000,
    "n_final_samples": 5,
    "relax_fmax": 0.02,     # eV/Å
    "relax_steps": 500,

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
                    label: str = "") -> tuple[list[Atoms], list[dict]]:
    """
    Melt-quench and return relaxed frames, each labelled clean or defective.

    Protocol:
        1. Energy minimization (remove worst clashes)
        2. Melt at T_melt, anneal at T_anneal
        3. Linear quench to T_final, sampling frames below quench_sample_below_T
        4. Equilibrate at T_final, sampling n_final_samples frames

    Every sampled frame is relaxed (fixed cell). Liquid / high-T frames are
    never returned: their density and bond angles are wrong, not just their
    coordination.

    Returns:
        (frames, sample_log); frames carry info["defect_free"] and
        info["n_defects"].
    """
    atoms.calc = calculator
    dt = config["timestep_fs"] * units.fs
    friction = 0.02   # Langevin friction coefficient
    frames, log = [], []

    def sample(step_label):
        frame = atoms.copy()
        frame.calc = calculator
        FIRE(frame, logfile=None).run(fmax=config["relax_fmax"],
                                      steps=config["relax_steps"])
        defects = defect_summary(frame)
        e = frame.get_potential_energy()
        log.append({"step_label": step_label, "energy_per_atom": e / len(frame),
                    "defects": dict(defects)})
        out = frame.copy()          # drops the calculator
        out.info.update({"label": label, "step_label": step_label,
                         "energy": e, "energy_per_atom": e / len(frame),
                         "defect_free": not defects,
                         "n_defects": sum(defects.values())})
        out.arrays["forces"] = frame.get_forces().copy()
        frames.append(out)

    FIRE(atoms, logfile=None).run(fmax=1.0, steps=50)

    MaxwellBoltzmannDistribution(atoms, temperature_K=config["T_melt"])
    Langevin(atoms, dt, temperature_K=config["T_melt"],
             friction=friction, logfile=None).run(config["steps_melt"])
    Langevin(atoms, dt, temperature_K=config["T_anneal"],
             friction=friction, logfile=None).run(config["steps_anneal"])

    # Linear quench: update the thermostat target in blocks of 100 steps
    T_start, T_end = config["T_anneal"], config["T_final"]
    steps_q = config["steps_quench"]
    dyn_q = Langevin(atoms, dt, temperature_K=T_start,
                     friction=friction, logfile=None)
    block = 100
    every = config["quench_sample_every"]
    for s in range(0, steps_q, block):
        T_now = T_start + (T_end - T_start) * s / steps_q
        dyn_q.set_temperature(temperature_K=T_now)
        dyn_q.run(min(block, steps_q - s))
        if (s + block) % every == 0 and T_now < config["quench_sample_below_T"]:
            sample(f"quench_{s + block}_T{int(T_now)}")

    # Equilibrate at T_final and sample relaxed frames
    dyn_f = Langevin(atoms, dt, temperature_K=T_end,
                     friction=friction, logfile=None)
    n_samp = config["n_final_samples"]
    chunk = config["steps_final"] // n_samp
    for k in range(n_samp):
        dyn_f.run(chunk)
        sample(f"final_{k}")

    return frames, log


# ============================================================
# DATASET GENERATION
# ============================================================

def _composition_by_name(config: dict, name: str) -> dict:
    return next(c for c in config["compositions"] if c["name"] == name)


def generate_dataset(config: dict, worker: int = 0, n_workers: int = 1):
    """
    Run melt-quench for every (composition, run index) assigned to this
    worker (round-robin), writing one .extxyz (clean frames) + one .json
    (sample log) per run under data/structures/<comp>/. Runs whose .json
    already exists are skipped, so jobs can be restarted. Combine with
    merge_dataset() afterwards.
    """
    struct_dir = Path(config["output_dir"]) / "structures"

    print(f"[worker {worker}/{n_workers}] loading MACE-MP-0 ({config['mace_model']}) "
          f"on {config['device']}")
    calculator = mace_mp(model=config["mace_model"], device=config["device"],
                         default_dtype="float32")

    jobs = [(c["name"], i) for c in config["compositions"]
            for i in range(config["n_structures_per_composition"])]
    jobs = jobs[worker::n_workers]

    for comp_name, run_idx in tqdm(jobs, desc=f"worker {worker}"):
        comp = _composition_by_name(config, comp_name)
        comp_dir = struct_dir / comp_name
        comp_dir.mkdir(parents=True, exist_ok=True)
        if list(comp_dir.glob(f"{comp_name}_run{run_idx:04d}_N*.json")):
            continue

        rng_seed = (zlib.crc32(comp_name.encode()) % 10_000) * 100_000 + run_idx
        np.random.seed(rng_seed)
        # Whole formula units only: off-stoichiometry forces defects
        fu_size = sum(comp["elements"].values())
        n_atoms = fu_size * max(1, round(np.random.randint(*config["n_atoms_range"]) / fu_size))
        atoms = build_random_cell(comp["elements"], n_atoms, comp["density"])
        label = f"{comp_name}_run{run_idx:04d}_N{n_atoms}"

        try:
            frames, log = run_melt_quench(atoms, calculator, config, label)
        except Exception as e:
            print(f"\n  Warning: run {label} failed: {e}")
            continue

        write(str(comp_dir / f"{label}.extxyz"), frames)
        with open(comp_dir / f"{label}.json", "w") as f:
            json.dump({"label": label, "composition": comp_name,
                       "n_atoms": int(n_atoms), "seed": rng_seed,
                       "n_frames": len(frames),
                       "n_clean": sum(fr.info["defect_free"] for fr in frames),
                       "samples": log}, f, indent=1)


def merge_dataset(config: dict):
    """Combine per-run labelled frames into data/dataset.extxyz + metadata.json."""
    out_dir = Path(config["output_dir"])
    struct_dir = out_dir / "structures"
    frames, runs = [], []
    for comp in config["compositions"]:
        for jf in sorted((struct_dir / comp["name"]).glob("*.json")):
            with open(jf) as f:
                meta = json.load(f)
            runs.append(meta)
            xyz = jf.with_suffix(".extxyz")
            if meta["n_frames"] and xyz.exists():
                frames.extend(read(str(xyz), index=":"))

    write(str(out_dir / "dataset.extxyz"), frames)
    summary = {}
    for comp in config["compositions"]:
        rs = [r for r in runs if r["composition"] == comp["name"]]
        summary[comp["name"]] = {
            "runs": len(rs),
            "sampled_frames": sum(len(r["samples"]) for r in rs),
            "clean_frames": sum(r["n_clean"] for r in rs),
        }
    with open(out_dir / "metadata.json", "w") as f:
        json.dump({"config": config, "total_structures": len(frames),
                   "compositions": summary, "runs": runs}, f, indent=1)

    print(f"Merged {len(frames)} labelled frames into {out_dir/'dataset.extxyz'}")
    for name, s in summary.items():
        print(f"  {name:7s}: {s['clean_frames']}/{s['sampled_frames']} "
              f"sampled frames defect-free ({s['runs']} runs)")


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
        description="Generate defect-free amorphous training data with MACE melt-quench"
    )
    parser.add_argument("--n_runs", type=int, default=None,
                        help="Override n_structures_per_composition")
    parser.add_argument("--compositions", nargs="+", default=None,
                        help="Subset of compositions, e.g. SiO2 HfO2")
    parser.add_argument("--device", type=str, default=None,
                        help="Override device (cpu/cuda)")
    parser.add_argument("--steps_quench", type=int, default=None,
                        help="Override quench length (steps of 1 fs)")
    parser.add_argument("--worker", type=int, default=0,
                        help="Index of this worker (for parallel runs)")
    parser.add_argument("--n_workers", type=int, default=1,
                        help="Total number of parallel workers")
    parser.add_argument("--merge", action="store_true",
                        help="Only merge per-run files into dataset.extxyz")
    parser.add_argument("--validate", action="store_true",
                        help="Run validation on existing dataset")
    args = parser.parse_args()

    if args.n_runs:
        CONFIG["n_structures_per_composition"] = args.n_runs
    if args.device:
        CONFIG["device"] = args.device
    if args.steps_quench:
        CONFIG["steps_quench"] = args.steps_quench
    if args.compositions:
        CONFIG["compositions"] = [c for c in CONFIG["compositions"]
                                  if c["name"] in args.compositions]

    if args.validate:
        validate_dataset_sample("data/dataset.extxyz")
    elif args.merge:
        merge_dataset(CONFIG)
    else:
        generate_dataset(CONFIG, worker=args.worker, n_workers=args.n_workers)
        if args.n_workers == 1:
            merge_dataset(CONFIG)
