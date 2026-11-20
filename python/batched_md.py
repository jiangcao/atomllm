"""
Batched MACE molecular dynamics and relaxation on the GPU.

ASE drives one structure per calculator call, and for the small cells used
here (48-96 atoms) a call is dominated by fixed per-call overhead, not GPU
work. This module advances many independent structures with a single MACE
forward/backward pass per step: all atoms are concatenated into one graph
(as in MACE training batches), the neighbour list is built on the GPU, and
the Langevin / FIRE integrators are vectorised over structures.

Units are ASE units throughout (eV, Å, amu, time in Å·sqrt(amu/eV)).
"""

import numpy as np
import torch
from ase import Atoms, units

from mace import data as mace_data
from mace.tools import torch_geometric


class BatchedSystem:
    """A set of independent periodic structures evaluated in one MACE call."""

    def __init__(self, atoms_list: list[Atoms], calculator):
        self.calc = calculator
        self.model = calculator.models[0]
        self.device = torch.device(calculator.device)
        self.r_max = float(calculator.r_max)
        self.n_struct = len(atoms_list)
        self.atoms_list = [a.copy() for a in atoms_list]

        for a in self.atoms_list:
            cell = np.asarray(a.cell)
            if not a.pbc.all() or np.abs(cell - np.diag(np.diag(cell))).max() > 1e-8:
                raise ValueError("BatchedSystem needs periodic orthorhombic cells")
            a.wrap()

        # Static part of the model input (species, cell, head, batch index):
        # built once with MACE's own machinery, edges are replaced every call.
        keyspec = mace_data.KeySpecification(
            info_keys=calculator.info_keys, arrays_keys=calculator.arrays_keys)
        dataset = [
            mace_data.AtomicData.from_config(
                mace_data.config_from_atoms(a, key_specification=keyspec,
                                            head_name=calculator.head),
                z_table=calculator.z_table, cutoff=self.r_max,
                heads=calculator.available_heads)
            for a in self.atoms_list
        ]
        loader = torch_geometric.dataloader.DataLoader(
            dataset=dataset, batch_size=len(dataset), shuffle=False, drop_last=False)
        self.base = next(iter(loader)).to(self.device).to_dict()
        self.mdtype = self.base["positions"].dtype      # model precision

        f64 = dict(dtype=torch.float64, device=self.device)
        self.batch = self.base["batch"]                                  # (n_atoms,)
        self.n_atoms = torch.bincount(self.batch, minlength=self.n_struct)
        # atoms held fixed by MD and FIRE (e.g. a crystalline substrate);
        # set from outside, may change during a run
        self.frozen = torch.zeros(len(self.batch), dtype=torch.bool, device=self.device)
        self.pos = torch.tensor(
            np.concatenate([a.positions for a in self.atoms_list]), **f64)
        self.masses = torch.tensor(
            np.concatenate([a.get_masses() for a in self.atoms_list]), **f64)
        self.box = torch.tensor(
            np.array([np.diag(np.asarray(a.cell)) for a in self.atoms_list]), **f64)

        # Padded (structure, slot) -> flat atom index, for the neighbour list
        n_max = int(self.n_atoms.max())
        ptr = self.base["ptr"]
        slot = torch.arange(n_max, device=self.device)[None, :]
        self.pad_mask = slot < self.n_atoms[:, None]                     # (B, n_max)
        self.pad_idx = (ptr[:-1, None] + slot).clamp(max=len(self.pos) - 1)

        # Periodic images needed to cover r_max (1 shell when box >= r_max)
        n_rep = int(torch.ceil(self.r_max / self.box.min()).item())
        rng = torch.arange(-n_rep, n_rep + 1, device=self.device)
        self.unit_shifts = torch.cartesian_prod(rng, rng, rng)           # (S, 3)
        self.zero_shift = (self.unit_shifts == 0).all(dim=1)             # (S,)

    # ------------------------------------------------------------------
    def wrap(self):
        box = self.box[self.batch]
        self.pos -= torch.floor(self.pos / box) * box

    def _neighbour_list(self, pos):
        """All ordered pairs within r_max, MACE convention:
        vector = pos[receiver] - pos[sender] + shift."""
        p = pos[self.pad_idx]                                            # (B, N, 3)
        shift_vec = self.unit_shifts.to(pos.dtype)[None] * self.box[:, None, :]  # (B, S, 3)
        d = p[:, None, :, :] - p[:, :, None, :]                          # (B, i, j, 3)
        d = d[:, :, :, None, :] + shift_vec[:, None, None, :, :]         # (B, i, j, S, 3)
        within = (d * d).sum(-1) < self.r_max ** 2
        within &= self.pad_mask[:, :, None, None] & self.pad_mask[:, None, :, None]
        n = p.shape[1]
        self_pair = torch.eye(n, dtype=torch.bool, device=pos.device)
        within &= ~(self_pair[None, :, :, None] & self.zero_shift[None, None, None, :])
        b, i, j, s = within.nonzero(as_tuple=True)
        edge_index = torch.stack([self.pad_idx[b, i], self.pad_idx[b, j]])
        return edge_index, self.unit_shifts[s], shift_vec[b, s]

    def compute(self):
        """Energies (B,) and forces (n_atoms, 3) at the current positions."""
        self.wrap()
        edge_index, unit_shifts, shifts = self._neighbour_list(self.pos)
        data = dict(self.base)
        data["positions"] = self.pos.to(self.mdtype)
        data["edge_index"] = edge_index
        data["unit_shifts"] = unit_shifts.to(self.mdtype)
        data["shifts"] = shifts.to(self.mdtype)
        out = self.model(data, compute_force=True, training=False)
        energy = out["energy"].detach().to(torch.float64)
        forces = out["forces"].detach().to(torch.float64)
        return energy, forces

    # ------------------------------------------------------------------
    def per_struct_sum(self, x):
        """Sum a per-atom quantity (n_atoms,) or (n_atoms, k) per structure."""
        out = torch.zeros((self.n_struct,) + x.shape[1:], dtype=x.dtype, device=x.device)
        return out.index_add_(0, self.batch, x)

    def per_struct_max(self, x):
        out = torch.zeros(self.n_struct, dtype=x.dtype, device=x.device)
        return out.scatter_reduce_(0, self.batch, x, reduce="amax", include_self=True)

    def to_atoms(self, pos=None) -> list[Atoms]:
        """Copies of the structures with the current (or given) positions."""
        pos = (self.pos if pos is None else pos).cpu().numpy()
        out, start = [], 0
        for a in self.atoms_list:
            new = a.copy()
            new.positions = pos[start:start + len(a)]
            start += len(a)
            out.append(new)
        return out


class BatchedLangevin:
    """BAOAB Langevin dynamics for all structures of a BatchedSystem.

    Structures whose forces become non-finite or explode are frozen and
    flagged in `self.failed` instead of aborting the whole batch.
    """

    MAX_FORCE = 1e4   # eV/Å; anything above this is a blown-up structure

    def __init__(self, system: BatchedSystem, timestep: float, friction: float,
                 temperature_K: float, seed: int = 0):
        self.sys = system
        self.dt = timestep
        self.friction = friction
        self.gen = torch.Generator(device=system.device).manual_seed(seed)
        self.failed = torch.zeros(system.n_struct, dtype=torch.bool, device=system.device)
        self.m = system.masses[:, None]
        self.vel = torch.zeros_like(system.pos)
        self.set_temperature(temperature_K)
        self.vel = self._noise() * torch.sqrt(self.kT / self.m)
        self._remove_com()
        _, self.forces = system.compute()
        self._check(self.forces)

    def set_temperature(self, temperature_K: float):
        self.kT = units.kB * temperature_K

    def _noise(self):
        return torch.randn(self.vel.shape, generator=self.gen,
                           dtype=self.vel.dtype, device=self.vel.device)

    def _remove_com(self):
        s = self.sys
        p = s.per_struct_sum(self.vel * self.m)
        mtot = s.per_struct_sum(s.masses)
        self.vel -= (p / mtot[:, None])[s.batch]

    def _check(self, forces):
        s = self.sys
        fmax = s.per_struct_max(torch.nan_to_num(
            forces.norm(dim=1), nan=float("inf"), posinf=float("inf")))
        self.failed |= fmax > self.MAX_FORCE
        dead = self.failed[s.batch]
        if dead.any():
            forces[dead] = 0.0
            self.vel[dead] = 0.0
            # park blown-up atoms at valid coordinates so they stay harmless
            s.pos[dead] = torch.nan_to_num(s.pos[dead], nan=0.0, posinf=0.0, neginf=0.0)

    def run(self, n_steps: int):
        s, dt = self.sys, self.dt
        c1 = float(np.exp(-self.friction * dt))
        c2 = float(np.sqrt(1.0 - c1 * c1))
        for _ in range(n_steps):
            alive = (~self.failed)[s.batch][:, None]
            self.vel += 0.5 * dt * self.forces / self.m
            self.vel[s.frozen] = 0.0
            s.pos += 0.5 * dt * self.vel * alive
            self.vel = c1 * self.vel + c2 * torch.sqrt(self.kT / self.m) * self._noise()
            self._remove_com()
            self.vel *= alive
            self.vel[s.frozen] = 0.0
            s.pos += 0.5 * dt * self.vel
            _, self.forces = s.compute()
            self._check(self.forces)
            self.vel += 0.5 * dt * self.forces / self.m

    def temperatures(self):
        """Instantaneous kinetic temperature (K) per structure."""
        s = self.sys
        ke = 0.5 * s.per_struct_sum((self.m * self.vel ** 2).sum(dim=1))
        return (2.0 * ke / (3.0 * s.n_atoms * units.kB)).cpu().numpy()


def batched_fire(system: BatchedSystem, fmax: float, steps: int, pos=None,
                 dt_start=0.1, dt_max=1.0, max_step=0.2, n_min=5,
                 f_inc=1.1, f_dec=0.5, a_start=0.1, f_a=0.99):
    """
    FIRE relaxation (same scheme and defaults as ase.optimize.FIRE) of every
    structure at once, each with its own time step and convergence test.

    Starts from `pos` (default: the system's current positions) and leaves
    the relaxed positions in system.pos. Returns
    (energies (B,), forces (n_atoms, 3), converged (B,) bool, n_steps).
    """
    s = system
    if pos is not None:
        s.pos = pos.clone()
    B = s.n_struct
    f64 = dict(dtype=torch.float64, device=s.device)
    vel = torch.zeros_like(s.pos)
    dt = torch.full((B,), dt_start, **f64)
    a = torch.full((B,), a_start, **f64)
    n_pos = torch.zeros(B, dtype=torch.long, device=s.device)

    step = 0
    while True:
        energy, forces = s.compute()
        forces = forces.clone()
        forces[s.frozen] = 0.0
        f_norm = torch.nan_to_num(forces.norm(dim=1), nan=float("inf"))
        converged = s.per_struct_max(f_norm) < fmax
        if converged.all() or step >= steps:
            return energy, forces, converged, step
        forces = torch.nan_to_num(forces, nan=0.0, posinf=0.0, neginf=0.0)

        if step > 0:
            power = s.per_struct_sum((forces * vel).sum(dim=1))
            downhill = power > 0
            v_len = torch.sqrt(s.per_struct_sum((vel * vel).sum(dim=1)))
            f_len = torch.sqrt(s.per_struct_sum((forces * forces).sum(dim=1))).clamp(min=1e-30)
            mixed = ((1.0 - a)[s.batch][:, None] * vel
                     + (a * v_len / f_len)[s.batch][:, None] * forces)
            vel = torch.where(downhill[s.batch][:, None], mixed, torch.zeros_like(vel))
            grow = downhill & (n_pos > n_min)
            dt = torch.where(grow, (dt * f_inc).clamp(max=dt_max), dt)
            dt = torch.where(downhill, dt, dt * f_dec)
            a = torch.where(grow, a * f_a, a)
            a = torch.where(downhill, a, torch.full_like(a, a_start))
            n_pos = torch.where(downhill, n_pos + 1, torch.zeros_like(n_pos))

        vel = vel + dt[s.batch][:, None] * forces
        dr = dt[s.batch][:, None] * vel
        dr_len = torch.sqrt(s.per_struct_sum((dr * dr).sum(dim=1)))
        scale = torch.where(dr_len > max_step, max_step / dr_len.clamp(min=1e-30),
                            torch.ones_like(dr_len))
        dr = dr * scale[s.batch][:, None]
        # converged structures stay put while the others finish
        s.pos += dr * (~converged)[s.batch][:, None]
        step += 1
