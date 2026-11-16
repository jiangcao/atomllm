"""
Silicon Nanosheet MD Simulation using ASE and MACE-MLIP

Creates a 1nm × 3nm × 20nm Silicon nanosheet surrounded by 5nm thick SiO2 layer
and performs molecular dynamics using the MACE machine learning potential.

Requirements:
    pip install ase mace-torch torch
"""

import numpy as np
from ase import Atoms
from ase.build import bulk, add_vacuum
from ase.io import write, read
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.md.verlet import VelocityVerlet
from ase.units import kB
import matplotlib.pyplot as plt

# Try to import MACE, fallback to EMT for testing
try:
    from mace.calculators import MACECalculator
    MACE_AVAILABLE = True
except ImportError:
    MACE_AVAILABLE = False
    print("Warning: MACE not installed. Install with: pip install mace-torch")
    try:
        from ase.calculators.emt import EMT
        print("Falling back to EMT calculator for testing purposes.")
    except:
        pass


def create_si_nanosheet(xs=1.0, ys=3.0, zs=20.0, a=5.431):
    """
    Create a Silicon nanosheet with specified dimensions.
    
    Parameters:
    -----------
    xs : float
        Cross-section in x-direction (nm)
    ys : float
        Cross-section in y-direction (nm)
    zs : float
        Length in z-direction (nm)
    a : float
        Si lattice constant (Angstrom), default 5.431 for Si
    
    Returns:
    --------
    atoms : ASE Atoms object
        Silicon nanosheet structure
    """
    
    # Convert nm to Angstrom
    xs_aa = xs * 10
    ys_aa = ys * 10
    zs_aa = zs * 10
    
    # Create bulk Si structure
    si_bulk = bulk('Si', 'diamond', a=a)
    
    # Calculate number of unit cells needed
    nx = int(np.ceil(xs_aa / a)) + 1
    ny = int(np.ceil(ys_aa / a)) + 1
    nz = int(np.ceil(zs_aa / a)) + 1
    
    # Replicate the unit cell
    nanosheet = si_bulk.repeat((nx, ny, nz))
    
    # Center the structure
    nanosheet.center(vacuum=0, axis=0)
    nanosheet.center(vacuum=0, axis=1)
    nanosheet.center(vacuum=0, axis=2)
    
    # Get current dimensions
    cell = nanosheet.get_cell()
    
    # Crop to desired size by removing atoms outside boundaries
    pos = nanosheet.get_positions()
    
    # Select atoms within the specified dimensions
    mask = (
        (pos[:, 0] >= -xs_aa/2) & (pos[:, 0] <= xs_aa/2) &
        (pos[:, 1] >= -ys_aa/2) & (pos[:, 1] <= ys_aa/2) &
        (pos[:, 2] >= -zs_aa/2) & (pos[:, 2] <= zs_aa/2)
    )
    
    nanosheet = nanosheet[mask]
    
    # Set cell dimensions with vacuum on sides
    nanosheet.set_cell([xs_aa + 15, ys_aa + 15, zs_aa + 15], scale_atoms=False)
    nanosheet.center()
    
    return nanosheet


def create_sio2_coating(si_nanosheet, thickness=5.0):
    """
    Create SiO2 coating around the Si nanosheet.
    
    Parameters:
    -----------
    si_nanosheet : ASE Atoms object
        Silicon nanosheet structure
    thickness : float
        Thickness of SiO2 layer (nm)
    
    Returns:
    --------
    sio2_atoms : ASE Atoms object
        SiO2 atoms to be added
    """
    
    thickness_aa = thickness * 10  # Convert to Angstrom
    
    # SiO2 lattice parameter (approximate, for cristobalite structure)
    # We'll create a simple approach using random placement with constraints
    sio2_atoms = Atoms()
    
    # Get bounding box of Si nanosheet
    pos_si = si_nanosheet.get_positions()
    
    x_min, x_max = pos_si[:, 0].min(), pos_si[:, 0].max()
    y_min, y_max = pos_si[:, 1].min(), pos_si[:, 1].max()
    z_min, z_max = pos_si[:, 2].min(), pos_si[:, 2].max()
    
    # Extend boundaries for SiO2 coating
    x_min_sio2, x_max_sio2 = x_min - thickness_aa, x_max + thickness_aa
    y_min_sio2, y_max_sio2 = y_min - thickness_aa, y_max + thickness_aa
    z_min_sio2, z_max_sio2 = z_min - thickness_aa, z_max + thickness_aa
    
    # Create crystalline SiO2 (simplified - using amorphous approach)
    # For a more accurate structure, consider using specialized libraries
    
    sio2_density = 2.2  # g/cm^3 for SiO2
    molar_mass_sio2 = 60.08  # g/mol
    avogadro = 6.022e23
    
    # Calculate volume of SiO2 shell
    outer_vol = (x_max_sio2 - x_min_sio2) * (y_max_sio2 - y_min_sio2) * (z_max_sio2 - z_min_sio2)
    inner_vol = (x_max - x_min) * (y_max - y_min) * (z_max - z_min)
    shell_vol = outer_vol - inner_vol  # in Angstrom^3
    shell_vol_cm3 = shell_vol * 1e-24  # Convert to cm^3
    
    # Calculate number of SiO2 molecules needed
    mass_sio2 = sio2_density * shell_vol_cm3  # grams
    num_sio2_molecules = (mass_sio2 / molar_mass_sio2) * avogadro
    
    # For each SiO2 molecule, we need 1 Si and 2 O atoms
    num_si_atoms = int(num_sio2_molecules)
    num_o_atoms = 2 * num_si_atoms
    
    print(f"\nSiO2 Coating Information:")
    print(f"  Shell volume: {shell_vol:.2e} Ų")
    print(f"  Number of SiO2 molecules: {num_sio2_molecules:.2e}")
    print(f"  Si atoms in coating: {num_si_atoms}")
    print(f"  O atoms in coating: {num_o_atoms}")
    
    # Create SiO2 atoms with random positions (amorphous approximation)
    np.random.seed(42)
    
    # Generate positions in the shell region
    positions = []
    symbols = []
    
    # Place Si atoms randomly in the shell
    for i in range(num_si_atoms):
        # Use rejection sampling to stay within shell
        valid = False
        while not valid:
            x = np.random.uniform(x_min_sio2, x_max_sio2)
            y = np.random.uniform(y_min_sio2, y_max_sio2)
            z = np.random.uniform(z_min_sio2, z_max_sio2)
            
            # Check if inside nanosheet (reject if so)
            if not ((x_min <= x <= x_max) and (y_min <= y <= y_max) and (z_min <= z <= z_max)):
                valid = True
                positions.append([x, y, z])
                symbols.append('Si')
    
    # Place O atoms near Si atoms
    for si_pos in positions:
        for _ in range(2):
            # Place O atoms within ~1.6 Angstrom of Si (typical Si-O bond length)
            angle = np.random.uniform(0, 2*np.pi)
            phi = np.random.uniform(0, np.pi)
            radius = np.random.uniform(1.4, 1.8)
            
            x = si_pos[0] + radius * np.sin(phi) * np.cos(angle)
            y = si_pos[1] + radius * np.sin(phi) * np.sin(angle)
            z = si_pos[2] + radius * np.cos(phi)
            
            positions.append([x, y, z])
            symbols.append('O')
    
    sio2_atoms = Atoms(symbols=symbols, positions=positions)
    
    return sio2_atoms


def create_core_shell_structure(xs=1.0, ys=3.0, zs=20.0, sio2_thickness=5.0):
    """
    Combine Si nanosheet and SiO2 coating into a single structure.
    
    Parameters:
    -----------
    xs, ys, zs : float
        Nanosheet dimensions in nm
    sio2_thickness : float
        SiO2 coating thickness in nm
    
    Returns:
    --------
    structure : ASE Atoms object
        Combined Si-SiO2 core-shell structure
    """
    
    print("Creating Si nanosheet...")
    si_nanosheet = create_si_nanosheet(xs, ys, zs)
    print(f"  Si atoms: {len(si_nanosheet)}")
    
    print(f"\nCreating SiO2 coating ({sio2_thickness} nm thick)...")
    sio2_coating = create_sio2_coating(si_nanosheet, sio2_thickness)
    print(f"  SiO2 atoms: {len(sio2_coating)}")
    
    # Combine structures
    structure = si_nanosheet + sio2_coating
    
    # Set proper cell dimensions with vacuum
    cell_size = 60  # Angstrom with plenty of vacuum
    structure.set_cell([cell_size, cell_size, cell_size], scale_atoms=False)
    structure.center()
    
    print(f"\nTotal atoms in structure: {len(structure)}")
    print(f"  Si: {structure.get_chemical_symbols().count('Si')}")
    print(f"  O: {structure.get_chemical_symbols().count('O')}")
    
    return structure


def setup_mace_calculator(model='mace_mp_0_small_no_pbc_std.model', device='cpu'):
    """
    Setup MACE calculator for energy and force calculations.
    
    Parameters:
    -----------
    model : str
        MACE model file path (e.g., 'mace_mp_0_small_no_pbc_std.model')
        Download from: https://github.com/ACEsuit/mace/releases
    device : str
        Computation device: 'cpu' or 'cuda'
    
    Returns:
    --------
    calculator : Calculator
        Configured MACE or fallback calculator
    """
    
    print(f"\nSetting up calculator (device={device})...")
    
    if not MACE_AVAILABLE:
        print("  MACE not available, using EMT for testing...")
        try:
            from ase.calculators.emt import EMT
            print("  EMT calculator ready (fast, test only)!")
            return EMT()
        except:
            print("  ERROR: Could not import EMT either!")
            return None
    
    try:
        # Try to load MACE model
        import os
        
        # Check if model file exists
        if not os.path.exists(model):
            print(f"\n  ERROR: Model file not found: {model}")
            print(f"\n  Download MACE models from:")
            print(f"  wget https://github.com/ACEsuit/mace/releases/download/mace_mp_0_small/mace_mp_0_small_no_pbc_std.model")
            print(f"\n  Or provide full path to model file.")
            
            # Try fallback to EMT
            try:
                from ase.calculators.emt import EMT
                print("\n  Falling back to EMT calculator for testing...")
                return EMT()
            except:
                return None
        
        calculator = MACECalculator(
            model_paths=model,
            device=device,
            default_dtype="float32"
        )
        print(f"  MACE calculator ready! (model: {model})")
        return calculator
    
    except Exception as e:
        print(f"\n  ERROR loading MACE: {e}")
        print(f"\n  To install MACE:")
        print(f"  pip install mace-torch torch")
        print(f"\n  Download model:")
        print(f"  wget https://github.com/ACEsuit/mace/releases/download/mace_mp_0_small/mace_mp_0_small_no_pbc_std.model")
        
        # Try fallback to EMT
        try:
            from ase.calculators.emt import EMT
            print("\n  Using EMT calculator for testing...")
            return EMT()
        except:
            return None


def run_md_simulation(structure, calculator, temperature=300, timestep=1.0, 
                      steps=1000, output_file='md_trajectory.traj'):
    """
    Run molecular dynamics simulation.
    
    Parameters:
    -----------
    structure : ASE Atoms object
        System to simulate
    calculator : ASE Calculator
        MACE or other calculator
    temperature : float
        Temperature in Kelvin
    timestep : float
        MD timestep in femtoseconds
    steps : int
        Number of MD steps
    output_file : str
        Output trajectory file
    
    Returns:
    --------
    None
    """
    
    if calculator is None:
        print("Cannot run MD without calculator!")
        return
    
    print(f"\nSetting up MD simulation...")
    print(f"  Temperature: {temperature} K")
    print(f"  Timestep: {timestep} fs")
    print(f"  Total steps: {steps}")
    print(f"  Total simulation time: {steps * timestep:.2f} fs")
    
    structure.set_calculator(calculator)
    
    # Initialize velocities from Maxwell-Boltzmann distribution
    MaxwellBoltzmannDistribution(structure, temperature_K=temperature)
    
    # Set up the dynamics
    dyn = VelocityVerlet(structure, timestep=timestep * 1.0, trajectory=output_file)
    
    # Add a simple print function to monitor temperature
    def print_status(dyn=dyn):
        ke = dyn.atoms.get_kinetic_energy()
        pe = dyn.atoms.get_potential_energy()
        temp = ke / (1.5 * len(dyn.atoms) * kB)
        print(f"Step {dyn.nsteps:5d}: E_kin={ke:10.4f} eV, E_pot={pe:10.4f} eV, T={temp:7.1f} K")
    
    dyn.attach(print_status, interval=50)
    
    # Run the simulation
    print(f"\nRunning MD simulation...")
    dyn.run(steps)
    
    print(f"\nMD simulation completed!")
    print(f"Trajectory saved to: {output_file}")
    
    return dyn


def analyze_trajectory(trajectory_file):
    """
    Basic analysis of MD trajectory.
    
    Parameters:
    -----------
    trajectory_file : str
        Path to trajectory file
    """
    
    print(f"\nAnalyzing trajectory: {trajectory_file}")
    
    # Read trajectory
    trajectory = read(trajectory_file, index=':')
    
    if not isinstance(trajectory, list):
        trajectory = [trajectory]
    
    energies = []
    temperatures = []
    
    print(f"Total frames: {len(trajectory)}")
    
    for i, atoms in enumerate(trajectory):
        if atoms.calc is not None:
            try:
                e = atoms.get_potential_energy()
                energies.append(e)
            except:
                pass
        
        ke = atoms.get_kinetic_energy()
        temp = ke / (1.5 * len(atoms) * kB)
        temperatures.append(temp)
    
    if energies:
        print(f"Energy range: {min(energies):.4f} - {max(energies):.4f} eV")
        print(f"Energy change: {energies[-1] - energies[0]:.4f} eV")
    
    print(f"Temperature range: {min(temperatures):.2f} - {max(temperatures):.2f} K")
    print(f"Average temperature: {np.mean(temperatures):.2f} K")
    
    # Plot results
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    if energies:
        axes[0].plot(energies)
        axes[0].set_xlabel('MD Step')
        axes[0].set_ylabel('Potential Energy (eV)')
        axes[0].set_title('Energy vs Time')
        axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(temperatures)
    axes[1].set_xlabel('MD Step')
    axes[1].set_ylabel('Temperature (K)')
    axes[1].set_title('Temperature vs Time')
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('md_analysis.png', dpi=150)
    print("Analysis plot saved to: md_analysis.png")
    plt.close()


def main():
    """Main execution function."""
    
    print("=" * 70)
    print("Si Nanosheet with SiO2 Coating - MD Simulation")
    print("=" * 70)
    
    # Parameters
    xs = 1.0      # nm
    ys = 2.0      # nm
    zs = 3.0     # nm
    sio2_thickness = 0.5  # nm
    
    temperature = 300  # K
    timestep = 1.0     # fs
    md_steps = 500     # For testing; use 10000+ for production
    
    # Create structure
    print("\n" + "-" * 70)
    print("STEP 1: Creating Core-Shell Structure")
    print("-" * 70)
    
    structure = create_core_shell_structure(xs, ys, zs, sio2_thickness)
    
    # Save initial structure
    write('initial_structure.xyzout', structure)
    write('initial_structure.cif', structure)
    print(f"\nInitial structure saved:")
    print(f"  initial_structure.xyz")
    print(f"  initial_structure.cif")
    
    # Setup calculator
    print("\n" + "-" * 70)
    print("STEP 2: Setting up MACE Calculator")
    print("-" * 70)
    
    # Try to use MACE calculator
    # If models are not available, you can:
    # 1. Download from: https://github.com/ACEsuit/mace/releases
    # 2. Use EMT for testing (automatic fallback)
    
    calculator = setup_mace_calculator(
        model='mace_mp_0_small_no_pbc_std.model',
        device='cpu'
    )
    
    if calculator is None:
        print("\nERROR: Could not load calculator!")
        return
    
    # Run MD simulation
    print("\n" + "-" * 70)
    print("STEP 3: Running MD Simulation")
    print("-" * 70)
    
    run_md_simulation(
        structure, 
        calculator,
        temperature=temperature,
        timestep=timestep,
        steps=md_steps,
        output_file='md_trajectory.traj'
    )
    
    # Analyze trajectory
    print("\n" + "-" * 70)
    print("STEP 4: Analyzing Results")
    print("-" * 70)
    
    analyze_trajectory('md_trajectory.traj')
    
    print("\n" + "=" * 70)
    print("Simulation Complete!")
    print("=" * 70)


if __name__ == '__main__':
    main()
