import numpy as np
import matplotlib.pyplot as plt
from solve_for_flow import solve_for_flow
from cell_migration import cell_migration
from realign_polarity import realign_polarity
from plot_network import plot_network
from make_segments import make_segments
import copy #the module for deepcopy

# Set random seed for reproducibility
np.random.seed(123456789)

# Input parameters
Nt =  100  # Number of time steps
Pin = 4 * 98  # Inlet pressure (Pa)
Pout = 1 * 98  # Outlet pressure (Pa)

mu = 3.5e-3  # Dynamic viscosity of blood (Pa-s)
Nn = 40  # Number of nodes
Nseg = 40  # Number of segments
num_cell = 10  # Initial number of cells per segment
cell_size = 5e-6  # Size of each cell (m)

branch_rule = 5 # Branching rules: 1, 4, 5
branch_alpha = 1.0  # Branching parameter

# Polarization re-alignment weights
w2 = 1  # Flow component weight
w3 = 0.00  # Neighbor re-alignment weight
w4 = 0.00  # Random re-alignment weight
w1 = 1 - w2 - w3 - w4  # Persistence component

# Initialize segment properties
L = np.ones(Nseg) * 10e-6  # Segment lengths (m)
Ncell = np.ones(Nseg) * num_cell  # Segment cell number array
D = np.zeros(Nseg)  # Segment diameters (m)
G = np.zeros(Nseg)  # Segment conductance array (m^4/Pa-s-m)
H = np.zeros(Nseg)  # Shear stress calculation factor

tau = np.zeros(Nseg)  # Shear stress array
segments = make_segments(L)  # Generate segment structure

# Initialize segment cell structures
def initialize_segments(Nseg, num_cell):
    seg_cells = [{} for _ in range(Nseg)]
    for seg in range(Nseg):
        seg_cells[seg]['num'] = int(num_cell)  # Number of cells
        seg_cells[seg]['polarity'] = [np.random.randn(2) for _ in range(num_cell)]  # Random polarity vectors
        for v in seg_cells[seg]['polarity']:
            v /= np.linalg.norm(v)  # Normalize to unit vectors
        seg_cells[seg]['migration'] = np.zeros(num_cell)  # Migration indicator
    return seg_cells

seg_cells = initialize_segments(Nseg, num_cell)

# Compute initial segment conductance and shear stress
def compute_conductance(Nseg, Ncell, cell_size, mu, L):
    D = np.zeros(Nseg)
    G = np.zeros(Nseg)
    H = np.zeros(Nseg)
    min_D = 1e-7 # Minimum diameter 
    for seg in range(Nseg):
        if Ncell[seg] >= 1:
            D[seg] = Ncell[seg] * cell_size / np.pi
            if D[seg] < min_D:
                D[seg] = min_D
        
        else:
            D[seg] = min_D
        G[seg] = (np.pi * D[seg]**4) / (128 * mu * L[seg])
        if D[seg] != 0:
            H[seg] = (32 * mu) / (np.pi * D[seg]**3)
    return D, G, H

D, G, H = compute_conductance(Nseg, Ncell, cell_size, mu, L)

# Solve for initial flow
P, Q, tau = solve_for_flow(G, Pin, Pout, H)

plot_network(segments, D, P, Q, seg_cells, tau)

# Time stepping for migration process
for t in range(Nt):
    print(f'Time step {t+1}/{Nt}')
    
    migrate = np.zeros(Nseg)
    new_seg_cells = copy.deepcopy(seg_cells)#use deepcopy to not change the original seg_cells
    
    for seg in range(Nseg):
        seg_cells, new_seg_cells = realign_polarity(seg, Q, seg_cells, new_seg_cells, w1, w2, w3, w4)
        seg_cells, new_seg_cells = cell_migration(seg, seg_cells, new_seg_cells, migrate, Q, branch_rule, branch_alpha, tau)
    
    seg_cells = new_seg_cells
    
    for seg in range(Nseg):
        Ncell[seg] = seg_cells[seg]['num']#update the segment cell number

    # Update conductance and shear stress
    D, G, H = compute_conductance(Nseg, Ncell, cell_size, mu, L)
    
    # Solve for flow with updated parameters
    P, Q, tau = solve_for_flow(G, Pin, Pout, H)
    
    # Plot only every 20 time steps
    if (t + 1) % 20 == 0:
        plot_network(segments, D, P, Q, seg_cells, tau)
