import numpy as np

def solve_for_flow(G, Pin, Pout, H=None):
    """Solve for flow in the bifurcating vessel network."""
    Nn = 40  # Number of nodes
    Nseg = 40  # Number of segments

    # Set very small values for zero conductance to avoid singular matrix errors
    G[G == 0] = 1e-25
    
    # Initialize matrices
    P = np.zeros(Nn)  # Nodal pressure array (Pa)
    Q = np.zeros(Nseg)  # Segment flow array (m^3/s)
    C = np.zeros((Nn, Nn))  # Conductance matrix
    B = np.zeros(Nn)  # Solution vector

    # Set boundary conditions
    C[0, 0] = G[0] * 1
    B[0] = G[0] * Pin
    
    # Set equations for internal nodes
    for seg in range(1, Nn - 1):
        C[seg, seg - 1] = -G[seg - 1]
        C[seg, seg] = G[seg - 1] + G[seg]
        C[seg, seg + 1] = -G[seg]
    
    # Set equation for last node
    C[Nn - 1, Nn - 1] = G[Nn - 2] * 1
    B[Nn - 1] = G[Nn - 2] * Pout

    # Solve for pressure
    P = np.linalg.solve(C, B)
    
    # Compute segment flow rates
    for seg in range(Nseg):
        if seg < Nseg - 1:  # Prevent out-of-bounds access
            Q[seg] = -G[seg] * (P[seg + 1] - P[seg])
    
    # Compute shear stress if H is provided
    if H is not None:
        tau = H * Q
        return P, Q, tau
    else:
        return P, Q
