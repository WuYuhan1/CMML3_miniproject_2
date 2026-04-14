import numpy as np
import matplotlib.pyplot as plt

def plot_network(segments, D, P, Q, seg_cells, tau=None):  # Add tau=None
    """Plot the vessel network along with pressure, flow, and cell polarity vectors."""
    
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.title('Pressure, Flow, Diameter of Network')
    
    for seg in range(len(segments) - 1):
        if Q[seg] > 0:
            color = "red"
        else:
            color = "blue"
        plt.plot([segments[seg, 0], segments[seg + 1, 0]], 
                 [segments[seg, 1], segments[seg + 1, 1]], 
                 color=color, linewidth=D[seg] * 1e6 / 2)
    
    plt.grid()
    
    # Polarity distribution plot
    plt.subplot(1, 2, 2)
    plt.title('Distribution of Cell Polarity')
    plt.axis([-1, 1, -1, 1])
    plt.grid()
    
    print("Segment coordinates:\n", segments)
    for seg in range(len(seg_cells)):
        for cell in range(seg_cells[seg]['num']):
            polarity = seg_cells[seg]['polarity'][cell]
            plt.plot([0, polarity[0]], [0, polarity[1]], 'b-')
    
    plt.show()
