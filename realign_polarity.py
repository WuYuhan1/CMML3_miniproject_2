import numpy as np

def realign_polarity(seg, Q, seg_cells, new_seg_cells, w1, w2, w3, w4):
    """Realign the polarity vectors of all cells in a segment based on weight factors."""
    
    if seg_cells[seg]['num'] != 0:
        for cell in range(seg_cells[seg]['num']):
            
            # Persistence alignment component
            polar_vect = seg_cells[seg]['polarity'][cell]
            
            # Guarantees that flow_vect always has a value, even if seg falls outside the expected ranges
            flow_vect = np.array([0, 0])  # Default to zero vector
            
            # Flow alignment component
            #the cells should migrate against the direction of blood flow
            #change the segment index
            if 0 <= seg <= 4 or 20 <= seg <= 24:
                flow_vect = np.array([0, 1]) * (-np.sign(Q[seg]))
            elif 5 <= seg <= 14 or 25 <= seg <= 34:
                flow_vect = np.array([1, 0]) * (-np.sign(Q[seg]))
            elif 15 <= seg <= 19 or 35 <= seg <= 39:
                flow_vect = np.array([0, -1]) * (-np.sign(Q[seg]))
            
            # Random walk alignment component
            rand_walk_vect = np.random.randn(2)
            rand_walk_vect /= np.linalg.norm(rand_walk_vect)
            
            # Calculate alignment angles
            phi2 = np.arccos(np.clip(np.dot(flow_vect, polar_vect), -1, 1))
            phi4 = np.arccos(np.clip(np.dot(rand_walk_vect, polar_vect), -1, 1))
            
            # Determine rotation direction
            theta = w1 * 0 + w2 * phi2 + w4 * phi4
            rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
            new_polar_vect = np.dot(rotation_matrix, polar_vect)
            new_polar_vect /= np.linalg.norm(new_polar_vect)
            
            # Assign the updated polarity vector
            seg_cells[seg]['polarity'][cell] = new_polar_vect
            new_seg_cells[seg]['polarity'][cell] = new_polar_vect
    
    return seg_cells, new_seg_cells
