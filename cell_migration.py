import numpy as np

def cell_migration(seg, seg_cells, new_seg_cells, migrate, Q, branch_rule=None, branch_alpha=None, tau=None):
    """Handle cellular migration in the agent-based model."""

    cell_size = 10e-6  # Set the size of each cell (m)
    mchance = 0.01  #migration probability

    #the upstream and downstream sgements of every segment
    migration_map = {
        0: (19, 1),   
        1: (0, 2),       
        2: (1, 3),       
        3: (2, 4),       
        4: (3, 5),       
        5: (4, 6),     
        6: (5, 7),       
        7: (6, 8),       
        8: (7, 9),       
        9: (8, 10),      
        10: (9, 11),     
        11: (10, 12),    
        12: (11, 13),    
        13: (12, 14),   
        14: (13, 15),    
        15: (14, 16),    
        16: (15, 17),    
        17: (16, 18),    
        18: (17, 19),    
        19: (18, 0),  
        20: (5, 21),     
        21: (20, 22),    
        22: (21, 23),    
        23: (22, 24),    
        24: (23, 25),    
        25: (24, 26),    
        26: (25, 27),    
        27: (26, 28),    
        28: (27, 29),    
        29: (28, 30),    
        30: (29, 31),    
        31: (30, 32),    
        32: (31, 33),    
        33: (32, 34),   
        34: (33, 35),    
        35: (34, 36),    
        36: (35, 37),    
        37: (36, 38),    
        38: (37, 39),    
        39: (38, 15),    
    }

    # Check if the segment contains cells
    if seg_cells[seg]['num'] != 0:
        migrate_cells = []
        node15_target_segs = {}#for branching rules
        for cell in range(seg_cells[seg]['num']):
            mcell = np.random.rand()
            
            if mcell <= mchance:  # Determine if cell migrates
                polar_vect = seg_cells[seg]['polarity'][cell]
                migrate_vect = cell_size * polar_vect
                migrate_dir = 0  # initialize the migrating direction
                
                # Migration logic based on segment index
                if seg != 15:  # in segments except Reunion Node 15
                    if seg <= 4 or (20 <= seg <= 24) or (15 <= seg <= 19) or (35 <= seg <= 39):  # vertical segments
                        if migrate_vect[1] >= cell_size / 2:
                            migrate_dir = -1
                        elif migrate_vect[1] <= -cell_size / 2:
                            migrate_dir = 1
                    elif (5 <= seg <= 14) or (25 <= seg <= 34):  # horizontal segments
                        if migrate_vect[0] >= cell_size / 2:
                            migrate_dir = -1
                        elif migrate_vect[0] <= -cell_size / 2:
                            migrate_dir = 1
                
                # Reunion Node 15 (branching rules)
                else:
                    if branch_rule == 1:  #BR1: choose the branch with larger shear stress
                        if tau[14] > tau[39]:
                            target_seg_branch = 14  
                        else:
                            target_seg_branch = 39
                        node15_target_segs[cell] = target_seg_branch
                        migrate_cells.append((cell, -99)) #mark the cells 
                        migrate[seg] += 1
                    if branch_rule == 4: #BR4: set the probability of cells migrating to which branch artifically as P(proximal branch)=0.7, P(distal branch)=0.3
                        p=0.7
                        
                        if np.random.rand()<=p:
                            target_seg_branch = 14
                        else:
                            target_seg_branch = 39
                        node15_target_segs[cell] = target_seg_branch
                        migrate_cells.append((cell, -99)) #mark the cells 
                        migrate[seg] += 1
                    if branch_rule == 5:  #BR5: a weighted average of two probability components: one due to shear stress, and one due to cell number
                        p_tau=tau[14]/(tau[14]+tau[39])
                        p_num=seg_cells[14]['num']/(seg_cells[14]['num']+seg_cells[39]['num'])
                        p=branch_alpha*p_tau+(1-branch_alpha)*p_num
                        if np.random.rand()<=p:
                            target_seg_branch = 14
                        else:
                            target_seg_branch = 39
                        node15_target_segs[cell] = target_seg_branch
                        migrate_cells.append((cell, -99)) #mark the cells 
                        migrate[seg] += 1


                if migrate_dir != 0:
                    migrate_cells.append((cell, migrate_dir))
                    migrate[seg] += 1

        #delete the cells in current segments and add cells to target segments
        for cell_idx, migrate_dir in sorted(migrate_cells, reverse=True):
            #normal segments
            if migrate_dir != -99:
                target_seg = migration_map[seg][0] if migrate_dir == 1 else migration_map[seg][1]
            #Node15
            else:
                target_seg = node15_target_segs.get(cell_idx, None)
                if target_seg is None or target_seg >= 40: 
                    continue
            
            #delete cells
            #delete polarity
            del new_seg_cells[seg]['polarity'][cell_idx]
            #delete migration marker
            new_seg_cells[seg]['migration'] = np.delete(new_seg_cells[seg]['migration'], cell_idx)
            #cell number -1
            new_seg_cells[seg]['num'] -= 1

            #add cells
            #copy polarity
            polar_vect = seg_cells[seg]['polarity'][cell_idx].copy()
            polar_vect /= np.linalg.norm(polar_vect)
            new_seg_cells[target_seg]['polarity'].append(polar_vect)
            #initialize migration marker
            new_seg_cells[target_seg]['migration'] = np.append(new_seg_cells[target_seg]['migration'], 0)
            #cell number +1
            new_seg_cells[target_seg]['num'] += 1

    return seg_cells, new_seg_cells