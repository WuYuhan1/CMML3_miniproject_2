import numpy as np

def make_segments(L):
    v1 = np.array([0, 1])
    v2 = np.array([1, 0])
    v3 = np.array([0, -1])
    v4 = np.array([0, 1])
    v5 = np.array([1, 0])
    v6 = np.array([0, -1])

    vessel1 = np.zeros((6, 2))

    for seg in range(5):
        vessel1[seg + 1, 0] = 0
        vessel1[seg + 1, 1] = np.sum(L[:seg + 1]) * 1e6

    vessel2 = np.zeros((11, 2))
    vessel2[0, :] = vessel1[5, :]

    for seg in range(6, 16):
        vessel2[seg - 5, 0] = np.sum(L[5:seg]) * 1e6
        vessel2[seg - 5, 1] = 50

    vessel3 = np.zeros((6, 2))  # Increased size to 6 rows
    vessel3[0, :] = vessel2[10, :]

    for seg in range(16, 21):  # Adjusted range
        vessel3[seg - 15, 0] = 100
        vessel3[seg - 15, 1] = 50 - np.sum(L[15:seg]) * 1e6

    vessel4 = np.zeros((6, 2))
    vessel4[0, :] = vessel1[5, :]

    for seg in range(21, 26):
        vessel4[seg - 20, 0] = 0
        vessel4[seg - 20, 1] = 50 + np.sum(L[20:seg]) * 1e6

    vessel5 = np.zeros((11, 2))
    vessel5[0, :] = vessel4[5, :]

    for seg in range(26, 36):
        vessel5[seg - 25, 0] = np.sum(L[25:seg]) * 1e6
        vessel5[seg - 25, 1] = 100

    vessel6 = np.zeros((6, 2))  # Increased size to 6 rows
    vessel6[0, :] = vessel5[10, :]

    for seg in range(36, 41):  # Adjusted range
        vessel6[seg - 35, 0] = 100
        vessel6[seg - 35, 1] = 100 - np.sum(L[35:seg]) * 1e6

    segments1 = np.vstack((vessel1, vessel2, vessel3))
    segments2 = np.vstack((vessel4, vessel5, vessel6))

    segments = np.hstack((segments1, segments2))

    return segments
