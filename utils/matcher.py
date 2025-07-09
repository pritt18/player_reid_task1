# utils/matcher.py
from scipy.spatial.distance import cdist
import numpy as np

def match_players(boxes_broadcast, boxes_tacticam):
    """
    Match players detected in broadcast and tacticam views using IoU or center-point distance.
    Returns a dictionary mapping tacticam index to broadcast index (ID).
    """
    centers_b = np.array([[((box[0] + box[2]) / 2), ((box[1] + box[3]) / 2)] for box in boxes_broadcast])
    centers_t = np.array([[((box[0] + box[2]) / 2), ((box[1] + box[3]) / 2)] for box in boxes_tacticam])

    if len(centers_b) == 0 or len(centers_t) == 0:
        return {}

    dist_matrix = cdist(centers_t, centers_b)  # tacticam to broadcast

    matched = {}
    for i, row in enumerate(dist_matrix):
        min_idx = np.argmin(row)
        matched[i] = min_idx

    return matched
