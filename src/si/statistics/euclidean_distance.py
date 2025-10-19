def euclidian_distance(x,y):
    import numpy as np
    return np.sqrt((x-y)**2).sum(axis=1)