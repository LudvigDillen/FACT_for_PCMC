class PC:
    def __init__(self, pc, distances):
        self.pc = pc
        self.distances_to_origin = distances
        self.N_dim = pc.shape[0]
        self.N_points = pc.shape[1]
