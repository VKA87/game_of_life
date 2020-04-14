import numpy as np

class GameOfLife(object):

    def __init__(self, init_state):
        self.state = init_state

    def get_state(self):
        return self.state

    def update_state(self):
        self.state = self.get_next_state()

    def get_next_state(self):
        next_state = np.zeros(self.state.shape).astype(int)
        live_neighbours = self._get_live_neighbours_count()
        ix_1 = ((live_neighbours == 2) | (live_neighbours == 3)) & (self.state == 1)
        ix_2 = (live_neighbours == 3) & (self.state == 0)
        next_state[ix_1 | ix_2] = 1
        return next_state

    def _get_live_neighbours_count(self):
        n, m = self.state.shape
        res = [[self._cell_live_neighbours_count(i, j) for j in range(m)] for i in range(n)]
        return np.array(res)

    def _cell_live_neighbours_count(self, i, j):
        n, m = self.state.shape
        values = 0
        for a in range(i -1, i+2):
            for b in range(j-1, j+2):
                values += self.state[a%n, b%m]
        return values - self.state[i, j]
