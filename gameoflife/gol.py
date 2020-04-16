import numpy as np

class GameOfLife(object):

    def __init__(self, init_state, cyclic=False):
        self._check_init_state(init_state)
        self.state = init_state.astype(int)
        self.cyclic = cyclic

    def _check_init_state(self, init_state):
        init_state = init_state.astype(float)
        ix = (init_state == 0.) | (init_state == 1.)
        if not ix.all():
            raise ValueError("initial state must be boolean or contains only 0 and 1")

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
        if self.cyclic:
            num_live_neighbours = self._cell_live_neighbours_count_cyclic(i, j)
        else:
            num_live_neighbours = self._cell_live_neighbours_count_not_cyclic(i, j)
        return num_live_neighbours

    def _cell_live_neighbours_count_cyclic(self, i, j):
        n, m = self.state.shape
        values = 0
        for a in range(i-1, i+2):
            for b in range(j-1, j+2):
                values += self.state[a%n, b%m]
        return values - self.state[i, j]

    def _cell_live_neighbours_count_not_cyclic(self, i, j):
        n, m = self.state.shape
        i_min = max(i-1, 0)
        i_max = min(i+1, n) + 1
        j_min = max(j-1, 0)
        j_max = min(j+1, m) + 1
        return np.sum(self.state[i_min:i_max, j_min:j_max]) - self.state[i, j]
