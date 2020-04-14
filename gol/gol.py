import numpy as np

class GameOfLife(object):

    def __init__(self, init_state):
        self.state = init_state

    def update_state(self):
        self.state = self.get_next_state()

    def get_next_state(self):
        next_state = np.zeros(self.state.shape)
        live_neighbours = self._get_number_of_neighbouring_live_cells()
        ix_1 = ((live_neighbours == 2) | (live_neighbours == 3)) & (self.state == 1)
        ix_2 = (live_neighbours == 3) & (self.state == 0)
        next_state[ix_1 | ix_2] = 1
        return next_state

    def get_state(self):
        return self.state

    def _get_number_of_neighbouring_live_cells(self):
        number_of_life_cells = np.zeros(self.state.shape)
        n, m = self.state.shape
        for i in range(n):
            for j in range(m):
                neighbours = self._get_neighbouring_cell_values(i, j)
                number_of_life_cells[i, j] = np.sum(neighbours) - self.state[i][j]
        return number_of_life_cells

    def _get_neighbouring_cell_values(self, i, j):
        n, m = self.state.shape
        return [self.state[a%n, b%m] for a in range(i-1, i+2) for b in range(j-1, j+2)]
