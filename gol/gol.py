import numpy as np

class GameOfLife(object):

    def __init__(self, init_state):
        self.state = init_state

    def update_state(self):
        neighbours = self._get_number_of_life_cells()
        new_state = np.zeros(self.state.shape)
        ix_1 = ((neighbours == 2) | (neighbours == 3)) & (self.state == 1)
        ix_2 = (neighbours == 3) & (self.state == 0)
        new_state[ix_1 | ix_2] = 1
        self.state = new_state

    def get_state(self):
        return self.state

    def _get_number_of_life_cells(self):
        number_of_life_cells = np.zeros(self.state.shape)
        n, m = self.state.shape
        for i in range(n):
            for j in range(m):
                neighbours = self._get_neighbour(i, j)
                number_of_life_cells[i, j] = np.sum(neighbours) - self.state[i][j]
        return number_of_life_cells

    def _get_neighbour(self, i, j):
        n, m = self.state.shape
        return [self.state[a%n, b%m] for a in range(i-1, i+2) for b in range(j-1, j+2)]
