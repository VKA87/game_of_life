import numpy as np

from gol.gol import GameOfLife

def test_gol_inial_sate():
    init_state = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0]])

    gol = GameOfLife(init_state)
    assert np.allclose(gol.get_state(), init_state)


def test_gol_update_state():
    init_state = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0]])

    gol = GameOfLife(init_state)
    num_life_cells = gol._get_number_of_life_cells()
    expected = np.array([
        [3, 4, 3],
        [4, 3, 4],
        [4, 3, 4]])
    assert np.allclose(num_life_cells, expected)
