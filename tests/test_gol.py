import numpy as np
import pytest

from gol.gol import GameOfLife


def test_gol_initial_state():
    init_state = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0]])
    gol = GameOfLife(init_state)
    assert np.allclose(gol.get_state(), init_state)


def test_gol_raises_for_invalid_initial_state():
    init_state = np.array([
        [1, 0, 1.4],
        [0, 1, 0],
        [0, 10, 0]])
    with pytest.raises(ValueError):
        GameOfLife(init_state)


def test_gol_number_of_live_cells():
    init_state = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0]])

    gol = GameOfLife(init_state)
    num_life_cells = gol._get_live_neighbours_count()
    expected = np.array([
        [3, 4, 3],
        [4, 3, 4],
        [4, 3, 4]])
    assert np.allclose(num_life_cells, expected)


update_args = [
        (np.array([[0, 0, 0, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]),
         np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])),

        (np.array([[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]]),
         np.array([[0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0],
                   [0, 0, 0, 0]])),

        (np.array([[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 0]]),
         np.array([[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0]])),

        (np.array([[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 0, 1, 0],
                   [0, 0, 0, 0]]),
         np.array([[0, 0, 0, 0],
                   [0, 1, 1, 0],
                   [0, 1, 1, 0],
                   [0, 0, 0, 0]])),

        (np.array([[0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0],
                   [0, 0, 1, 1, 0],
                   [0, 0, 0, 1, 0],
                   [0, 0, 0, 0, 0]]),
         np.array([[0, 0, 0, 0, 0],
                   [0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 1],
                   [0, 0, 1, 1, 0],
                   [0, 0, 0, 0, 0]]))
         ]


@pytest.mark.parametrize('init_state, expected', update_args)
def test_gol_update_state(init_state, expected):
    gol = GameOfLife(init_state)
    gol.update_state()
    assert np.allclose(gol.get_state(), expected)
