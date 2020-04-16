import numpy as np
import pytest

from gol import gol


def test_gol_initial_state():
    init_state = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0]])
    g = gol.GameOfLife(init_state)
    assert np.allclose(g.get_state(), init_state)


def test_gol_raises_for_invalid_initial_state():
    init_state = np.array([
        [1, 0, 1.4],
        [0, 1, 0],
        [0, 10, 0]])
    with pytest.raises(ValueError):
        gol.GameOfLife(init_state)


args_number_of_live_cells = [(np.array([
                                        [1, 3, 1],
                                        [3, 3, 3],
                                        [2, 1, 2]]), False),
                             (np.array([
                                        [3, 4, 3],
                                        [4, 3, 4],
                                        [4, 3, 4]]), True)]


@pytest.mark.parametrize('expected, cyclic', args_number_of_live_cells)
def test_gol_number_of_live_cells(expected, cyclic):
    init_state = np.array([
        [1, 0, 1],
        [0, 1, 0],
        [0, 1, 0]])
    g = gol.GameOfLife(init_state, cyclic=cyclic)
    num_life_cells = g._get_live_neighbours_count()
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

        (np.array([[0, 0, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 1, 1],
                   [0, 0, 0, 1]]),
         np.array([[0, 0, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 0, 0],
                   [0, 0, 1, 1]]))
         ]


@pytest.mark.parametrize('init_state, expected', update_args)
def test_gol_update_state(init_state, expected):
    g = gol.GameOfLife(init_state, cyclic=False)
    g.update_state()
    assert np.allclose(g.get_state(), expected)


update_cyclic_args = [
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

        (np.array([[0, 0, 0, 0],
                   [0, 0, 1, 1],
                   [0, 0, 1, 1],
                   [0, 0, 0, 1]]),
         np.array([[0, 0, 1, 1],
                   [0, 0, 1, 1],
                   [1, 0, 0, 0],
                   [0, 0, 1, 1]]))
         ]


@pytest.mark.parametrize('init_state, expected', update_cyclic_args)
def test_gol_cyclic_update_state(init_state, expected):
    g= gol.GameOfLife(init_state, cyclic=True)
    g.update_state()
    assert np.allclose(g.get_state(), expected)
