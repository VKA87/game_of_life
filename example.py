import numpy as np

from gol.animation import create_animation
from gol.gol import GameOfLife


def get_init_state():
    init_state = np.zeros((100, 100)).astype(int)
    init_state[51, 49:51] = 1
    init_state[50, 48:50] = 1
    init_state[49, 49] = 1
    return init_state

if __name__ =='__main__':
    init_state = get_init_state()

    # cyclic example
    g = GameOfLife(init_state, cyclic=True)
    anim = create_animation(g, interval=200, n_frames=1000)
    anim.save('gol_cyclic.mp4')

    # non-cyclic example
    g = GameOfLife(init_state, cyclic=False)
    anim = create_animation(g, interval=200, n_frames=1000)
    anim.save('gol_non_cyclic.mp4')
