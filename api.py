import numpy as np

from gol.animation import create_animation


if __name__ =='__main__':
    init_state = np.zeros((100, 100))
    init_state[51, 49:51] = 1
    init_state[50, 48:50] = 1
    init_state[49, 49] = 1
    anim = create_animation(init_state, speed=100)
