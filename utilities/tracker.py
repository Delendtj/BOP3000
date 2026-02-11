import numpy as np
from pathlib import Path
from boxmot import BoostTrack
from sympy.physics.units import length


def tracking(input, frame):
    print("TYPE OF INPUT: ", type(input))
    print(len(input.shape))

    input = input.reshape(input.shape[0], -1)
    print(input[0,:])
    # Half kan settes til True for raskere men mindre accurate inference.
    tracker = BoostTrack(reid_weights=Path('osnet_x0_25_msmt17.pt'), device='cpu', half=False)

    tracker.update(input, frame)
    tracker.plot_results(frame, show_trajectories=True)

    print(tracker)

    return tracker # input with ids

