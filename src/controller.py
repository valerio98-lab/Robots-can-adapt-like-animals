import numpy as np


class CPG:
    def __init__(self, params):
        self.A, self.phi = params[:8], params[8:]
        self.freq = 1.5
        self.t = 0.0

    def select_action(self, _state, dt=1 / 60):
        self.t += dt
        return self.A * np.sin(2 * np.pi * self.freq * self.t + self.phi)
