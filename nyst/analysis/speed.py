import numpy as np

class FirstSpeedExtractor:
    def __init__(self):
        self.time_resolutions = [3, 5, 7, 9]

    def apply(self, position, fps=None):
        speed_dict = {}

        for time_resolution in self.time_resolutions:
            speed_dict[time_resolution] = self.compute_speed(position, time_resolution)
            if fps is not None:
                speed_dict[time_resolution] *= fps

        return speed_dict
    
    def compute_speed(self, position, time_resolution):
        aug_factor = time_resolution // 2
        head = np.array([position[0] for _ in range(aug_factor)])
        tail = np.array([position[-1] for _ in range(aug_factor)])
        position_aug = np.vstack((head, position, tail))

        speed = (position_aug[aug_factor:] - position_aug[:-aug_factor]) / time_resolution

        return speed