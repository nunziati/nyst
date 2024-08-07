import numpy as np

class FirstSpeedExtractor:
    def __init__(self):
        # Initialize the class with predefined time resolutions
        self.time_resolutions = [3, 5, 7, 9] # number of frame window resolution

    def apply(self, positions, fps=None):
        # Create an empty dictionary to store speed calculations
        speed_dict = {}

        # Iterate over each time resolution
        for time_resolution in self.time_resolutions:
            # Compute speed for the given position and time resolution
            speed_dict[time_resolution] = self.compute_speed(positions, time_resolution)

            # If fps is provided, adjust the speed by multiplying with fps
            if fps is not None:
                speed_dict[time_resolution] *= fps # units of measurement: position/seconds 

        return speed_dict
    
    def compute_speed(self, positions, time_resolution):
        # Calculate augmentation factor as half of the time resolution
        aug_factor = time_resolution // 2

        # Create arrays for augmenting the start and end of the positions array
        head = np.array([positions[0] for _ in range(aug_factor)])
        tail = np.array([positions[-1] for _ in range(aug_factor)])

        # Augment the position array by adding head and tail
        position_aug = np.vstack((head, positions, tail))

        # Initialize speed array with zeros
        speed = np.zeros(len(positions))

        # Compute the speed array
        for i in range(len(positions)):
            pos_start = position_aug[i]
            pos_end = position_aug[(i + time_resolution) - 1]

            # If either position is None, set speed to 0
            if pos_start is None or pos_end is None:
                speed[i] = 0
            else: # Compute the speed array
                speed[i] = (pos_end - pos_start) / time_resolution

        return speed

    '''def compute_speed(self, positions, time_resolution):

        # Calculate augmentation factor as half of the time resolution
        aug_factor = time_resolution // 2

        # Create arrays for augmenting the start and end of the positions array
        head = np.array([positions[0] for _ in range(aug_factor)])
        tail = np.array([positions[-1] for _ in range(aug_factor)])

        # Augment the position array by adding head and tail
        position_aug = np.vstack((head, positions, tail))

        # Compute the speed array. Calculation of the speed array between two positions (first and last) in the scrolling time resolution window.
        speed = (position_aug[aug_factor:] - position_aug[:-aug_factor]) / time_resolution # units of measurement: position/frame

        return speed'''