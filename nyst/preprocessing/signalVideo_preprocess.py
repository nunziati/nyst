import numpy as np

class PreprocessingSignalsVideos:
    def __init__(self):
        pass
    
    # Convert None values in the array to np.nan
    def convert_none_to_nan(self, array):
        '''
        Converts None values in the array to np.nan.

        Args:
        - array (np.array): A 2D numpy array that may contain None values.

        Returns:
        - np.array: The input array with None values replaced by np.nan.
        '''
        array = np.array(array, dtype=object)  # Ensure array is of object type
        array = np.where(array == None, np.nan, array)
        return array.astype(float)

    # Interpolate NaN values in a 2D array of positions by replacing them with the average of the previous and next valid values
    def interpolate_nans(self, positions:np.array):
        '''
        Interpolates NaN values in a 2D array of positions by replacing them with the average of the previous and next valid values.

        Args:
        - positions (np.array): A 2D numpy array where each element is an array [x, y], with possible NaN values.

        Returns:
        - np.array: The input array with NaN values replaced by interpolated values.
        '''
        
        # Convert None to np.nan
        positions = self.convert_none_to_nan(positions)
        # Elements count
        n = len(positions)
       
        # Replace initial NaN positions with the first valid value
        if np.isnan(positions[0]).any():
            first_valid_index = np.where(~np.isnan(positions).any(axis=1))[0][0]
            positions[0] = positions[first_valid_index]

        # Replace final NaN positions with the last valid value
        if np.isnan(positions[-1]).any():
            last_valid_index = np.where(~np.isnan(positions).any(axis=1))[0][-1]
            positions[-1] = positions[last_valid_index]
        
        # Flag initialization and counter 
        flag = True
        i = 0

        # Loop through the array to identify and interpolate NaNs
        while flag:
            if np.isnan(positions[i]).any():
                start = i-1 # Start index of interpolation
                while i < n and np.isnan(positions[i]).any():
                    i += 1
                end = i # End index of interpolation
                
                # Interpolate between the previous valid value and the next valid value of the sequence NaN
                prev_val = positions[start]
                next_val = positions[end]
                interpolated_val = (prev_val + next_val) / 2
                
                # Sostituzione dei NaN con il valore interpolato
                positions[(start+1):end] = np.rint(interpolated_val).astype(int)
            else:
                # Flag check
                if i == n-1:
                    flag = False
                
                i += 1
        # Ensure the entire array is of integer type
        positions = positions.astype(int)
                
        return positions
