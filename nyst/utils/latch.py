# Class for storing and getting ROI values
class FirstLatch:
    '''
    Class that acts as a latch to store and retrieve a single value, such as an ROI (Region of Interest) value.

    Attributes:
    - value: The stored value, initialized as None.
    '''

    def __init__(self):
        self.value = None

    def set(self, value):
        """
        Sets the latch with a specified value.

        Arguments:
        - value: The value to be stored in the latch (e.g., an ROI).
        """
        self.value = value

    def get(self):
        """
        Retrieves the currently stored value from the latch.

        Returns:
        - The stored value, or None if no value has been set.
        """
        return self.value
