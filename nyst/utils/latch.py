# Class for storing and getting ROI values

class FirstLatch():
    def __init__(self):
        self.value = None

    def set(self, value): # Set ROI value
        self.value = value

    def get(self): # Get saved value
        return self.value