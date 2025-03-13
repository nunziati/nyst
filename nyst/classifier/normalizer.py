class SignalNormalizer:
    r"""Class for normalizing signals."""
    
    def __init__(self, std):
        r"""Initialize the normalizer."""
        self.std = std

    def normalize(self, signal):
        r"""Normalize the signal."""
        raise NotImplementedError