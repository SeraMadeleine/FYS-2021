import numpy as np

class DataLoader:
    def __init__(self, data_path, delimiter=','):
        self.data = np.loadtxt(data_path, delimiter=delimiter)
        self.x = self.data[0]  # Features
        self.y = self.data[1]  # Labels (0 or 1)
        self.c0 = self.x[np.where(self.y == 0)]  # Extract x values where y = 0
        self.c1 = self.x[np.where(self.y == 1)]  # Extract x values where y = 1

    def print_data_info(self):
        print('Number of samples in the dataset:', len(self.x))
        print('Number of samples in class 0:', len(self.c0))
        print('Number of samples in class 1:', len(self.c1))
