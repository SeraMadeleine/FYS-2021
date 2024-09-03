import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt


class DataProcessing(): 
    def __init__(self, filepath): 
        self.filepath = filepath
        self.data = None


    def load_data(self):
        self.data = pd.read_csv(self.filepath, delimiter=",")
        print(f'filepath: {self.filepath} \n number of rows: {len(self.data)} \n number of columns: {len(self.data.columns)}')
        

if __name__ == "__main__":
    preprocessing = DataProcessing("../data/SpotifyFeatures.csv")
    preprocessing.load_data()