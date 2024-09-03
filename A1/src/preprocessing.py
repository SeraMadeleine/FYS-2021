import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


class DataProcessing(): 
    def __init__(self, filepath): 
        self.filepath = filepath
        self.data = None


    def load_data(self):
        self.data = pd.read_csv(self.filepath, delimiter=",")
        print(f'filepath: {self.filepath} \n number of rows: {len(self.data)} \n number of columns: {len(self.data.columns)}')
        
    def classify_data(self):
        self.data = self.data[self.data["genre"].isin(["Pop", "Classical"])]
       
        self.data.loc[self.data["genre"] == "Pop", "genre"] = 1
        self.data.loc[self.data["genre"] == "Classical", "genre"] = 0
        print(f'Pop: {len(self.data[self.data["genre"]==1])}\nClassical: {len(self.data[self.data["genre"]==0])}')


    def split_data(self, test_size, random_state):
        features = self.data[['liveness', 'loudness']]
        labels = self.data['genre']
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)
        print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def plot_data(self):
        pop_data = self.data[self.data['genre'] == 1]
        classical_data = self.data[self.data['genre'] == 0]
        plt.scatter(pop_data['liveness'], pop_data['loudness'], color='purple', edgecolor='black', label='Pop', alpha=0.7, s=50)
        plt.scatter(classical_data['liveness'], classical_data['loudness'], color='pink', edgecolor='black', label='Classical', alpha=0.7, s=50)
        plt.title('Liveness vs Loudness by Genre', fontsize=15)
        plt.xlabel('Liveness', fontsize=12)
        plt.ylabel('Loudness', fontsize=12)
        plt.legend()
        plt.grid(True)  # Adding a grid
        plt.show()




if __name__ == "__main__":
    filepath = "../data/SpotifyFeatures.csv"

    preprocessing = DataProcessing(filepath)

    # task 1a 
    preprocessing.load_data()

    # task 1b
    preprocessing.classify_data()

    X_train, X_test, y_train, y_test = preprocessing.split_data(0.2, 42)

    preprocessing.plot_data()