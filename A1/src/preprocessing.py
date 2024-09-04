import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os


class DataProcessing(): 
    """ 
    A class to handle data loading, preprocessing, and basic analysis for the Spotify dataset.
    """

    def __init__(self, filepath): 
        """
        Initializes the DataProcessing class with the provided filepath.

        ### Parameters:
        filepath : str \\
            The path to the dataset file.
        """

        self.filepath = filepath
        self.data = None

        # Load and process the data when the class is initialized 
        self.load_data()
        self.process_data()

# 1a) 
    def load_data(self):
        """ 
        Load the data from the given filepath 
        """

        try: 
            self.data = pd.read_csv(self.filepath, delimiter=",")
            print(f'filepath: {self.filepath} \n number of samples (rows-1): {len(self.data)-1} \n number of features (columns): {len(self.data.columns)}')
        except Exception as e:
            print(f"Failed to load data from {self.filepath}, {e}")

# 1b) 
    def process_data(self):
        """ 
        Process the data by classifying and filtering the features 
        """

        self.classify_data()
        self.filter_features()
# 1b)       
    def classify_data(self):
        """        
        Classifies the data into 'Pop' (1) and 'Classical' (0) genres.
        """
        
        self.data = self.data[self.data["genre"].isin(["Pop", "Classical"])]
       
        self.data.loc[self.data["genre"] == "Pop", "genre"] = 1
        self.data.loc[self.data["genre"] == "Classical", "genre"] = 0
        print(f'Pop: {len(self.data[self.data["genre"]==1])}\nClassical: {len(self.data[self.data["genre"]==0])}')

# 1c) 
    def split_data(self, test_size, random_state):
        """
        Splits the data into training and testing sets.

        ### Parameters:
        - test_size : float \\
            The proportion of the dataset to include in the test split.
        - random_state : int \\
            Controls the shuffling applied to the data before applying the split.

        ### Returns:
        - X_train, X_test, y_train, y_test. \\
        The training and testing data splits.
        """

        features = self.data[['liveness', 'loudness']]
        labels = self.data['genre']
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels)
        print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    
    def filter_features(self):
        """
        Filters the dataset to include only the relevant features.
        """
        
        self.data = self.data[['liveness', 'loudness', 'genre']]
        print("Filtered to necessary features: 'liveness', 'loudness', and 'genre'")

    def save_plot(self, plot_title, plot_directory="../plots"):
        """ 
        Save the plot in the plots directory with the name of the plot 

        ### Parameters:
        - plot_title : str \\
            The title of the plot, which will also be used as the filename.
        - plot_directory : str, optional \\
            The directory where the plot will be saved (default is "../plots").
        """ 
        
        plot_filename = f"{plot_title.replace(' ', '_').lower()}.png"

        # Create the directory if it doesn't exist 
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        plt.savefig(os.path.join(plot_directory, plot_filename))
        print(f"Plot saved to {plot_directory}/{plot_filename}")

    def plot_data(self, plot_title):
        """
        Plots the data on a 2D plane with 'liveness' on the x-axis and 'loudness' on the y-axis.

        ### Parameters:
        plot_title : str \\
            The title of the plot.
        """

        pop_data = self.data[self.data['genre'] == 1]
        classical_data = self.data[self.data['genre'] == 0]
        plt.scatter(pop_data['liveness'], pop_data['loudness'], color='purple', edgecolor='black', label='Pop', alpha=0.3, s=50)
        plt.scatter(classical_data['liveness'], classical_data['loudness'], color='pink', edgecolor='black', label='Classical', alpha=0.3, s=50)
        plt.title(plot_title, fontsize=15)
        plt.xlabel('Liveness', fontsize=12)
        plt.ylabel('Loudness', fontsize=12)
        plt.legend()
        plt.grid(True)  

        # Save the plot
        self.save_plot(plot_title)

        plt.show()




