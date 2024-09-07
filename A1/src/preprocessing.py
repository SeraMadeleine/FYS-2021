import pandas as pd
from sklearn.model_selection import train_test_split


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

    def filter_features(self):
        """
        Filters the dataset to include only the relevant features.
        """
        
        self.data = self.data[['liveness', 'loudness', 'genre']]
        print("Filtered to necessary features: 'liveness', 'loudness', and 'genre'")

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

        # Split the data into training and testing sets, and shuffle the data
        X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state, stratify=labels, shuffle=True)
        print(f"Training set size: {len(X_train)}, Testing set size: {len(X_test)}")
        return X_train, X_test, y_train, y_test
    








