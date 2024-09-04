from preprocessing import DataProcessing 
from machineLearning import Perception
import numpy as np

if __name__ == "__main__":
    filepath = "../data/SpotifyFeatures.csv"

    preprocessing = DataProcessing(filepath)
    X_train, X_test, y_train, y_test = preprocessing.split_data(0.2, 42)
    # preprocessing.plot_data('Liveness vs Loudness by Genre')


    model = Perception() 
    model.train(np.array(X_train).T, np.array(y_train).T, 1000)
    acc = model.predict(y_train)


