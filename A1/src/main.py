from preprocessing import DataProcessing 
from machineLearning import Perceptron
import numpy as np

if __name__ == "__main__":
    filepath = "../data/SpotifyFeatures.csv"

    preprocessing = DataProcessing(filepath)
    X_train, X_test, y_train, y_test = preprocessing.split_data(0.2, 42)
    # preprocessing.plot_data('Liveness vs Loudness by Genre')
    
    
    # Shuffle the dataset 
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train.iloc[indices]
    y_train = y_train.iloc[indices]


    # X_test = np.random.permutation(X_test)
    # y_test = np.random.permutation(y_test)

    model = Perceptron() 
    model.train(np.array(X_train).T, np.array(y_train).T, 600)
    accuracy = model.predict(y_train)
    y_pred = model.predict(X_test)
    print(f'Test Accuracy: {accuracy:.4f}')



