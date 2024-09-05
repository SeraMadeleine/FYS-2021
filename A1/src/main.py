from preprocessing import DataProcessing 
from machineLearning import Perceptron
import numpy as np
from plot import Plots

if __name__ == "__main__":

    # The filepath to the dataset
    filepath = "../data/SpotifyFeatures.csv"

    # Initialize the Plots and dataprossesing classes, and load the data
    plot = Plots()
    preprocessing = DataProcessing(filepath)

    # Split the data into training and testing sets 
    X_train, X_test, y_train, y_test = preprocessing.split_data(0.2, 42)
    
    # Generate the initial scatter plot
    plot.scatter_plot('Liveness vs Loudness by Genre', preprocessing.data)
        
    # Shuffle the dataset 
    indices = np.arange(X_train.shape[0])
    np.random.shuffle(indices)
    X_train = X_train.iloc[indices]
    y_train = y_train.iloc[indices]

    # Train the model on the training data 
    model = Perceptron() 
    loss_list, accuracy_list = model.train(np.array(X_train).T, np.array(y_train).T, 1)
    plot.plot_loss_vs_accuracy(accuracy_list, loss_list)
    

    # Evaluate the model 
    accuracy_train = model.predict(y_train)
    print(f'Train Accuracy: {accuracy_train:.4f}')
    model.forward_pass(np.array(X_test).T)
    accuracy_test = model.predict(y_test)
    print(f'Test Accuracy: {accuracy_test:.4f}')
    
    
    # Plot decision boundaries for the the complete data set, and the training and test data
    plot.plot_decision_boundary(model, preprocessing)
    plot.subplots(X_train, y_train, X_test, y_test, model)

 




    # confusion matrix
    # model.confusion_matrix(y_test)

