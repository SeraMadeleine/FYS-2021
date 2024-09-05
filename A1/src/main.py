from preprocessing import DataProcessing 
from machineLearning import Perceptron
import numpy as np
import matplotlib.pyplot as plt

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

    # Prediction 
    model = Perceptron() 
    model.train(np.array(X_train).T, np.array(y_train).T, 1)
    accuracy_train = model.predict(y_train)
    print(f'Train Accuracy: {accuracy_train:.4f}')

    # Test set 
    model.forward_pass(np.array(X_test).T)
    accuracy_test = model.predict(y_test)
    print(f'Test Accuracy: {accuracy_test:.4f}')



    # plott streken for å klassifisere dataene
    print("model weight",model.weight)
    print("model weight[0]",model.weight[0])
    print("model.bias[0]: ",model.bias[0])
    print("model.weight[1]: ",model.weight[1])
    print("model.weight[0,0]: ",model.weight[0,0])
    preprocessing.plot_data('Liveness vs Loudness by Genre')
    
    # 2c 
    # [0,-model.bias[0]/model.weight[1,0]] er punktet der linjen krysser hver akse 
    plt.axline(xy1=[0,-model.bias[0]/model.weight[1,0]], xy2=[-model.bias[0]/model.weight[0,0],0], color='black')
    plt.xlim(0,1)
    plt.ylim(-60,5)
    plt.show()


    # plot test og training scatter ved siden av hverandre med linje 
    plt.subplot(1,2, 1)
    plt.scatter(X_train['liveness'], X_train['loudness'], c=y_train, cmap='coolwarm')
    plt.axline(xy1=[0,-model.bias[0]/model.weight[1,0]], xy2=[-model.bias[0]/model.weight[0,0],0], color='black')
    plt.xlim(0,1)
    plt.ylim(-60,5)
    plt.title('Training Data')

    plt.subplot(1,2, 2)
    plt.scatter(X_test['liveness'], X_test['loudness'], c=y_test, cmap='coolwarm')
    plt.title('Test Data')
    plt.axline(xy1=[0,-model.bias[0]/model.weight[1,0]], xy2=[-model.bias[0]/model.weight[0,0],0], color='black')
    plt.xlim(0,1)
    plt.ylim(-60,5)
    plt.show()






    # confusion matrix
    # model.confusion_matrix(y_test)

