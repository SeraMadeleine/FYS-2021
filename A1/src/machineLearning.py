import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class MachineLearning(): 
    def __init__(self):
        pass


class Perception(): 
    def __init__(self, input_dim=2): 
        self.weight = np.random.rand(input_dim,1)   
        self.bias = np.random.rand(1)
        self.y_hat = None
        self.r = 1e-6                      # learning rate 

    def sigmoid_function(self, x):
        '''
        ## Sigmoid function

        ### Parameters:
        - x : float \\
            The input value to the sigmoid function
        '''

        return (1/(1+np.exp(-x)))

    def forward_pass(self, x):
        '''
        ## Forward pass of the perceptron
        
        ### Parameters:
        - x : np.array \\
            The input data of shape (n, m), where n is the number of features and m is the number of samples.
        '''
        
        x = np.array(x)                                                           # Convert input to numpy array
        self.y_hat = self.sigmoid_function(self.weight.T @ x + self.bias)         # @ = dot product


    def loss (self, y):
        '''
        ## MSE - mean squared error

        ### Parameters:
        - y : float \\
            The true label of sample x (0,1)
        '''

        return np.sum((y - self.y_hat)**2)/len(y)


    def backward_pass(self, x, y):
        '''
        ## Update the weight and bias of the perceptron
        ### Parameters:
        - x : np.array 
        - y : np.array 
        '''
        # Calculate the derivative of the loss function
        d_loss = x @ (self.y_hat - y).T 
        
        # Calculate the new weight
        dw = - np.array(self.r * d_loss, dtype=float)
        db = -np.array(self.r*(self.y_hat-y), dtype=float)
        self.weight += dw 
        self.bias += np.sum(db)


    def predict(self, y):
        '''
        ## Predict the accuracy of the perceptron
        ### Parameters:
        -  y : np.array \\
            The true label of sample x (0,1)
        '''
        self.y_hat = np.round(self.y_hat)          # round so we only work with 0 and 1 values 
        accuracy = np.sum(self.y_hat.reshape(y.shape) == y) / len(y)

        return accuracy
    


    def train(self, x, y, epochs): 
        '''
        ## Train the perceptron
        ### Parameters:
        - x : np.array \\
            The input data of shape (n, m), where n is the number of features and m is the number of samples.
        - y : np.array \\
            The true label of sample x (0,1)
        - epochs : int \\
            The number of epochs to train the perceptron
        '''
        # list for loss and accuracy 
        loss_list = []
        accuracy_list = []


        for epoch in range(epochs): 
            self.forward_pass(x)
            self.backward_pass(x, y)
            print(f'Epoch: {epoch} \n Loss: {self.loss(y)}')
            #print(np.min(self.y_hat), np.max(self.y_hat))
            
            loss_list.append(self.loss(y))
            accuracy_list.append(self.predict(y))

            if epoch % 10 == 0: 
                print(f'Accuracy: {self.predict(y)}')
            print(f'Accuracy: {self.predict(y)}')


        # Plot 
        plt.suptitle('Loss vs accuracy')
        plt.subplot(1,2,1)
        plt.plot(accuracy_list)
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy')

        plt.subplot(1,2,2) 
        plt.plot(loss_list)
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss')
        plt.show()

            

        


  


        
