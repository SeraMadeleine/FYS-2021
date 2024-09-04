import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt

class MachineLearning(): 
    def __init__(self):
        pass


class Perception(): 
    def __init__(self): 
        self.weight = np.random.rand(2,1)   
        self.bias = np.random.rand(1)
        self.y_hat = None
        self.r = 0.00001                           # learning rate 

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
        # print(f"y_hat shape: {self.y_hat.shape} \n\n ")
        # print(f'y hat: {self.y_hat}')
        # return self.y_hat

    def loss (self, y):
        '''
        ## MSE - mean squared error

        ### Parameters:
        - y : float \\
            The true label of sample x (0,1)
        '''

        return np.sum((y - self.y_hat)**2)/len(y)


    def backward_pass(self, x, y):
        # backward pass oppdaterer vekten i vektmatrisen for å minnimere loss
        # derivere loss med hensyn på vektene (w) 

        # Calculate the derivative of the loss function
        d_loss = x @ (self.y_hat - y).T 
        
        
        # Calculate the new weight
        dw = - np.array(self.r * d_loss, dtype=float)
        db = -np.array((self.y_hat-y), dtype=float)
        self.weight += dw 
        self.bias += np.sum(db)

    def train(self, x, y, epochs): 

        # TODO: shuffle shit 


        for epoch in range(epochs): 
            self.forward_pass(x)
            self.backward_pass(x, y)
            print(f'Epoch: {epoch} \n Loss: {self.loss(y)}')
            #print(np.min(self.y_hat), np.max(self.y_hat))
            
            if epoch % 10 == 0: 
                print(f'Accuracy: {self.predict(y)}')
            print(f'Accuracy: {self.predict(y)}')
            

    def predict(self, y):
        # sammelikne yhat og y 
        self.y_hat = np.round(self.y_hat)
        accuracy = np.sum(self.y_hat.reshape(y.shape) == y) / len(y)

        return accuracy
    


        
