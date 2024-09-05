import numpy as np 
import matplotlib.pyplot as plt
from tqdm import tqdm

class MachineLearning(): 
    def __init__(self):
        pass


class Perceptron(): 
    def __init__(self, input_dim=2): 
        self.weight = np.random.rand(input_dim,1)   
        self.bias = np.random.rand(1)
        self.y_hat = None
        self.learning_rate  = 1e-6                     
        
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
        ## cross-entropy loss function

        ### Parameters:
        - y : float 
            The true label of sample x (0,1)
        '''
        return -np.mean(y * np.log(self.y_hat) + (1 - y) * np.log(1 - self.y_hat))
        # return np.sum((y - self.y_hat)**2)/len(y)


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
        dw = - np.array(self.learning_rate  * d_loss, dtype=float)
        db = -np.array(self.learning_rate *(self.y_hat-y), dtype=float)
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
            for i in tqdm(range(len(y))):
                self.forward_pass(x)
                self.backward_pass(x, y)
                current_loss = self.loss(y)
                current_accuracy = self.predict(y)
                if i % 500 == 0: 
                    print(f'Epoch {epoch}: Loss {current_loss:.4f}, Accuracy {current_accuracy:.4f}')

                loss_list.append(current_loss)
                accuracy_list.append(current_accuracy) 

        return loss_list, accuracy_list


    def confusion_matrix(self, y): 
        '''
        3a

        # TP - true positive 
        # FP - fa1lse positive
        # TN - true negative
        # FN - false negative
        
        '''
        TP, FP, TN, FN = 0, 0, 0, 0

        for i in range(len(y)): 
            if y[i] == 1 and self.y_hat[i] == 1: 
                TP += 1
            elif y[i] == 0 and self.y_hat[i] == 1: 
                FP += 1
            elif y[i] == 0 and self.y_hat[i] == 0: 
                TN += 1
            elif y[i] == 1 and self.y_hat[i] == 0: 
                FN += 1
        
        print(f'TP: {TP}, FP: {FP}, TN: {TN}, FN: {FN}')
            

        


  


        
