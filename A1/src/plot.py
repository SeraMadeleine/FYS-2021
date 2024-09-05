import os
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

class Plots(): 
    def __init__(self):
        self.data = None
        self.plot_directory="../plots"


    def scatter_plot(self, plot_title, data, save_plot=True):
        """
        Plots the data on a 2D plane with 'liveness' on the x-axis and 'loudness' on the y-axis.

        ### Parameters:
        plot_title : str \\
            The title of the plot.
        """
        self.data = data

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
        if save_plot == True: 
            self.save_plot(plot_title)
        
            plt.show()

    def save_plot(self, plot_title):
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
        if not os.path.exists(self.plot_directory):
            os.makedirs(self.plot_directory)

        plt.savefig(os.path.join(self.plot_directory, plot_filename))
        print(f"Plot saved to {self.plot_directory}/{plot_filename}")


    def plot_data(self):
        pass
        

    # 2c
    def plot_decision_boundary(self, model, preprocessing):
        self.scatter_plot('Liveness vs Loudness with Decision Boundary', preprocessing.data, save_plot=False)
    
        # [0,-model.bias[0]/model.weight[1,0]] er punktet der linjen krysser hver akse 
        plt.axline(xy1=[0,-model.bias[0]/model.weight[1,0]], xy2=[-model.bias[0]/model.weight[0,0],0], color='black')
        plt.xlim(0,1)
        plt.ylim(-60,5)

        self.save_plot('Liveness vs Loudness with Decision Boundary')
        plt.show()

    def subplots(self, X_train, y_train, X_test, y_test, model, plot_title='Training and Test Data with Decision Boundary'):
        # plot test og training scatter ved siden av hverandre med linje 
        plt.suptitle(plot_title)

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

        self.save_plot(plot_title)
        plt.show()


    def plot_loss_vs_accuracy(self, accuracy_list, loss_list):
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

        self.save_plot('Loss vs accuracy')
        plt.show()

    

    def plot_confusion_matrix(self, confusion_matrix):
        '''
        Plot the confusion matrix using seaborn's heatmap.

        Parameters:
        - confusion_matrix : np.array
            A 2x2 matrix with [TN, FP; FN, TP].
        '''
        plt.figure(figsize=(6,4))
        sns.heatmap(confusion_matrix, annot=True, fmt='d', cmap='Blues', cbar=False,
                    xticklabels=['Predicted 0', 'Predicted 1'], 
                    yticklabels=['Actual 0', 'Actual 1'])

        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.title('Confusion Matrix')
        plt.show()
