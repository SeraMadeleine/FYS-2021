import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import gamma, norm
import os 

class DataVisualizer:
    def __init__(self):
        pass  

    def plot_histogram(self, c0, c1):
        """Plot the histogram of the data for each class."""
        plt.figure(figsize=(8, 6))
        plt.hist(c0, bins=20, alpha=0.5, label='Class 0', color='blue', edgecolor='black')
        plt.hist(c1, bins=20, alpha=0.5, label='Class 1', color='pink', edgecolor='black')
        plt.legend(loc='upper right')
        plt.title('Histogram of x by Class', fontsize=14)
        plt.xlabel('x Values', fontsize=12)
        plt.ylabel('Frequency', fontsize=12)
        plt.tight_layout()
        self.save_plot('histogram.png')
        # plt.show()

    def plot_misclassified(self, x_test, y_test, y_pred, classifier):
        """Plot the data distribution with misclassified points and PDF curves."""
        plt.figure(figsize=(12, 8))
        
        # Scatter plot for each class
        plt.scatter(x_test[y_test == 0], np.zeros_like(x_test[y_test == 0]) - 0.02, 
                    color='blue', label='Class 0 (True)', alpha=0.6, s=50)
        plt.scatter(x_test[y_test == 1], np.zeros_like(x_test[y_test == 1]) + 0.02, 
                    color='pink', label='Class 1 (True)', alpha=0.6, s=50)
        
        # Highlight misclassified points
        misclassified = x_test[y_test != y_pred]
        plt.scatter(misclassified, np.zeros_like(misclassified), 
                    color='green', marker='x', s=100, label='misclassified', alpha=1)
        
        # Plot PDF curves
        x_range = np.linspace(0, max(x_test), 1000)
        plt.plot(x_range, gamma.pdf(x_range, a=classifier.alpha, scale=1/classifier.beta) * 0.5, 
                 color='blue', lw=2, label='Class 0 PDF (Gamma)')
        plt.plot(x_range, norm.pdf(x_range, loc=classifier.mu, scale=classifier.sigma) * 0.5, 
                 color='pink', lw=2, label='Class 1 PDF (Gaussian)')

        plt.legend(loc='upper right')
        plt.title('Data Distribution with misclassified Points and PDFs', fontsize=16)
        plt.xlabel('x Values', fontsize=14)
        plt.ylabel('Density', fontsize=14)
        plt.ylim(-0.1, 0.15)  
        plt.grid(True, alpha=0.3)
        plt.tight_layout()

        self.save_plot('misclassified.png')
        # plt.show()

    def save_plot(self, filename):
        """Save the plot in the plots directory."""
        # Create the 'plots' directory if it doesn't exist
        directory = 'plots'
        if not os.path.exists(directory):
            os.makedirs(directory)
        
        # Save the plot in the 'plots' directory
        filepath = os.path.join(directory, filename)
        plt.savefig(filepath)
        print(f"Plot saved as {filepath}")



