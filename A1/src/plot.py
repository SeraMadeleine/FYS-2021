import os
import matplotlib.pyplot as plt


class Plots(): 
    def __init__(self):
        self.data = None
        pass

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
        if save_plot: 
            self.save_plot(plot_title)

    def save_plot(self, plot_title, plot_directory="../plots"):
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
        if not os.path.exists(plot_directory):
            os.makedirs(plot_directory)

        plt.savefig(os.path.join(plot_directory, plot_filename))
        print(f"Plot saved to {plot_directory}/{plot_filename}")


    def plot_data(self):
        pass

    def subplots(self):
        pass