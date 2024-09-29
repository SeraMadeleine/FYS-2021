from loadData import DataLoader
from dataVisualizer import DataVisualizer
from bayesClassifier import BayesClassifier


if __name__ == '__main__':
    # Load the data from the CSV file and print the information
    data_loader = DataLoader('./data/data_problem2.csv')
    data_loader.print_data_info()

    # Create the data visualizer object
    visualizer = DataVisualizer()

    # Plot the histogram of the data
    visualizer.plot_histogram(data_loader.c0, data_loader.c1)

    # Train the classifier and test it on the test set
    classifier = BayesClassifier(data_loader.x, data_loader.y)
    x_test, y_test, y_pred, accuracy = classifier.train_and_test()
    print(f"Test accuracy: {accuracy:.4f}")

    visualizer.plot_misclassified(x_test, y_test, y_pred, classifier)

