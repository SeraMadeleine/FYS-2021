# Assignment 2 - FYS-2021: Machine Learning

## Project Structure

```
A2/
│
├── data/                              # Contains dataset files
│   └── data_problem2.csv              # The dataset used for the assignment
│
├── doc/                               # Documentation and related files
│   └── assignment2.pdf                # The assignment details
|   └── report.pdf                     # The report for the given assignment 
│
├── plots/                             # Folder to store generated plots
│   ├── histogram.png                  # Histogram plot of the dataset
│   ├── loss_surface.png               # Loss surface plot for classification
│   └── misclassified.png              # Plot of misclassified points
│
├── src/                               # Source code for the project
│   ├── bayesClassifier.py             # Implementation of Naive Bayes Classifier
│   ├── dataVisualizer.py              # Data visualization utilities
│   ├── loadData.py                    # Data loading utilities
│   └── main.py                        # Main script to run the project
│
├── requirements.txt                   # Overview of the needed packages 
│
└── README.md                          # Project description and setup guide
```


## Setup and Running

1. **Environment Setup**
   - Ensure Python 3.x is installed on your system.
   - Install required Python libraries (pandas, matplotlib, and numpy) by running:
     ```bash
     pip3 install -r requirements.txt
     ```

2. **Running the Code**
   - Navigate to the `src` directory in the terminal.
   - Execute the following command:
     ```bash
     python3 main.py
     ```


