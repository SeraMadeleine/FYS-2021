import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load data
data = np.loadtxt('./data/data_problem2.csv', delimiter=',')

# Split data: assuming the first row is x (features) and the second row is y (labels)
x = data[0]     # Features
y = data[1]     # Labels (0 or 1)

# Separate data by class
c0 = x[np.where(y == 0)]        # Extract x values where y = 0
c1 = x[np.where(y == 1)]        # Extract x values where y = 1

# Plot histogram
plt.figure(figsize=(8, 6))

plt.hist(c0, bins=20, alpha=0.5, label='Class 0', color='blue', edgecolor='black')
plt.hist(c1, bins=20, alpha=0.5, label='Class 1', color='pink', edgecolor='black')

# Add plot details
plt.legend(loc='upper right')
plt.title('Histogram of x by Class', fontsize=14)
plt.xlabel('x Values', fontsize=12)
plt.ylabel('Frequency', fontsize=12)

plt.tight_layout()
plt.show()

# print the number of samples in each class 
print('Number of samples in the dataset:', len(x))
print('Number of samples in class 0:', len(c0))
print('Number of samples in class 1:', len(c1))
