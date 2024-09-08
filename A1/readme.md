# Assignment 1 - Machine Learning 
## Music Genre Classification


**Project Overview**

This is the first mandatory assignment in Machine Learning (FYS-2021) at UiT for the fall of 2024. This repository contains all the code, the dataset, as well as the assignment text and the report for the assignment. The project involves the classification of music tracks into "Pop" or "Classical" genres based on their features using a custom-built logistic regression model.


### Project Structure

- **A1/**
  - **data/**
    - `SpotifyFeatures.csv` - The dataset file.
  - **doc/**
    - `report.pdf` - The final report document.
    - `assignment_text.pdf` - Text of the assignment.
  - **src/**
    - `main.py` - Orchestrates preprocessing, training, and testing.
    - `preprocessing.py` - Functions for loading and preprocessing data.
    - `machineLearning.py` - Implements and trains the logistic regression classifier.
    - `plot.py` - Functions for visualizing data and results.
  - **plots/**
    - Directory for storing generated plots.
  - `README.md` - Overview and instructions for setting up and running the project.
  - `requirements.txt` - Specifies necessary libraries for the project.

**Setup and Running**

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
