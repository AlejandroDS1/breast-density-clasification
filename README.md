# Breast Density Classification using DenseNet121

This repository contains the code developed for the final grade project at the University of Barcelona by Alejandro Cantero. The project focuses on classifying breast density using the DenseNet121 architecture.

## Directory Structure  

- **`code/`**:  
  The main directory with the implementation of the DenseNet121 model, training pipeline, and helper utilities necessary for model development and testing.

- **`metrics/`**:  
Contains jupyter notebooks that compute and show metrics belonging to the csv files produced by the model's evaluation.

- **`data_analisis/`**:  
Contain one jupyter notebook per database. Computes and prints database information usefull to understand the size and shape of the mammograms.

- **`README.md`**:  
  This file. Provides an overview of the project, setup instructions, and explanations of the repository structure.

## Getting Started

To set up and run the project, follow these steps:

1. **Clone the repository**:
    ```bash
    git clone https://github.com/AlejandroDS1/breast-density-clasification.git
    ```
2. **Navigate to the project directory**:
    ```bash
    cd breast-density-clasification
    ```
3. **Install dependencies**:
Ensure you have Python 3.x installed. Then, install the required packages:

    ```bash
    pip install -r requirements.txt
    ```

4. **Prepare the data**:
Place your mammogram images in the appropriate directory and run the preprocessing scripts in preprocess/ to prepare the dataset.

5. **Train the model**:
Use the training scripts found in the project/ directory to train the DenseNet121 model on your preprocessed data.

6. **Evaluate the model**:
After training, run the evaluation scripts to assess the performance of your model.

# Notes
- Ensure that your dataset is properly labeled and structured before preprocessing.

- Modify the hyperparameters and configurations in the training scripts as needed.

- For optimal performance, use a GPU-enabled environment during training.

Acknowledgments
This project was developed as part of the final degree thesis at the University of Barcelona.

I want to thank to Dra. Laura Igual for her guidance during all the project and for providing a usefull server that made the development easier.

I want to thank Phd. Lidia Garrucho for providing a preprocessed version of the dataset used in this final degree project.