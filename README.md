# OWPCP: A Deep Learning Model to Predict Octanol-Water Partition Coefficient 
# Motivation 
The octanol-water partition coefficient (LogP) is a crucial parameter in drug design, influencing a molecule's absorption, distribution, metabolism, excretion, and toxicity (ADMET) properties. Experimental determination of LogP is often costly and time-consuming, which has led to the development of computational models for its prediction. In this context, we present OWPC, a deep learning-based framework that leverages combined molecular fingerprints and chemical feature descriptors to predict LogP values. The model learns a distinct set of representations for each type of descriptor, resulting in a robust feature space that enhances the predictive capability. A comprehensive evaluation of OWPC demonstrated its superior performance compared to traditional computational methods and other machine learning models.
# Requirements
To run the OWPC.py script, you need to have Python 3.10.12 or later installed. Additionally, the following Python libraries are required:
- [Numpy](https://numpy.org/)
- [Pandas](https://pandas.pydata.org/)
- [Keras](https://keras.io/)
- [Tensorflow](https://www.tensorflow.org/)
- [RDKit](https://www.rdkit.org/)
- [Keras Tuner](https://keras.io/keras_tuner/)
- [Scikit-Learn](https://scikit-learn.org/stable/)
# Data Files
## Description of Data
The dataset used for this project is a CSV file named Full_lib.csv, containing molecular structures and their corresponding LogP values. Each row in the dataset represents a single molecule and is structured as follows:

SMILES: The SMILES (Simplified Molecular Input Line Entry System) representation of the molecule.
logP: The experimentally measured LogP value.
Example of dataset:

|  Smiles | logP |
| ---- | --- |
| [H]C1=C(Cl)C(Cl)=C(Cl)C2=C(Cl)C(Cl)=C(Cl)C(Cl)=C12| 8.2 |
|  CCN1CCc2nc(N)oc2CC1	| 0.05 |
|  Nc1ccc(cc1)C(=N)N | -0.37 |


## Feature Extraction
The input features for each molecule are generated using the following molecular descriptors:

Morgan Fingerprints: These are circular fingerprints generated using RDKit with a specified radius and bit size.
MACCS Keys: A predefined set of 167 chemical feature descriptors.
The features are concatenated into a single feature vector for each molecule, which is then used as the input to the model. The dataset is divided into training and testing sets, and an internal validation set is created for hyperparameter tuning.


# Machine Learning Model
## Model Architecture

The OWPC model consists of two main components:

Feature Encoders: Separate encoders for Morgan Fingerprints and MACCS Keys learn compact representations of the input features.
Decoder: A multi-layer neural network that combines the encoded features to predict the LogP value.
The architecture is defined using the model_builder() function and optimized through a hyperparameter tuning process using Keras Tuner.

## Model Implementation
The architecture of the model is as follows:

Morgan Fingerprint Encoder: A series of dense layers that progressively reduce the dimensionality of the input.
MACCS Key Encoder: Similar to the Morgan Fingerprint Encoder but adapted for the different input size of the MACCS keys.
Decoder: Combines the outputs of both encoders and passes them through additional dense layers to predict the LogP value.
A detailed diagram of the architecture can be found in the following figure.
![Presentation1](https://github.com/user-attachments/assets/9f8910ec-0475-4bb5-a6ae-4a56386640ed)

## Running OWPC
The OWPC.py file provided in this repository contains a basic implementation of the OWPC model. The script performs the following steps:

- Data Preparation: Reads the input dataset (Full_lib.csv) and extracts the features using the feature extraction functions.
- Model Building: Creates the OWPC model using the defined architecture.
- Training and Evaluation: Trains the model using the training set and evaluates it on the test set.
