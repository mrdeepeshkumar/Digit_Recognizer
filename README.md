# Image Processing
# Digit_Recognizer 
## Installation
### Install the requirements 
This project requires Python 3 and the following Python libraries installed:

* [matplotlib](http://matplotlib.org/)
* [NumPy](http://www.numpy.org/) 
* [Pandas](http://pandas.pydata.org/)
* [scikit-learn](http://scikit-learn.org/stable/)

You will also need to have tool to run and execute a [Jupyter Notebook.](http://ipython.org/index.html)
Make sure you use Python 3.If you do not have Python installed yet, it is highly recommended that you install the Anaconda distribution of Python, which already has the above packages and more included.

### Intalling Anaconda Python
The easiest way to install Jupyter Notebook as well as NumPy, Pandas,matplotlib is to start with the Anaconda Python distribution.
Follow the installation instructions to download the [Anaconda](https://www.anaconda.com/distribution/) Python. We recommend using Python 3.7.

git clone https://github.com/mrdeepeshkumar/Digit_Recognizer Use cd to navigate into the top directory of the repo on your machine

Launch Jupyter by entering

This will open the Jupyter Notebook software and project file in your browser. 
## Download the data 
Before running the notebook, you'll first need to download all data we'll be using. This data is located in the MNIST_dataset.csv. 
We will extract these into the same directory as Digit_Recognizer.

       or 
You can download the dataset from Kaggle [MNIST Dataset](https://www.kaggle.com/c/digit-recognizer/data)
## Objective:
The objective is to identify digits from a dataset of tens of thousands of handwritten digits

## Project overview:
A classic problem in the field of pattern recognition is that of handwritten digit recognition. We have images of handwritten digits 
ranging from 0-9 written by various people in boxes of a specific size - similar to the application forms in banks and universities.

The goal is to develop a model that can correctly identify the digit (between 0-9) written in an image. 

## Data Description:
I used the MNIST data which is a large database of handwritten digits. The 'pixel values' of each digit (image) comprise the features, 
and the actual number between 0-9 is the label. 

Since each image is of 28 x 28 pixels, and each pixel forms a feature, there are 784 feature. I trained SVMs models achieved thi

##### Target Variable:
`label:`  0-9 digits {0,1,2,3,4,5,6,7,8,9} 

### Starting the project:
`Digit_Recognizer`repository contains the all necessary project files. To run this project , you may download the all files.The implimented code is provided in the `Handwritten_Digit_Recognition.ipynb` jupyter notebook file.You will also be required to use the MNIST_data.csv dataset files to complete your work. 

### Run
To successfully run the project In a terminal or command window, navigate to the top-level project directory Digit_Rcognizer/ (that contains this README) and run one of the following commands:
     
    ipython notebook Handwritten_Digit_Recognition.ipynb
  or
    
    jupyter notebook Handwritten_Digit_Recognition.ipynb
    
### Methodology:
#### Data Understanding: 
The training dataset is quite large (42,000 labelled images), it would take a lot of time for training an SVM on the full MNIST data,
so we could sub-sample the data for training (10-20% of the data should be enough to achieve decent accuracy).
#### Data Cleaning & Preprocessing:
Each digit/label has an approximately 9%-11% fraction in the dataset and the dataset is balanced
#### Model Building:
Support vector machine performs well on balanced dataset so applied SVMs and performed the hyperparameter tuning.
Built two SVMs model: 
1. `Linear Model`: linear kernel
2. `Non Linear Model`: In non linear model kernels are Polynomial and RBF
#### Evaluation:
  Deployed a SVM model with 0.92 % of an accuracy. I used RBF kernel.
### Conclusions
The final accuracy on test data is 92%. Note that this can be significantly increased by using the entire training data of 42,000 images (we have used just 10% of that!).

