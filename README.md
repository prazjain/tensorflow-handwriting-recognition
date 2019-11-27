# Handwriting recognition in Tensor Flow
---

### Tensor Flow
Tensor Flow is a open sourced numerical computation library (by google) for data flow graphs. These graphs must directed acyclic graphs.

*Nodes* are called computation / operators

*Edges* are called data / tensors

In this exercise we will be doing image recognition on handwritten digits with supervised learning.


* We will download MNIST handwritten data set for around 5000 digits, and train our model on that.

* We will prepare our model using above training data set.

* We will download test data also from MNIST for 10 digits, and predict labels for this test data using model prepared with training data. Along with prediction we will also measure accuracy of our prediction.

* Then I have written down some digits myself, and after some image processing, we will test my handwritten digits and predict the labels, and check the accuracy too.

### Setup


        chmod +x predict_digits.py
        pip3 install numpy
        pip3 install tensorflow
        pip3 install matplotlib
        pip3 install Pillow
        ./predict_digits.py

### Output

        Extracting mnist_data/train-images-idx3-ubyte.gz
        Extracting mnist_data/train-labels-idx1-ubyte.gz
        Extracting mnist_data/t10k-images-idx3-ubyte.gz
        Extracting mnist_data/t10k-labels-idx1-ubyte.gz        
        ---------------------------------------------------------
        Predicting 20 digits after learning from 5000 handwritten digits database
        ---------------------------------------------------------
        Test : 0  Prediction:  9  True Label:  9
        Test : 1  Prediction:  6  True Label:  6
        Test : 2  Prediction:  6  True Label:  6
        Test : 3  Prediction:  8  True Label:  8
        Test : 4  Prediction:  9  True Label:  9
        Test : 5  Prediction:  6  True Label:  6
        Test : 6  Prediction:  6  True Label:  6
        Test : 7  Prediction:  3  True Label:  3
        Test : 8  Prediction:  3  True Label:  3
        Test : 9  Prediction:  7  True Label:  7
        ---------------------------------------------------------
        Now Predicting my handwritten digits from trained labels
        ---------------------------------------------------------
        Prediction : 3 , Handwritten label : 8
        Prediction : 7 , Handwritten label : 9
        Prediction : 4 , Handwritten label : 4
        Prediction : 5 , Handwritten label : 5
        Prediction : 8 , Handwritten label : 6
        Prediction : 2 , Handwritten label : 2
        Prediction : 3 , Handwritten label : 3
        Prediction : 1 , Handwritten label : 1
        Prediction : 0 , Handwritten label : 0
        Done
        Accuracy : 0.9999999999999999
        Handwritten accuracy : 0.6666666666666666
