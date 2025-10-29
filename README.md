This repository contains raw implementation of foundational deep learning models, tested on bechmark datasets. Its an attempt to understand the evolution of modern deep learning.
Models:-
  1. A simple image classification model with 6 layer CNN feature extractor and a Fully connected layer. Trained on CIFAR 10 Dataset achieveing an accuracy of 75%. 
  2. Resnet50- A resnet50 classifier reproduced as described in the paper "Deep Residual learning for image recognition". Tested on CIFAR10 dataset(original architecture is tweeked for the input data). no of epochs: 50. achieved an accuracy of 77.54 on the test set.
  3. InceptionNet(V1): modelled the InceptionNet as decribed in the literature going deeper with convolutions. the model was trained on CIFAR10 dataset with scheduled learning rate as mentioned in the paper for 100 epochs. It achieved an accuracy of 88.7%.
