---
license: mit
---

# Introduction
This project mainly aims to predict the function of protein sequence based on the data set provided by CAFA5.
the CAFA5 competation provided large scale and real protein sequences for training and testing, for the protein sequences information, it is saved as fasta file, containing 142246 protein for training with corresponding lable (GO term) and 141865 protein for testing. In fasta file, each protein contains their unique ID, description, and sequence information (in alphabet). 

Due to the restriction of storage space, this project only contains the core project files for embedding and prediction. You can download the dataset (including the embeddings) files via the following link
https://drive.google.com/file/d/1AMGjTnYXA47rkB2skcZ1HMl9pq-007UQ/view?usp=sharing

A workflow of our project is shown as below which may be helpful when you check the codes.
![image](workflow.png)