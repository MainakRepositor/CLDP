# **Corn Leaf Diseases Detection**

[![Python](https://img.shields.io/badge/python-3670A0?style=for-the-badge&logo=python&logoColor=ffdd54)](https://www.python.org)
[![streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)](https://streamlit.io/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-%23D00000.svg?style=for-the-badge&logo=Keras&logoColor=white)](https://keras.io/)

An end-to-end Deep Learning project to detect diseases that attack corn leaves

## **Problem Statement**
The plant diseases compose a threat to global food security and smallholder farmers whose livelihoods depend mainly on agriculture and healthy crops. In developing countries, smallholder farmers produce more than 80% of the agricultural production, and reports indicate that more than fifty percent loss in crop due to pests and diseases. The world population expected to grow to more than 9.7 billion by 2050, making food security a major concern in the upcoming years. Hence, rapid and accurate methods of indentying plant diseases are needed to do the appropiate measures.

This Streamlit App utilizes a Deep Learning model to detect diseases that attact the corn leaves, based in digital images.



## **Data Preparation**

In this project, we use the dataset version with augmentation for the corn dataset, which contained four different classes (Blight, Common rust, Gray Leaf Spot, and Healthy).

Data preprocessing steps:

- Data normalization[0,1]
- Data augmentation using saveral techniques such as:
  - image flipping
  - zoom
  - shear
  - width and height shift
  - image rotation
  - Image brightness range
  - Featurewise center
  - Featurewise std normalization

[Dataset Link](https://www.kaggle.com/datasets/smaranjitghose/corn-or-maize-leaf-disease-dataset/data)

## Modelling
In this project I tasted the folling Convolutional Neural Network Architecture by transfer learning method:

 - MobileNetV2 (Pretrained with imginet images)

Fine Tune model Performnace: 98.5% (accuracy score metric)


