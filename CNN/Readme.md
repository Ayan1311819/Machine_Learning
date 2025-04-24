# Crop disease classification
This project aims to classify crop diseases using a deep learning model. 
It utilizes a dataset of images of various crops and their diseases, and trains a convolutional neural network (CNN) to identify the specific diseases affecting each crop.
# Dataset
The dataset used in this project is the "Bangladeshi Crops Disease Dataset" available on Kaggle(bec that was the nearest we could get identical of indian crops). 
It contains images of 14 different crop diseases.
There were like 4 major crop categories: 1)White 2)Rice 3)Corn 4)Potato
Each category has 3-4 disease categories. For example: 
- Rice: Brown Spot
- Rice: Leaf Blast
- Rice: Healthy
- Rice: Hispa 
# Cleaning Data and preprocessing
1)Basic data cleaning: like getting rid of corrupt images or large image files (>6MB)

2)Rescaling the images

3)**Class Mode:** The `class_mode` parameter in `flow_from_directory` is set to `categorical` for multi-class classification, where the target variable represents multiple categories.

  **Color Mode:** The `color_mode` parameter in `flow_from_directory` is set to `"rgb"` to ensure that the images are loaded in color with three channels (Red, Green, and Blue). This is crucial for deep learning models that leverage color information for analysis and classification.
# Recipe for a Machine Learning algorithm
1) Data specification (input prob, output)

Input: Consists of images of various crops. Will be passed to the model as 3d tensor(rgb values)

Output: Classifying images into disease classes. (softmax at output layer)
2) Cost Function (metrics->cost function) + regularization
Metrics: Using metrics like accuracy, precision, recall, and F1-score to evaluate the model's performance.

Cost Function: Categorical cross entropy

Regularization: Applied data augmentation techniques(like rotation,shear,zoom,) to prevent overfitting.
3) Optimization procedure

Adam optimizer 

Learning rate: Found one manually. 
4) Model

Architechure: CNN based on Alexnet architechure.(Moved from 16 million parameter model to 200k to 800k now)

Implementation: handled by keras.

Class weights: Because of data imbalance between classes.

# Results
Training accuracy:96,  Validation accuracy:93 

# Future Improvements
1) Model design : Could try models for categrozing the major categories first and then diff models for each category's disease classification.

2) Hypermaraters: Using validation set to search or grid search.

3) Data Mismatch: The images it is trained on is diff from the ones people will be taken from cameras. Thats a data problem and have to diversify the dataset.
