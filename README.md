# **Opticat Character Recognition**

## **About**

**NOT COMPLETED PROJECT** for OCR

## **Description**

In this repo there is an attempt to create an OCR model for the first time 





## **Project Structure**
```
├── Interface
│   ├── google_app
│   ├── interface
├── data
├── notebooks
├── src
│   ├── dataset
│   ├── text encoding
│   ├── text processing
│   ├── training
│   └── inference
└── utils
```

The repository is organized as follows:

- **Interface/:** This directory is the django project and contains `google_app` as the django app.  

- **data/:** This directory contains the dataset used for training and evaluation. It includes both the raw data and preprocessed versions, if applicable.

- **notebooks/:** This directory contains Jupyter notebooks that provide step-by-step explanations of the data exploration, preprocessing, model training, and evaluation processes. **Explore if you are feeling adventurous**

- **src/:** This directory contains the source code for the project, including data preprocessing scripts, model training scripts, and evaluation scripts.

- **utils/:** This directory contains utility functions and helper scripts used throughout the project.

## Getting Started

*   It is recommended to set up a virtual environment for this project using **python 3.8.16**



To get started with the project, follow these steps:

1. Clone the repository: 
   ```
   git clone https://github.com/Aylore/blnk-OCR.git
   ```
2. Change directory into the repository:
   ```
   cd blnk-OCR
   ```

3. Install the **required** dependencies:
     ```
     make install
     ```


## **Pipeline**


1. ### Text Processing

    I tired various methods for text encoding , *none of them seemed to get good results so I just setteled for simple encoding*

2. ### Image Processing

    The images were resized to 50 * 200
    I extracted the text only in new images with all black background but seemed to confuse the model for some reason
   

3. ### Django Integration
   
   After training I used django to create and easy to use *not too fancy* GUI.


## **Running The Project**

1.  Use 

   ```
    make run
   ```


## **Results**

![accuracy](accuracy.jpg)


## **Contributing**
If you would like to contribute to this project, Feel Free to make a pull request 



## **Future Work**
*  Extract characters one by one using CV and start classifying them by a pretrained model 
*  Use attention based model 







