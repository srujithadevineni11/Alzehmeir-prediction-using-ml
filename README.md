# Week-1  
## Topic selection :- Searching on different healthcare related topics to define our problem statement.    

Selected topics related to common diseases.    

### Alzehmeir prediction using ml(srujitha)     
#### RESEARCH PAPER   
https://www.frontiersin.org/articles/10.3389/fpubh.2022.853294/full    
##### Alzehmeir Datasets links.   
https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images  
https://www.kaggle.com/datasets/sachinkumar413/alzheimer-mri-dataset  
https://www.kaggle.com/datasets/lukechugh/best-alzheimer-mri-dataset-99-accuracy  

### Diabetes Prediction Using ML Algorithms(Vyshnavi)  
#### RESEARCH PAPERS  
https://www.sciencedirect.com/science/article/pii/S1877050920300557    
##### Diabetes Datasets links.   
https://www.kaggle.com/datasets/akshaydattatraykhare/diabetes-dataset   
https://www.kaggle.com/datasets/mathchi/diabetes-data-set   
https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset   
https://www.kaggle.com/datasets/whenamancodes/predict-diabities  

### Glaucoma Prediction using ML(Lakshmi Sathvika)  
#### RESEARCH PAPERS  
https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0177726  
https://www.mdpi.com/2075-4418/11/3/510
##### DATASETS:  
https://www.kaggle.com/datasets/azmatsiddique/glaucoma-dataset  
https://www.kaggle.com/datasets/sshikamaru/glaucoma-detection  

###  Autism Prediction using ML(Tanya Kavuru) 
#### RESEARCH PAPERS  
##### DATASETS:
https://www.kaggle.com/datasets/faruk268/determinants-of-autism-spectrum-disorder  
https://www.kaggle.com/code/konikarani/autism-prediction  
https://www.kaggle.com/datasets/andrewmvd/autism-screening-on-adults  

### Embarking on a Journey to Illuminate Alzheimer's Disease: Early Detection and Multidisciplinary Insights".  

After conducting extensive research on various diseases, we have decided to concentrate our efforts on Alzheimer's disease prediction. We've chosen to focus on Alzheimer's disease prediction, not only due to its significance but also because it provides an excellent platform for learning and exploring cutting-edge machine learning techniques. Early detection is critical for this progressive condition affecting memory and cognition. Our research spans diverse fields, including neuroscience, genetics, data science, and healthcare, enabling us to develop and apply state-of-the-art ML techniques. With a growing aging population, our work carries substantial societal impact and may lead to improved diagnostic tools. Our mission is to advance early detection and management of Alzheimer's, all while gaining valuable insights into the world of machine learning.

# Week-2 
## Engaging in the exploration of various datasets related to Alzheimer's Disease and thoroughly reviewing a wide range of research papers.   


### We have identified several research papers and thoroughly examined the papers to comprehend the diverse approaches and techniques utilized by different researchers in the field.
https://sci-hub.se/10.1109/ICACCS48705.2020.9074248  
https://www.nature.com/articles/s41598-022-20674-x   
https://www.frontiersin.org/articles/10.3389/fpubh.2022.853294/full   
https://www.sciencedirect.com/science/article/pii/S1532046420301428   
https://www.hindawi.com/journals/jhe/2021/9917919/   
##### We have found some datasets that we can use for Alzheimer's research.
https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images   
https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers    

We selected our dataset from Kaggle, accessible at https://www.kaggle.com/datasets/jboysen/mri-and-alzheimers, to conduct our analysis.


# week-3
## Thorough Analysis of the Numerical Dataset: Comprehensive examination and exploration of the dataset have been conducted to extract valuable insights.

we meticulously executed the following steps for analyzing the Alzheimer's dataset:

### Data Loading and Exploration

- Imported necessary libraries: NumPy, Matplotlib, Pandas, Seaborn.
- Loaded dataset from CSV files using Pandas.
- Explored dataset structure and contents.

### Data Preprocessing

- Handled missing values by filling them with column medians.
- Converted categorical data into numerical format.

### Data Visualization

- Utilized Matplotlib and Seaborn for visualization.
- Created bar graphs to showcase Alzheimer's cases distribution.
- Explored relationship between gender and dementia rates.
- Visualized impact of education years on dementia using kernel density estimation plots.

### Model Building and Evaluation

- Built machine learning models using scikit-learn:
  - Gaussian Navie bayes
  - Bernoulli Navie bayes
  - Support Vector Machine (SVM)
  - Decision Tree
  - XGBoost
  - AdaBoost
  - Random Forest
  - Logistic regression
  - knn
- Split dataset into training and testing sets.
- Evaluated models using:
  - Confusion matrices
  - Classification reports
  - ROC curves
  - Accuracy scores

The accuracys are shown in the below graph. 

<img width="533" alt="Screenshot 2023-10-12 at 4 35 35 PM" src="https://github.com/srujithadevineni11/UROP_project/assets/114294796/61796f4b-6952-4c88-a104-b8ebd524408f">


# week-4

## Thorough Analysis of the Image Dataset: Comprehensive examination and exploration of the dataset have been conducted to extract valuable insights.

To bolster our machine learning model's robustness in early Alzheimer's prediction, we're broadening our dataset exploration. Additional datasets related to Alzheimer's disease are being sought to enhance our model's performance and accuracy. By expanding our data sources, we aim to create a more comprehensive and reliable predictive tool for early Alzheimer's detection.
As part of our dataset expansion strategy, we've chosen to include the Alzheimer's Image Dataset available on Kaggle. This dataset comprises images classified into four distinct classes related to Alzheimer's disease. 

We selected this [Alzheimer's Image Dataset](https://www.kaggle.com/datasets/tourist55/alzheimers-dataset-4-class-of-images) due to its potential to offer valuable visual insights into Alzheimer's disease progression. Integrating visual data alongside structured data is anticipated to lead to a more comprehensive understanding of the condition, thereby improving predictive performance.

We meticulously executed the following steps on the Alzheimer's Image Dataset:

1. **Data Loading and Preparation**: Utilized Python libraries like OpenCV and NumPy to load images from the dataset directory structure. Preprocessed the images by converting them to RGB format, resizing them to a uniform size (224x224 pixels), and storing them along with their corresponding category labels.

2. **Data Visualization**: Visualized a sample of images from each category using Matplotlib, providing insights into the dataset's structure and contents.

3. **Data Preprocessing**: Ensured consistency in pixel values across categories by scaling them to the range [0, 1]. Calculated and displayed the minimum and maximum pixel values for one of the categories, aiding in understanding the data distribution.

4. **Data Augmentation**: Employed TensorFlow's ImageDataGenerator to perform data augmentation, enhancing the diversity of the training data. Various augmentation techniques such as rotation, shifting, shearing, zooming, and flipping were applied to generate augmented images, which were then saved to an output directory.

5. **Feature Extraction**: Utilized a pre-trained VGG16 model to extract features from the images. The model was loaded, and its top classification layers were removed to create a feature extraction model. Features were extracted from each image using this model, enabling downstream tasks such as classification based on learned representations.

6. **Model Training and Evaluation**: Trained multiple machine learning models on the Alzheimer's Image Dataset to predict different stages of Alzheimer's disease. Here are the models we evaluated:

    - Support Vector Machine (SVM)
    - Logistic Regression
    - Multinomial Naive Bayes
    - k-Nearest Neighbors (k-NN)
    - Decision Tree
    - Cnn
    - Ann
      
The accuracys are shown in the below graph. 

<img width="883" alt="Screenshot 2024-04-15 at 7 24 32 PM" src="https://github.com/srujithadevineni11/Alzehmeir-prediction-using-ml/assets/114294796/4dd85f43-fac2-485b-a62d-d5137f169f7d">


## References links
https://ieeexplore.ieee.org/abstract/document/8389326     
https://www.v7labs.com/blog/image-processing-guide
P. Lodha, A. Talele and K. Degaonkar, "Diagnosis of Alzheimer's Disease Using Machine Learning," 2018 Fourth International Conference on Computing Communication Control and Automation (ICCUBEA), Pune, India, 2018, pp. 1-4, doi: 10.1109/ICCUBEA.2018.8697386.
