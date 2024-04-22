![ntu-placeholder-d](https://github.com/angyvette03/SC1015-FCE1-Team1/assets/115095394/33bd50a3-f51e-4385-9476-6b07d42ec02b)
# SC1015 FCE1 Team 1 Breast Cancer Prediction
## [Video Submission](https://youtu.be/LCMnqtbWRWw)
---
### Introduction
#### Preface 
Our team's research into breast cancer diagnostic measures was prompted by its alarming prevalence. Breast cancer is the most common cancer among women in Singapore, and globally, it affected 2.3 million women with 670,000 deaths. 

Early detection can significantly improve breast cancer treatment outcomes. 

These statistics for the current diagnostic solutions for breast cancer also show how more efforts should be put into increasing the accuracy of diagnostic solutions, and how solutions should be able to alleviate the heavy workload on our healthcare sector. 

#### Machine Learning as a potential solution
With these issues in mind, our team came up with our problem statement. How can machine learning models, trained on patient data, enhance the early diagnosis of breast cancer by accurately classifying the disease?

While a medical professional should confirm the results, machine learning definitely can augment the initial prediction phase.

---

### The Data Science Pipeline
#### Practical Motivation
We note that diagnosis of breast cancer can be a data-driven problem and can be predicted through various sample data. Through machine learning, we can significantly reduce the workload for medical professionals by providing a preliminary prediction that they can verify with their domain knowledge.

#### Sample Collection
Our dataset (Breast Cancer Wisconsin (Diagnostic) Data Set) was obtained from Kaggle. This data set is updated annually.

Features from the dataset are computed from a digitised image of a fine needle aspirate (FNA) of a breast mass. They describe characteristics of the cell nuclei present in the image. These features make up the 30 predictor columns available in the dataset. These columns are split into 3 categories of 10 identical columns each: mean, worst and error.

#### Problem Formulation
Problem Statement: **How can machine learning models, trained on patient data, enhance the early diagnosis of breast cancer by accurately classifying the disease?**

#### Data Preparation
**Goal:** To clean the dataset and prepare it for use in machine learning techniques. To evaluate the appropriateness of our dataset we focused on skewness and kurtosis.

**Skewness:** Reflected how symmetric the data was. Ideally should be close to zero.
We addressed skewness through normalising the data with a log scale.

**Kurtosis:** A measure of how heavy-tailed or light-tailed or data was relative to a normal distribution. Ideally should be close to zero.
We addressed kurtosis through removing outliers (defined as points lying outside of the _Q1 - 1.5(IQR) and Q3 + 1.5(IQR)_ whiskers of the box plots)

#### Exploratory Analysis
Data Visualisation provided us with a clearer understanding of the dataset.

We plotted box plots, histograms, heatmaps and swarm plots for the EDA process.

#### Statistical Description and Analytic Visualisation
For each of our plots we focused on the following:

**Box plots: Have outliers been successfully removed?**

Box Plots with raw data

<img width="816" alt="Screenshot 2024-04-22 at 6 14 44 PM" src="https://github.com/angyvette03/SC1015-FCE1-Team1/assets/115095394/c00a4fe2-7e98-4fd3-b0c2-f59cb2c2b058">

Box Plots with cleaned data
<img width="816" alt="Screenshot 2024-04-22 at 6 15 15 PM" src="https://github.com/angyvette03/SC1015-FCE1-Team1/assets/115095394/ca2ab1a4-b8c4-450e-a20d-8db596c37d4c">

**Histograms: Have we successfully corrected for skewness in our data?**

Histograms with raw data

<img width="816" alt="Screenshot 2024-04-22 at 6 15 48 PM" src="https://github.com/angyvette03/SC1015-FCE1-Team1/assets/115095394/0aa0a4fd-8fea-482d-af26-0fa56fb9d5ed">

Histograms with cleaned data

<img width="816" alt="Screenshot 2024-04-22 at 6 16 35 PM" src="https://github.com/angyvette03/SC1015-FCE1-Team1/assets/115095394/4e2edce5-cd5a-4172-ae96-43ad5dc659e6">

**Heatmaps: What is the correlation between our data?**
Which features have the highest correlation with 'benign_0__mal_1'? We picked out the columns with the highest correlations (<-0.7 OR >0.7) for use in our predictive models. The 9 columns to focus on: **'mean radius', 'mean perimeter', 'mean area', 'mean concavity', 'mean concave points', 'worst radius', 'worst perimeter', 'worst area', 'worst concave points'**
![download (6)](https://github.com/angyvette03/SC1015-FCE1-Team1/assets/115095394/15aa2d5b-6376-4f80-95a3-adf53e0355d2)


**Swarm plots: Do our selected features have low degrees of overlap between the Benign and Malignant box plots?**
![download (7)](https://github.com/angyvette03/SC1015-FCE1-Team1/assets/115095394/4a112942-b30f-4404-8d18-5ba2c7d580f7)


#### Machine Learning
In this project, we chose to go with the decision tree classifier, logistic regression, random forest for model. 

We selected these models based on 2 main criteria:
**Dataset Types**
Our dataset comprised predominantly continuous variables, and our goal was to predict a categorical variable.
**Performance**
The model needed to perform well on metrics which were used for validation in the healthcare industry.

#### Algorithmic Optimisation
**Logistic Regression:** We choose the hyperparameter C to be relatively small (0.1) compared to its default value of 1 as we are uncertain if the dataset is truly representative of the real world situation in breast cancer. This small C value ensures that a larger weightage is placed on the complexity penalty for extreme parameters in the model.

**Decision Trees:** To ascertain which model is better, we used a tree of depth 2 and a tree of depth 4 to understand if classification accuracy would be improved.

#### Information Presentation
**ROC Curve:**
![download (8)](https://github.com/angyvette03/SC1015-FCE1-Team1/assets/115095394/5655f759-1879-427c-97db-15d0167c5ed6)

**Comparison of models:**

<img width="603" alt="Screenshot 2024-04-22 at 6 19 46 PM" src="https://github.com/angyvette03/SC1015-FCE1-Team1/assets/115095394/5ba63044-ef5a-4aca-86a9-c94c06ecdb78">

![download (9)](https://github.com/angyvette03/SC1015-FCE1-Team1/assets/115095394/9813f1ee-57b2-49a7-a3f4-e90379d9eabc)


#### Statistical Inference
**Classification accuracy**
- It is the simplest measure of how well the model is performing. 
- Calculated by the sum of correct predictions over all the predictions made by the model.
- 
**Area under curve for ROC Curve**
- Better conforms to the standards set by clinicians in clinical trials
- An ROC curve (receiver operating characteristic curve) is a graph showing the performance of a classification model at all classification thresholds 
- This curve plots two parameters: True Positive Rate. False Positive Rate
  
**F1-Score**
- Described as the harmonic mean of the precision and recall of a classification model. 
- The two metrics contribute equally to the score, ensuring that the F1 metric correctly indicates the reliability of a model.

**False Negative Rate**
- Calculated by dividing the number of false negatives over the actual number of positive rates
- Chosen as it is more dangerous to label a patient as negative if they are in fact positive as it would cause patients to avoid checkups thinking that they are healthy when they are actually not.

#### Intelligent Decision
Based on the results, we decided that Random Forest Classification is the most suitable model.

#### Ethical Consideration
Trust is as important as technical accuracy when considering AI in medicine like breast cancer prediction. Patients and doctors may hesitate to rely on 'black box' models that don't explain their reasoning. However, dismissing AI altogether would be a wasted opportunity. To bridge this trust gap, we need AI that allows doctors to understand its decision-making process. Decision trees are a good option, providing clear insights into the factors considered at each step, promoting confidence in the results.

---

### Concluding Thoughts:

#### Future recommendations:
**Dataset Selection**
Train on larger and better datasets with more inputs from both the benign and malignant cases to improve the robustness of the models.
**Hyperparameter Tuning**
Use informed search strategies to reduce training time by automatically tuning the hyperparameters of the models

#### Final Remarks:
Ultimately, our model is not aimed to replace the work of radiologists and doctors, but rather to aid in the diagnosis of breast cancer.

The aim of our model is to augment the diagnosis and to provide a second point of confirmation. A trained medical professional is still needed to confirm if the results are accurate and consistent with their domain knowledge. However, we have shown that breast cancer diagnosis can be a data-driven problem, and that technology and machine learning can be used to aid in the medical field.







