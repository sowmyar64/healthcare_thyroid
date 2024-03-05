# healthcare_thyroid
healthcare_thyroid_differentiated_analysis
# Introduction:

This dataset comprises clinicopathologic features collected over a span of 15 years, with the objective of predicting the recurrence of well-differentiated thyroid cancer. Each entry in the dataset represents an individual patient who was monitored for a minimum of 10 years. The dataset was created as part of research in the intersection of Artificial Intelligence and Medicine, without specific funding provided for its development.
The dataset contains 13 features, each representing different clinicopathologic characteristics relevant to thyroid cancer recurrence prediction. Notably, there are no recommended data splits, indicating that the dataset may be used flexibly for various analyses and modeling tasks.
Importantly, the dataset does not include any sensitive information, ensuring patient privacy and confidentiality. Additionally, there are no missing values in the dataset, facilitating seamless analysis and modeling processes.

# Objectives
The primary objective of this study is to develop machine learning models capable of predicting the likelihood of recurrence in patients diagnosed with well-differentiated thyroid cancer. Despite the generally low mortality rate associated with thyroid cancer, the risk of recurrence remains a significant concern. Accurately identifying an individual patient's risk of recurrence is crucial for guiding subsequent management and follow-up protocols.
In pursuit of this objective, the study will undertake exploratory data analysis (EDA) and employ data visualization techniques to gain insights into the characteristics and distributions of the clinicopathologic features present in the dataset. Through EDA and visualization, the aim is to deepen our understanding of the relationships between various features and the recurrence of thyroid cancer.

The project's primary goals encompass several key steps:
1) Ensuring data quality and consistency by addressing missing values, outliers, and duplicates.
2) Standardizing data formats and rectifying any inconsistencies in column names or values to enhance data integrity.
3) Utilizing EDA techniques to analyze data distributions, trends, and interrelationships effectively.
4) Summarizing and visually representing key features using descriptive statistics, histograms, scatter plots, and heat maps.
5) Identifying correlations between variables and uncovering potential patterns or anomalies that may influence recurrence risk.
6) Employing various visualization tools and libraries, such as matplotlib, Seaborn, and Plotly, to generate both static and interactive visualizations, including plots and charts, for comprehensive data exploration and presentation.

# About Dataset
The dataset contains 13 features, each representing different clinicopathologic characteristics relevant to thyroid cancer recurrence prediction.

1) Age: Represents the age of individuals in the dataset.

2) Gender: Indicates the gender of individuals (e.g., Male or Female).

3) Smoking: Possibly an attribute related to smoking behavior. The specific values or categories would need further exploration.

4) HX Smoking : Indicates whether individuals have a history of smoking (e.g., Yes or No).

5) HX Radiotherapy : Indicates whether individuals have a history of radiotherapy treatment (e.g., Yes or No).

6) Thyroid Function: Possibly indicates the status or function of the thyroid gland.

7) Physical Examination: Describes the results of a physical examination, likely related to the thyroid.

8) Adenopathy: Indicates the presence and location of adenopathy (enlarged lymph nodes).

9) Pathology: Describes the types of thyroid cancer based on pathology examinations, including specific subtypes like "Micropapillary Papillary," "Follicular," and "Hürthle cell."

10) Focality: Indicates whether the thyroid cancer is unifocal or multifocal.

11) Risk: Represents the risk category associated with thyroid cancer.

12) T: Represents the Tumor stage of thyroid cancer, indicating the size and extent of the primary tumor.

13) N: Represents the Node stage of thyroid cancer, indicating the involvement of nearby lymph nodes.

14) M: Represents the Metastasis stage of thyroid cancer, indicating whether the cancer has spread to distant organs.

15) Stage: Represents the overall stage of thyroid cancer based on the combination of T, N, and M stages.

16) Response: Describes the response to treatment, including categories such as 'Indeterminate,' 'Excellent,' 'Structural Incomplete,' and 'Biochemical Incomplete.' 

17) Recurred: Indicates whether thyroid cancer has recurred (e.g., Yes or No).


# 2. Methodology
### Data Collection
Gather relevant datasets from sources such as Differentiated Thyroid Cancer Recurrence.

### Data Cleaning
Identify and address missing values, outliers, and inconsistencies in the datasets to ensure data integrity and accuracy.

### Exploratory Data Analysis (EDA)
Explore the datasets to understand distributions, trends, and relationships between variables using descriptive statistics and data visualization techniques.

### Data preprocesing Pipeline
This project employs a comprehensive data preprocessing pipeline to prepare the dataset for machine learning analysis. 

#### Introduction to Preprocessing Methodology:
The preprocessing pipeline aims to address data inconsistencies, handle missing values, and prepare the dataset for machine learning modeling. It involves dividing the dataset into features and the target variable, performing necessary transformations, and splitting the data into training and testing sets.

The pipeline consists of the following key steps:
#### Preprocessing Step:
In the preprocessing step, the dataset was initially divided into features and the target variable. The features were stored in a variable named X, while the target variable, denoted as 'Recurred', was stored in y. Additionally, a dictionary called label2id was created to map unique labels in the target variable to numeric values, facilitating compatibility with certain machine learning algorithms.


#### Data Splitting:
The dataset was then split into training and testing sets using the train_test_split function from scikit-learn. This split was stratified, ensuring that the class distribution in the resulting training and testing sets mirrored that of the original dataset and also helps preventing biases and ensuring reliable model evaluation.

#### Target Variable Distribution:
Finally, the distribution of the target variable was verified in both the training and testing sets to confirm the balanced representation of classes. This step ensures that the machine learning model is trained and tested on data that accurately reflects the underlying distribution of classes, thereby improving the model's generalizability and performance.

#### Categorization of Predictors:
During preprocessing, features were categorized into numerical and categorical predictors to enable tailored transformation steps. Numerical predictors, containing 'int' and 'float' data types, were stored in numerical_predictor, while categorical predictors, including 'object' and 'category' data types, were stored in categorical_predictors. This approach allows for efficient preprocessing techniques such as scaling for numerical predictors and encoding for categorical predictors, ensuring data readiness for machine learning models.

#### The data preprocessing pipeline using ColumnTransformer from scikit-learn:
1) Importing Libraries: Import necessary libraries from scikit-learn for preprocessing, including ColumnTransformer, Pipeline, SimpleImputer, StandardScaler, and OneHotEncoder.
2) Define Preprocessing Steps:
  - For numerical variables:
      a) Use SimpleImputer to fill missing values with the median of the column.
      b) Use StandardScaler to standardize the features by removing the mean and scaling to unit variance.
  - For categorical variables:
      a) Use SimpleImputer to fill missing values with the most frequent value in the column.
      b) Use OneHotEncoder to encode categorical features as one-hot vectors.
3) Combine Preprocessing Steps:
Combine the preprocessing steps for both numerical and categorical variables using ColumnTransformer. This transformer applies the defined preprocessing steps to the appropriate columns in the dataset.
4) Fit and Transform Training Data:
Call the fit_transform method on the preprocessor to fit the preprocessing steps to the training data (X_train) and transform it.
5) Transform Test Data:
Call the transform method on the preprocessor to apply the fitted preprocessing steps to the test data (X_test). Note that only transformation is applied to the test data since the preprocessing steps were already fitted to the training dat

#### Implementation for Categorical Predictors
- For preprocessing the categorical predictors in the dataset, a ColumnTransformer from sklearn.compose is utilized. This transformer allows for the application of specific transformations to different columns in the dataset.
- The preprocessing pipeline involves encoding categorical variables into binary vectors using OneHotEncoder from sklearn.preprocessing. The parameter handle_unknown='ignore' is set to handle any unseen categories during the transformation process, ensuring robustness when dealing with new data.
- Additionally, the 'passthrough' argument is used to keep the remaining columns unchanged, preserving their original values.
- Fit and Transform Training Data: The fit_transform method was employed on the preprocessor to both fit and transform preprocessing steps to the training data.
- Transform Test Data: Subsequently, the transform method was utilized on the preprocessor to apply previously fitted preprocessing steps to the test data. Notably, only transformation was applied to the test data, as preprocessing steps had already been fitted to the training data.

#### Overall Impact:
The data preprocessing pipeline plays a crucial role in ensuring data quality, accurate encoding of categorical predictors, consistency and compatibility with machine learning models, laying the foundation for accurate and reliable analysis.



### Logistic Regression Model Training and Evaluation
In this project, we trained a Logistic Regression model to predict the likelihood of recurrence in patients diagnosed with well-differentiated thyroid cancer. The model was trained on preprocessed data and evaluated using both training and test datasets.
#### Logistic Regression Model Initialization:
The Logistic Regression model is initialized with specified parameters such as random_state for reproducibility and n_jobs for parallel processing.
#### Model Training:
The initialized model is trained using the fit method on the preprocessed training data.
#### Prediction:
Predictions are made on both the training and test data using the trained model.
#### Performance Evaluation:
Balanced accuracy scores are computed for both the training and test predictions.
#### Output:
The computed balanced accuracy scores for both the training and test sets are printed.

### we can describe the model selection and training process as follows:

### Model Selection and Training:
- We began by defining a set of candidate models for the classification task, including Support Vector Classifier (SVC), Random Forest Classifier, Extra Trees Classifier, XGBoost Classifier, LightGBM Classifier, and CatBoost Classifier. Each model was initialized with parameters optimized for handling imbalanced datasets and ensuring reproducibility.
- The primary criterion for model selection was their performance on unseen data. To this end, we evaluated each model's balanced accuracy score on the test set.
- Models were trained using the preprocessed training data, consisting of standardized numerical features and one-hot encoded categorical predictors.

### Initialization of Models:
- Each model was configured with specific parameters to enhance its performance and adaptability to the dataset characteristics. For instance, SVC models were configured with probability=True to allow for probability estimates and class_weight='balanced' to address class imbalance.
- Similarly, Random Forest and Extra Trees Classifiers were initialized with class_weight='balanced' to handle class imbalance effectively.

### Model Training and Evaluation:
- The training process involved fitting each model to the preprocessed training data (X_train_prep_models) and corresponding target labels (y_train).
- After training, the models made predictions on both the training and test datasets (X_train_prep_models and X_test_prep_models).
- Subsequently, the balanced accuracy scores were computed for each model on both the training and test sets using the balanced_accuracy_score function from scikit-learn.

### Result Presentation:
- The balanced accuracy scores obtained for each model on both the training and test sets were stored in dictionaries (accuracy_train and accuracy_test), with the model names serving as keys.
- This systematic approach allowed for the efficient training and evaluation of multiple models, enabling the selection of the best-performing model for the classification task.

### Progress Monitoring:
Throughout the training loop, a progress bar was displayed using the tqdm library, providing real-time updates on the completion status of each model's training process.


![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/48bff962-1f74-41a3-b063-d6794ccfb48a)

The process of visualizing the performance metrics is succinctly summarized as follows:
- Metric Data Preparation: Balanced accuracy scores for each model on training and test sets are organized into DataFrames (metric_train and metric_test), facilitating easy comparison.
- Visualization Setup: A bar plot is generated using matplotlib, with appropriate figure size and axis labels.
- Bar Plot Creation: Two sets of bars represent training and test set scores for each model, providing a visual comparison.
- Labeling Bars: Height annotations are added to each bar for clarity using the autolabel function.
- Legend and Axis Labels: A legend distinguishes between training and test set scores, while axis labels and title provide context.
- Displaying the Plot: The finalized plot is displayed for easy interpretation.

##### The best model with respect to the evaluation metric is CatBoostClassifier and it outperforms the base model, so we will calculate some additional metrics with this model.

In the final stage of the model evaluation process, predictions were generated using the trained model (clf6). The predictions were made on both the training and test datasets.

For the training set, the accuracy score was calculated to be accuracy_train_score, while for the test set, the accuracy score was accuracy_test_score.

Furthermore, classification reports were generated for both the training and test sets to provide a detailed breakdown of the model's performance across different classes ('No' and 'Yes').

These evaluation metrics offer insights into the model's ability to make accurate predictions and its performance on unseen data. They serve as crucial indicators of the model's effectiveness and generalization capability.


![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/0a838133-8743-441a-a51c-a21bd42f898b)

The confusion matrices provide valuable insight into the performance of our predictive model for identifying thyroid cancer recurrence. 

#### Confusion Matrix Analysis
#### Training Dataset:
- True Negatives (TN): 192
- False Positives (FP): 0
- False Negatives (FN): 76
- True Positives (TP): 2

#### Test Dataset:
- True Negatives (TN): 83
- False Positives (FP): 0
- False Negatives (FN): 0
- True Positives (TP): 30

The model demonstrates relatively good performance in predicting the likelihood of recurrence in patients diagnosed with well-differentiated thyroid cancer. It correctly identifies a significant number of instances as either having or not having recurrence, as evidenced by the high number of true negatives and true positives. The absence of false positives and false negatives in the test dataset indicates that the model generalizes well to unseen data.



![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/443bcf73-1092-4fbb-92a4-cc1f29f047f1)

- The ROC AUC plot illustrates the model's performance in distinguishing between positive and negative instances of thyroid cancer recurrence. The x-axis represents the False Positive Rate (FPR), ranging from 0.0 to 1.0, while the y-axis represents the True Positive Rate (TPR), also ranging from 0.0 to 1.0.

- A perfect classifier would achieve an AUC score of 1.0, while a random classifier would score 0.5. In our case, the model exhibits excellent predictive performance, as evidenced by an AUC score of 1.0000 for the training dataset and 0.9959 for the test dataset.

- These high AUC scores indicate that the model effectively balances true positive and false positive rates, showcasing its robust discriminatory ability. Thus, the ROC AUC analysis affirms the model's capability to accurately predict the likelihood of thyroid cancer recurrence.


![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/112e2683-81af-4042-a4b4-25d11b86d5bb)
- The Precision-Recall Curve illustrates the precision and recall of the model across various thresholds. Precision is depicted on the y-axis, while recall is shown on the x-axis.
- This curve reflects the model's performance on both the training and test datasets. A curve that extends closer to the upper-right corner indicates better model performance.
- In the displayed graph, both the training and test Precision-Recall Curves demonstrate strong performance. This suggests that the model achieves high precision while maintaining high recall across different thresholds, indicating its efficacy in predicting thyroid cancer recurrence.

# Visualization

![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/d00a1ecb-30f0-44dc-a9c7-ea6d8cfa3eb8)

The visualization consists of a histogram and a kernel density estimate (KDE) curve, providing insights into the age distribution within the dataset:
#### Histogram Interpretation:
- The histogram displays the distribution of ages observed in the dataset.
- It represents age groups on the X-axis and the frequency of individuals falling within each age range on the Y-axis.
- Bins divide the age range into intervals, with each bin representing a specific age group.
- Peaks in the histogram indicate age ranges with higher frequencies of individuals.
#### KDE Curve Interpretation:
- The KDE curve overlays the histogram, offering a smoothed estimation of the probability density function of the age distribution.
- It provides a continuous representation of the distribution pattern, helping to identify trends or patterns in age distribution.


![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/df1190d2-3a1b-4843-a7fc-76e1d48e85a4)

#### Gender Distribution:
- The count plot displays the distribution of patients across gender categories.
- It provides insights into the prevalence of male and female patients with thyroid cancer.
- The majority of individuals diagnosed with thyroid cancer are female, as indicated by the higher count of female patients compared to males.

#### Smoking Distribution:
- The count plot illustrates the distribution of patients based on smoking status.
- It helps visualize the prevalence of smoking habits among patients with thyroid cancer.
- There is no significant difference in the distribution of smoking habits between smoking and non-smoking categories, suggesting a weak association between smoking and thyroid cancer in this dataset.

#### Treatment Response Distribution:
- The count plot shows the distribution of patients according to their treatment response.
- It provides insights into the effectiveness of treatment in patients with thyroid cancer.
- The high count in the "Yes" category suggests a substantial proportion of patients positively responded to treatment and recovered from the disease.


![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/a130a66a-d7df-498b-a2c7-22c4c96cf354)

The visualizations comprise three pie charts, each depicting the distribution of different categorical variables within the dataset.
#### Percentage of Risk:
- This pie chart illustrates the distribution of risk levels among patients.
- Each segment represents a risk category, with its size indicating the proportion of patients in that category.
- It provides an overview of the distribution of risk levels within the dataset.

#### Percentage of Each Stage:
- This pie chart displays the distribution of cancer stages among patients.
- Each segment corresponds to a cancer stage category, representing its proportion in the dataset.
- It offers insights into the prevalence of different cancer stages among the patient population.

#### Percentage of Adenopathy:
- This pie chart showcases the distribution of adenopathy categories within the dataset.
- Each segment represents an adenopathy category, displaying its percentage in the dataset.
- It provides an understanding of the prevalence of adenopathy among patients.

![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/91a8a0ba-b765-4d61-ad7b-c53201adf457)

The violin plot visualizes age distributions across cancer stages, with gender segmentation.The x-axis represents the cancer stages, while the y-axis denotes the age of patients. Each plot represents a cancer stage, displaying the age distribution density through violin width. Gender categories are split within each stage, facilitating comparison. This visualization aids in understanding age distribution patterns across stages and genders.



![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/5577ac8e-665f-4ffa-834f-9ff445821b99)

The count plot showcases the distribution of thyroid function categories among patients, categorized by recurrence status. 
- The x-axis denotes various thyroid function categories.
- The y-axis represents the count of patients within each category.
- Bars are color-coded based on recurrence status, allowing for easy differentiation between recurrence and non-recurrence.
- This visualization provides valuable insights into the distribution of thyroid function categories and their correlation with recurrence status.


![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/8a61b79d-7361-4d86-b36d-d096ed01968e)

The violin plot visualizes the distribution of patient ages across different thyroid function categories, with gender segmentation. 
- The x-axis represents thyroid function categories, while the y-axis displays patient ages.
- Each violin plot illustrates the age distribution within a specific thyroid function category.
- Gender differentiation is applied, with separate distributions for male and female patients shown side by side within each category.
- This visualization enables the identification of age distribution patterns across thyroid function categories and provides additional insights through gender segmentation.


![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/affdacba-fc34-4e45-b720-798da86ecb8a)


The count plot illustrates the distribution of pathology categories among patients, differentiated by recurrence status. 
- The x-axis represents different pathology categories.
- The y-axis indicates the count of patients in each category.
- Bars are color-coded based on recurrence status, with different hues representing recurrence and non-recurrence.
- This visualization offers insight into the distribution of pathology categories and their association with recurrence status.


  ![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/f28ae567-02ff-4334-a1e0-3621ae7565d7)

The count plot illustrates the distribution of thyroid nodules based on their focality, categorized by recurrence status.
- The x-axis represents different focality categories of thyroid nodules.
- The y-axis indicates the count of patients associated with each focality category.
- Bars are color-coded to differentiate between cases with recurrence and those without, providing a visual comparison.
- This visualization aids in understanding the distribution of thyroid nodules according to focality and their association with recurrence status.


![image](https://github.com/sowmyar64/healthcare_thyroid/assets/43263218/4d29a888-3fd2-434a-8eb9-b7cf029815d8)

The heatmap visualization presents the Predictive Power Score (PPS) matrix, which assesses the predictive power of features in predicting target variables. 
- The heatmap visualizes the PPS matrix, where each cell represents the predictive power score between two features.
- Higher PPS scores indicate stronger predictive relationships between features.
- The color intensity on the heatmap corresponds to the magnitude of the PPS scores, with darker shades indicating higher scores.
- Annotations within each cell display the precise PPS score, providing additional insights into feature predictability.
- This visualization aids in identifying the most influential features for predicting target variables, facilitating feature selection and model development.


# Conclusion:

The analysis of well-differentiated thyroid cancer recurrence reveals significant insights into patient demographics and the predictive capabilities of machine learning models. With a mean age of 40.87 years and a majority of patients being female, the dataset provides valuable demographic information.We employed various machine learning models, including Support Vector Classifier (SVC), Random Forest Classifier, Extra Trees Classifier, XGBoost Classifier, LightGBM Classifier, and CatBoost Classifier, to predict the likelihood of recurrence. Each model was optimized for handling imbalanced datasets and ensuring reproducibility.Among these models, the CatBoost Classifier demonstrated superior performance, outperforming the base model and achieving the highest evaluation metric. We further evaluated this model using additional metrics. Achieving an AUC score of 1.0000 for training and 0.9959 for testing. The model exhibited robust discriminatory ability and high precision-recall balance, affirming its efficacy in predicting thyroid cancer recurrence. These findings underscore the potential of machine learning in personalized treatment strategies and patient management.
