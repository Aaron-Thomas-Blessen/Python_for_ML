# Unsupervised Learning: Student Feedback Analysis

## Install
```python
pip install pandas
```

## Import 
```python
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
import warnings
warnings.filterwarnings('ignore')
```
## Load the dataset
```Python
df_class = pd.read_csv("https://raw.githubusercontent.com/sijuswamy/Intel-Unnati-sessions/main/Feed_back_data.csv")
df_class.head()
df_class.sample(5).style.set_properties(**{'background-color': 'darkgreen',
                           'color': 'white',
                           'border-color': 'darkblack'})
```

## Data Wrangling
```python
df_class.info()
# Drop unnecessary columns
df_class = df_class.drop(['Timestamp', 'Email ID',

# Rename columns for clarity
df_class.columns = ["Name", "Branch", "Semester", "Resource Person", "Content Quality", "Effectiveness", "Expertise", "Relevance", "Overall Organization"]
```

## Exploratory Data Analysis
```python
# Check for null values
df_class.isnull().sum().sum()

# Get dimensions of the dataframe
df_class.shape

# Percentage analysis of Resource Person distribution
round(df_class["Resource Person"].value_counts(normalize=True) * 100, 2)

# Percentage analysis of student distribution
round(df_class["Name"].value_counts(normalize=True) * 100, 2)
```

## Visualisation
```python
import seaborn as sns
import matplotlib.pyplot as plt

# Plot faculty-wise distribution
plt.figure(figsize=(12, 6))
sns.countplot(x='Resource Person', data=df_class)
plt.title("Faculty-wise Distribution of Data")
plt.show()

# Boxplot of each aspect rated by students
sns.boxplot(y=df_class['Resource Person'], x=df_class['Content Quality'])
plt.show()
```
![Image Colorizer](https://drive.google.com/uc?id=1-c9fmuiNRX8Z8BU2l0tgAN4Ls2Do8J1P)

```python
# Boxplot of effectiveness
sns.boxplot(y=df_class['Resource Person'], x=df_class['Effectiveness'])
plt.show()
```
![Image](https://drive.google.com/uc?id=1_j8JfO6YgY2TXR--6VQe0ZePfjASkCLX)

```python
# Boxplot of expertise
sns.boxplot(y=df_class['Resource Person'], x=df_class['Expertise'])
plt.show()
```
![Image](https://drive.google.com/uc?id=1sx1f0irX77ivFnYlyMKypLJ9szZPkUbF)
```python
# Boxplot of relevance
sns.boxplot(y=df_class['Resource Person'], x=df_class['Relevance'])
plt.show()
```
![Image](https://drive.google.com/uc?id=103B3vSuYre13DdyM77tiBQp6L91qK4So)

```python
# Boxplot of overall organization
sns.boxplot(y=df_class['Resource Person'], x=df_class['Overall Organization'])
plt.show()
```
![Image](https://drive.google.com/uc?id=1M4s8z3o0Z7hDiCOiquDSUN17nr-6Jm71)

```python
sns.boxplot(y=df_class['Resourse Person'],x=df_class['Branch'])
plt.show()
```
![Image](https://drive.google.com/uc?id=1wU2uM60piu9zJGJzxIoU2ujIKSf_OcX1)

```python
# Boxplot of Branch vs. Content Quality
sns.boxplot(y=df_class['Branch'], x=df_class['Content Quality'])
plt.show()
```
![Image](https://drive.google.com/uc?id=1dQUUUfTP1W75cIWHZM2quF0C7p9vL2ZE)


## Using K-means Clustering to Identify Segmentation Over Student Satisfaction

### Finding the Best Value of K Using Elbow Method

```python
# Import necessary libraries
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Define input columns and X data
input_col = ["Content Quality", "Effeciveness", "Expertise", "Relevance", "Overall Organization"]
X = df_class[input_col].values

# Initialize an empty list to store the within-cluster sum of squares
wcss = []

# Try different values of k
for k in range(1, 11):
    kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)  # Inertia calculates sum of square distance in each cluster

# Plot the within-cluster sum of squares for different values of k
plt.plot(range(1, 11), wcss, marker='o')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Within-Cluster Sum of Squares (WCSS)')
plt.title('Elbow Method')
plt.show()
```
![Image](https://drive.google.com/uc?id=1NPpV4YPjvFcUg6its20sNLCiQ2IKbbYV)

## Using Gridsearch Method

```python
# Import necessary libraries
from sklearn.model_selection import GridSearchCV

# Define the parameter grid
param_grid = {'n_clusters': [2, 3, 4, 5, 6]}

# Create a KMeans object
kmeans = KMeans(n_init='auto', random_state=42)

# Create a GridSearchCV object
grid_search = GridSearchCV(kmeans, param_grid, cv=5)

# Perform grid search
grid_search.fit(X)

# Get the best parameters and the best score
best_params = grid_search.best_params_
best_score = grid_search.best_score_
print("Best Parameters:", best_params)
print("Best Score:", best_score)
```
## Implementing K-means Clustering

```python
# Perform k-means clustering
k = 5  # Number of clusters
kmeans = KMeans(n_clusters=k, n_init='auto', random_state=42)
kmeans.fit(X)
```

## Extracting Labels and Cluster Centers
```python
# Get the cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_

# Add the cluster labels to the DataFrame
df_class['Cluster'] = labels
```

## Visualizing the Clustering Using the First Two Features
```python
# Visualize the clusters
plt.scatter(X[:, 1], X[:, 2], c=labels, cmap='viridis')
plt.scatter(centroids[:, 1], centroids[:, 2], marker='X', s=200, c='red')
plt.xlabel(input_col[1])
plt.ylabel(input_col[2])
plt.title('K-means Clustering')
plt.show()
```

![Image](https://drive.google.com/uc?id=1Pk0Yt0BTrbir7l8fPCvPvPJR27Psdg70)


## Perception on Content Quality Over Clusters
```python
import pandas as pd

pd.crosstab(columns=df_class['Cluster'], index=df_class['Content Quality'])
```


## Analysis

## 1. The problem statement:
A study of the segmentation of the Intel Certification course participants over satisfaction level.

## 2. An introduction stating the relevance of feedback analysis
- Feedback analysis serves as a cornerstone in understanding and improving performance across diverse domains. Whether in education, business, or personal development, the ability to dissect feedback provides invaluable insights that drive progress and refinement.

- In educational settings, feedback analysis empowers instructors to gauge the effectiveness of their teaching methods, identify areas of strength, and pinpoint opportunities for growth. 

- In essence, feedback analysis serves as a catalyst for growth and improvement across all facets of life.

## 3. Methodology: 
explaining the exploratory and ML approaches used in the analysis with proper justifications.

#### Exploratory Data Analysis (EDA)
- **Data Cleaning:** The initial step involved loading the dataset and inspecting its structure using `df.info()` to identify any missing values or inconsistencies. Unnecessary columns like 'Timestamp' and 'Email ID' were dropped to streamline the dataset.
- **Data Visualization:** Various visualizations such as count plots and boxplots were utilized to gain insights into the distribution and relationships between different variables. Visualizations were created using libraries like Seaborn and Matplotlib.

#### Machine Learning (ML) Approaches
- **K-means Clustering:** K-means clustering was employed to segment students based on their satisfaction levels across different aspects like content quality, effectiveness, expertise, relevance, and overall organization. The optimal number of clusters (k) was determined using the Elbow Method and Gridsearch Method to find the best value of k.
- **Justification:** K-means clustering was chosen as it is a simple yet effective unsupervised learning algorithm suitable for segmenting data into clusters based on similarity. It allows for the identification of patterns and groupings within the dataset, providing valuable insights into student satisfaction and feedback.

#### Justifications
- **EDA:** Exploratory data analysis was conducted to gain a comprehensive understanding of the dataset's characteristics and distributions. This helped identify any data anomalies, patterns, or trends that could influence the subsequent analysis. Visualizations were utilized to present the findings in an interpretable and insightful manner.
- **K-means Clustering:** K-means clustering was selected for its simplicity, efficiency, and scalability, making it suitable for analyzing large datasets and identifying distinct clusters or segments within the data. The Elbow Method and Gridsearch Method were employed to determine the optimal number of clusters, ensuring the robustness and accuracy of the clustering results.

By combining exploratory data analysis with machine learning techniques like K-means clustering, this analysis aims to provide actionable insights into student feedback and satisfaction, facilitating informed decision-making and continuous improvement in educational practices.

## 4.EDA
### Exploratory Data Analysis (EDA)

Exploratory Data Analysis (EDA) is an essential step in understanding the structure, patterns, and relationships within a dataset. In this section, we will explore the dataset and gain insights into its characteristics using various statistical and visual methods.

### Dataset Overview
```python
df_class.info()
```
#### Data Cleaning
We will check for any missing values and handle them accordingly. Additionally, we may drop any unnecessary columns to streamline the dataset.

#### Descriptive Statistics
We will compute summary statistics such as mean, median, standard deviation, etc., to understand the central tendency and dispersion of numerical variables.

```python

df_class.describe()
```
### Visualizations
Visualizations such as histograms, boxplots, and scatter plots will be utilized to explore the distribution and relationships between different variables. This will help identify any patterns or trends within the data.

```python
import seaborn as sns
import matplotlib.pyplot as plt

# Example visualization: Histogram of content quality ratings
plt.figure(figsize=(8, 6))
sns.histplot(df_class['Content Quality'], bins=10, kde=True)
plt.title('Distribution of Content Quality Ratings')
plt.xlabel('Content Quality')
plt.ylabel('Frequency')
plt.show()
```
### Correlation Analysis
We will examine the correlation between variables to identify any significant relationships or dependencies.

```python
correlation_matrix = df_class.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f")
plt.title('Correlation Matrix')
plt.show()
````
### Outlier Detection
Outliers may have a significant impact on the analysis. We will detect and handle outliers using appropriate methods such as IQR (Interquartile Range) or Z-score.

## 5. Machine Learning Model to study segmentation:
** K-means clustering**
### Machine Learning Model: K-means Clustering for Segmentation

K-means clustering is a popular unsupervised learning algorithm used for segmentation and clustering of data into distinct groups based on similarity. In this section, we will implement a K-means clustering model to study segmentation in the dataset.

### Introduction to K-means Clustering
K-means clustering is a centroid-based algorithm that aims to partition data into K clusters, where each data point belongs to the cluster with the nearest mean (centroid). It iteratively assigns data points to the nearest centroid and updates the centroids based on the mean of the data points in each cluster.

### Implementation in Python
We will use the scikit-learn library in Python to implement the K-means clustering algorithm.

```python
from sklearn.cluster import KMeans

# Initialize KMeans object
kmeans = KMeans(n_clusters=K, init='k-means++', random_state=42)

# Fit the model to the data
kmeans.fit(X)

# Get cluster labels and centroids
labels = kmeans.labels_
centroids = kmeans.cluster_centers_
```
### Key Parameters
n_clusters: The number of clusters to form.
init: Method for initialization ('k-means++' for smart initialization).
random_state: Seed for random number generation.
Interpretation of Results
After fitting the K-means model to the data, we can interpret the results by examining the cluster labels assigned to each data point and the centroids of each cluster. Visualization techniques such as scatter plots or cluster heatmaps can be used to visualize the clusters and their centroids.

### Evaluation
Evaluation of K-means clustering can be challenging as it is an unsupervised learning algorithm. However, metrics such as silhouette score or inertia can be used to evaluate the quality of clustering. Silhouette score measures the cohesion and separation of clusters, while inertia represents the sum of squared distances of samples to their closest cluster center.

## 6.RESULTS

### Content Quality and Effectiveness
- **Content Quality:** Most teachers are rated highly, generally above 3.5 out of 5, indicating good performance.
- **Effectiveness:** There is one outlier with a lower effectiveness score, suggesting a potential discrepancy between content quality and delivery.

### Expertise
- Ratings for expertise are generally high, but there are outliers with lower scores, indicating areas for improvement or mismatches between expectations and teaching style.

### Overall Organization
- Scores for overall organization are strong across the board, indicating well-prepared and structured teaching approaches.

### Branch Comparison (CSE, ECE, RB, IMCA)
- Variation in median scores, with RB branch having a noticeably lower median, indicating potential issues specific to that department.

### Individual Performance
- Some teachers show high consistency in scores across categories, while others exhibit more variability, reflecting differences in teaching methods or student engagement.

### Elbow Method and K-means Clustering
- Elbow method suggests 3 or 4 clusters for grouping data, indicating distinct groups.
- K-means plot shows potential clustering of individuals or branches based on evaluated metrics.

### Relevance
- Relevance ratings are consistently high, indicating material taught is considered pertinent and applicable by students.

**Note:** Outliers can significantly influence interpretation, and further investigation might be needed. Feedback is subjective and influenced by various factors, including student expectations and subject difficulty.

