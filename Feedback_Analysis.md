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


