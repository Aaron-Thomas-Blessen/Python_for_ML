# Unsupervised Learning: Student Feedback Analysis

## Import
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

# Exploratory Data Analysis
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

# Visualisation
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

# Boxplot of effectiveness
sns.boxplot(y=df_class['Resource Person'], x=df_class['Effectiveness'])
plt.show()

# Boxplot of expertise
sns.boxplot(y=df_class['Resource Person'], x=df_class['Expertise'])
plt.show()

# Boxplot of relevance
sns.boxplot(y=df_class['Resource Person'], x=df_class['Relevance'])
plt.show()

# Boxplot of overall organization
sns.boxplot(y=df_class['Resource Person'], x=df_class['Overall Organization'])
plt.show()

# Boxplot of Branch vs. Content Quality
sns.boxplot(y=df_class['Branch'], x=df_class['Content Quality'])
plt.show()
```

