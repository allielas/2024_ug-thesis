import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pycytominer import aggregate, annotate, normalize, feature_select

# Load the CSV files
cells = pd.read_csv('cells.csv')
lysosomes = pd.read_csv('lysosomes.csv')
mitochondria = pd.read_csv('mitochondria.csv')

# Merge the dataframes on the common identifiers
merged_data = cells.merge(lysosomes, left_on='object_number', right_on='parent_cell', suffixes=('_cell', '_lysosome'))
merged_data = merged_data.merge(mitochondria, left_on='object_number', right_on='parent_cell', suffixes=('', '_mitochondria'))

# Group by well and calculate the mean of each feature
grouped_data = merged_data.groupby('well').mean()

# Aggregate the data by well
aggregated_data = aggregate(merged_data, strata=['well'], features='infer', operation='mean')

# Normalize the data
normalized_data = normalize(aggregated_data, method='standardize')

# Feature selection
selected_features = feature_select(normalized_data, operation='variance_threshold', threshold=0.1)

# Plot the distribution of selected features per well
features = selected_features.columns
for feature in features:
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='well', y=feature, data=selected_features)
    plt.title(f'Distribution of {feature} per well')
    plt.xticks(rotation=90)
    plt.show()
