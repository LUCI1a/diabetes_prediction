import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
data = pd.read_csv("diabetes.csv")

X = data.iloc[:, :-1]
data_positive = data[data['Outcome'] == 1].drop(columns=['Outcome'])
scaler = StandardScaler()
X_scaled = scaler.fit_transform(data_positive)
pca = PCA(n_components=4)  # 降维
X_reduced = pca.fit_transform(X_scaled)
print(pca.explained_variance_ratio_)

components = pca.components_

feature_names = X.columns
important_features = []

for i, component in enumerate(components):
    max_index = abs(component).argmax()
    important_features.append((feature_names[max_index], component[max_index]))

print("\nThe most contributing feature in each principal component：")
for i, (feature, weight) in enumerate(important_features):
    print(f"Main Ingredients {i+1}: feature {feature}, weight {weight:.4f}")

total_importance = abs(components).sum(axis=0)
total_importance_sum = total_importance.sum()
normalized_importance = total_importance / total_importance_sum
print(normalized_importance )
important_indices = normalized_importance.argsort()[-8:][::-1]
important_features = feature_names[important_indices]

for feature in important_features:
    print(feature)
# 获取这些特征的重要性
importance_values = normalized_importance[important_indices]
plt.figure(figsize=(12,8))
plt.bar(important_features, importance_values, color='skyblue')
plt.ylabel('важность', fontsize=12)
plt.xticks(rotation=20)
plt.grid(True)
plt.show()

