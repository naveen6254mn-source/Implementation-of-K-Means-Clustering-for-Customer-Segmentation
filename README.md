# Implementation-of-K-Means-Clustering-for-Customer-Segmentation

## AIM:
To write a program to implement the K Means Clustering for Customer Segmentation.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1.Import the necessary packages using import statement.

2.Read the given csv file using read_csv() method and print the number of contents to be displayed using df.head().

3.Import KMeans and use for loop to cluster the data.

4.Predict the cluster and plot data graphs. 5.Print the outputs and end the program.



## Program:
```
/*
Program to implement the K Means Clustering for Customer Segmentation.
Developed by: NAVEEN M
RegisterNumber:  212225230197
*/
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans

# Read dataset
data = pd.read_csv("Mall_Customers (1).csv")

# Display information
print(data.head())
print(data.info())

# Check missing values
print(data.isnull().sum())

# Elbow Method
wcss = []

X = data.iloc[:, 3:]   # Annual Income and Spending Score

for i in range(1, 11):
    kmeans = KMeans(
        n_clusters=i,
        init="k-means++",
        random_state=42,
        n_init=10
    )

    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

# Plot Elbow graph
plt.figure(figsize=(8,5))
plt.plot(range(1,11), wcss, marker='o')
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.title("Elbow Method")
plt.show()

# Create KMeans model with 5 clusters
km = KMeans(
    n_clusters=5,
    random_state=42,
    n_init=10
)

km.fit(X)

# Predict clusters
y_pred = km.predict(X)

# Add cluster column
data["cluster"] = y_pred

# Separate clusters
df0 = data[data["cluster"] == 0]
df1 = data[data["cluster"] == 1]
df2 = data[data["cluster"] == 2]
df3 = data[data["cluster"] == 3]
df4 = data[data["cluster"] == 4]

# Plot clusters
plt.figure(figsize=(8,6))

plt.scatter(
    df0["Annual Income (k$)"],
    df0["Spending Score (1-100)"],
    c="black",
    label="Cluster 0"
)

plt.scatter(
    df1["Annual Income (k$)"],
    df1["Spending Score (1-100)"],
    c="cyan",
    label="Cluster 1"
)

plt.scatter(
    df2["Annual Income (k$)"],
    df2["Spending Score (1-100)"],
    c="yellow",
    label="Cluster 2"
)

plt.scatter(
    df3["Annual Income (k$)"],
    df3["Spending Score (1-100)"],
    c="blue",
    label="Cluster 3"
)

plt.scatter(
    df4["Annual Income (k$)"],
    df4["Spending Score (1-100)"],
    c="green",
    label="Cluster 4"
)

# Plot cluster centers
plt.scatter(
    km.cluster_centers_[:,0],
    km.cluster_centers_[:,1],
    s=300,
    marker='*',
    label="Centroids"
)

plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1-100)")
plt.title("Customer Segments")
plt.legend()
plt.show()
```

## Output:
<img width="689" height="584" alt="image" src="https://github.com/user-attachments/assets/3cc0eb07-05b0-47f1-abc6-9ae929b78187" />


<img width="934" height="576" alt="image" src="https://github.com/user-attachments/assets/41aad4f3-c184-4bf6-aa73-a0e4315f27e9" />


<img width="953" height="681" alt="image" src="https://github.com/user-attachments/assets/db7732e3-e610-421a-bbad-11f09c24fe21" />


## Result:
Thus the program to implement the K Means Clustering for Customer Segmentation is written and verified using python programming.
