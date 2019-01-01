
# coding: utf-8

# Problem Statement
# 
# In this assignment students have to transform iris data into 3 dimensions and plot a 3d
# chart with transformed dimensions and color each data point with specific class.

# In[1]:


##Import libraries##
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from sklearn import decomposition
from sklearn import datasets

import seaborn as sns
from sklearn.decomposition import PCA


# In[4]:


##Load Data##

iris = datasets.load_iris()
X = iris.data
y = iris.target
print("Number of samples:")

print(X.shape[0])
print('Number of features :')
print(X.shape[1])
print("Feature names:")
print(iris.feature_names)


# In[5]:


##Feature scaling##

from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_scaled = sc.fit_transform(X)
print('shape of scaled data points:')
print(X_scaled.shape)
print('first 5 rows of scaled data points :')
print(X_scaled[:5,:])


# In[6]:


##Variance as a function of components##

sns.set()
pca = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('number of components')
plt.ylabel('cumulative explained variance')
plt.show()


# In[7]:


##PCA using Eigen-decomposition: 5-step process##

#Normalize columns of A so that each feature has zero mean
A0 = iris.data
mu = np.mean(A0,axis=0)
A = A0 - mu
print("Does A have zero mean across rows?")
print(np.mean(A,axis=0))
print('Mean value : ')
print(mu)
print('Standardized Feature value first 5 rows: ')
print(A[:5,:])

# Compute sample covariance matrix Sigma = {A^TA}/{(m-1)}covariance matrix can also be computed 
# using np.cov(A.T)
m,n = A.shape
Sigma = (A.T @ A)/(m-1)
print("Sigma:")
print(Sigma)

#Perform eigen-decomposition of Sigma using `np.linalg.eig(Sigma)`
W,V = np.linalg.eig(Sigma)
print("Eigen values:")
print(W)
print("Eigen vectors:")
print(V)

#Compress by ordering 3 eigen vectors according to largest eigen values and compute AX_k
print("Compressed - 4D to 3D:")
print('First 3 eigen vectors :')
print(V[:,:3] )
Acomp = A @ V[:,:3] 
print('First first five rows of transformed features :')
print(Acomp[:5,:]) 


# 5. Reconstruct from compressed version by computing $A V_k V_k^T$
print("Reconstructed version - 3D to 4D:")
Arec = A @ V[:,:3] @ V[:,:3].T # first 3 vectors
print(Arec[:5,:]+mu) # first 5 obs, adding mu to compare to original


# In[8]:


##Original iris feature values##

iris.data[:5,:]


# In[10]:


###Visualization##

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)
y= iris.target
plt.cla()
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(Acomp[y == label, 0].mean(),
              Acomp[y == label, 1].mean() + 1.5,
              Acomp[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(Acomp[:, 0], Acomp[:, 1], Acomp[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()


# In[11]:


##Applying PCA for number of compents = 3 using sklearn##

pca = PCA(n_components=3)
pca.fit(X_scaled)
print('explained variance :')
print(pca.explained_variance_)
print('PCA Components : ')
print(pca.components_)
X_transformed = pca.transform(X)
print('Transformed Feature values first five rows :')
print(X_transformed[:5,:])
print('Transformed Feature shape :')
print(X_transformed.shape)
print('Original Feature shape :')
print(X.shape)
print('Retransformed  Feature  :')
X_retransformed = pca.inverse_transform(X_transformed)
print('Retransformed Feature values first five rows :')
print(X_retransformed[:5,:])


# In[12]:


print('First Principal Component PC1: ',pca.components_[0])
print('\nSecond Principal Component PC2: ',pca.components_[1])
print('\nThird Principal Component PC3: ',pca.components_[2])


# In[13]:


##Visualization##

np.random.seed(5)

centers = [[1, 1], [-1, -1], [1, -1]]
fig = plt.figure(1, figsize=(8, 6))
plt.clf()
ax = Axes3D(fig, rect=[0, 0, 1, 1], elev=48, azim=134)
y= iris.target
plt.cla()
for name, label in [('Setosa', 0), ('Versicolour', 1), ('Virginica', 2)]:
    ax.text3D(X_transformed[y == label, 0].mean(),
              X_transformed[y == label, 1].mean() + 1.5,
              X_transformed[y == label, 2].mean(), name,
              horizontalalignment='center',
              bbox=dict(alpha=.5, edgecolor='w', facecolor='w'))
# Reorder the labels to have colors matching the cluster results
y = np.choose(y, [1, 2, 0]).astype(np.float)
ax.scatter(X_transformed[:, 0], X_transformed[:, 1], X_transformed[:, 2], c=y, cmap=plt.cm.nipy_spectral,
           edgecolor='k')

ax.w_xaxis.set_ticklabels([])
ax.w_yaxis.set_ticklabels([])
ax.w_zaxis.set_ticklabels([])

plt.show()

