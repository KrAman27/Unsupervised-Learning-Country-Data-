# -*- coding: utf-8 -*-
"""
Created on Tue Jul 28 10:01:21 2020

@author: aman kumar
"""
"""Clustering the Countries by using Unsupervised Learning
for HELP International
Objective:
To categorise the countries using socio-economic and health factors that determine the overall
development of the country.
About organization:
HELP International is an international humanitarian NGO that is committed to fighting poverty and
providing the people of backward countries with basic amenities and relief during the time of
disasters and natural calamities.
Problem Statement:
HELP International have been able to raise around $ 10 million. Now the CEO of the NGO needs to
decide how to use this money strategically and effectively. So, CEO has to make decision to choose
the countries that are in the direst need of aid. Hence, your Job as a Data scientist is to categorise
the countries using some socio-economic and health factors that determine the overall development
of the country. Then you need to suggest the countries which the CEO needs to focus on the most."""

#importing the libraries
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

#importing the dataset
dataset = pd.read_csv('Country-data.csv')


"""
Clustering by taking CHILD_MORT feature

"""
#importing the required dataset
X = dataset.iloc[:,[1,9]].values

#finding the optimal number of clusters required using the dendogram
import scipy.cluster.hierarchy as shc
dendro_CM = shc.dendrogram(shc.linkage(X,method='ward'))
plt.title('Dendrogram_Child_Mort')
plt.xlabel('Countries')
plt.ylabel('Euclidean Distances')
plt.show()

#Applying Hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc_CM = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_pred_CM = hc_CM.fit_predict(X)

#Vsualising the clusters
plt.scatter(X[y_pred_CM==0,0],X[y_pred_CM==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_pred_CM==1,0],X[y_pred_CM==1,1],s=100,c='magenta',label='Cluster 2')
plt.scatter(X[y_pred_CM==2,0],X[y_pred_CM==2,1],s=100,c='blue',label='Cluster 3')
plt.scatter(X[y_pred_CM==3,0],X[y_pred_CM==3,1],s=100,c='green',label='Cluster 4')
plt.title('Clusters of Countries')
plt.xlabel('Child_Mort')
plt.ylabel('GDPP')
plt.legend()
plt.show()


"""
Clustering by using EXPORTS feature

"""
#importing the required dataset
X = dataset.iloc[:,[2,9]].values

#finding the optimal number of clusters required using the dendogram
import scipy.cluster.hierarchy as shc
dendro_exp = shc.dendrogram(shc.linkage(X,method='ward'))
plt.title('Dendrogram_Exports')
plt.xlabel('Countries')
plt.ylabel('Euclidean Distances')
plt.show()

#Applying Hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc_exp = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_pred_exp = hc_exp.fit_predict(X)

#Vsualising the clusters
plt.scatter(X[y_pred_exp==0,0],X[y_pred_exp==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_pred_exp==1,0],X[y_pred_exp==1,1],s=100,c='magenta',label='Cluster 2')
plt.scatter(X[y_pred_exp==2,0],X[y_pred_exp==2,1],s=100,c='blue',label='Cluster 3')
plt.scatter(X[y_pred_exp==3,0],X[y_pred_exp==3,1],s=100,c='green',label='Cluster 4')
plt.title('Clusters of Countries')
plt.xlabel('Exports')
plt.ylabel('GDPP')
plt.legend()
plt.show()

"""
Clustering by using HEALTH feature

"""
#importing the required dataset
X = dataset.iloc[:,[3,9]].values

#finding the optimal number of clusters required using the dendogram
import scipy.cluster.hierarchy as shc
dendro_health = shc.dendrogram(shc.linkage(X,method='ward'))
plt.title('Dendrogram_Health')
plt.xlabel('Countries')
plt.ylabel('Euclidean Distances')
plt.show()

#Applying Hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc_health = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_pred_health = hc_health.fit_predict(X)

#Vsualising the clusters
plt.scatter(X[y_pred_health==0,0],X[y_pred_health==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_pred_health==1,0],X[y_pred_health==1,1],s=100,c='magenta',label='Cluster 2')
plt.scatter(X[y_pred_health==2,0],X[y_pred_health==2,1],s=100,c='blue',label='Cluster 3')
plt.scatter(X[y_pred_health==3,0],X[y_pred_health==3,1],s=100,c='green',label='Cluster 4')
plt.title('Clusters of Countries')
plt.xlabel('Health')
plt.ylabel('GDPP')
plt.legend()
plt.show()

"""
Clustering by using IMPORTS feature

"""
#importing the required dataset
X = dataset.iloc[:,[4,9]].values

#finding the optimal number of clusters required using the dendogram
import scipy.cluster.hierarchy as shc
dendro_imp = shc.dendrogram(shc.linkage(X,method='ward'))
plt.title('Dendrogram_Imports')
plt.xlabel('Countries')
plt.ylabel('Euclidean Distances')
plt.show()

#Applying Hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc_imp = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_pred_imp = hc_imp.fit_predict(X)

#Vsualising the clusters
plt.scatter(X[y_pred_imp==0,0],X[y_pred_imp==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_pred_imp==1,0],X[y_pred_imp==1,1],s=100,c='magenta',label='Cluster 2')
plt.scatter(X[y_pred_imp==2,0],X[y_pred_imp==2,1],s=100,c='blue',label='Cluster 3')
plt.scatter(X[y_pred_imp==3,0],X[y_pred_imp==3,1],s=100,c='green',label='Cluster 4')
plt.title('Clusters of Countries')
plt.xlabel('Imports')
plt.ylabel('GDPP')
plt.legend()
plt.show()

"""
Clustering by using INCOME feature

"""
#importing the required dataset
X = dataset.iloc[:,[5,9]].values

#finding the optimal number of clusters required using the dendogram
import scipy.cluster.hierarchy as shc
dendro_income = shc.dendrogram(shc.linkage(X,method='ward'))
plt.title('Dendrogram_Income')
plt.xlabel('Countries')
plt.ylabel('Euclidean Distances')
plt.show()

#Applying Hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc_income = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_pred_income = hc_income.fit_predict(X)

#Vsualising the clusters
plt.scatter(X[y_pred_income==0,0],X[y_pred_income==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_pred_income==1,0],X[y_pred_income==1,1],s=100,c='magenta',label='Cluster 2')
plt.scatter(X[y_pred_income==2,0],X[y_pred_income==2,1],s=100,c='blue',label='Cluster 3')
plt.scatter(X[y_pred_income==3,0],X[y_pred_income==3,1],s=100,c='green',label='Cluster 4')
plt.title('Clusters of Countries')
plt.xlabel('Income')
plt.ylabel('GDPP')
plt.legend()
plt.show()

"""
Clustering by using INFLATION feature

"""
#importing the required dataset
X = dataset.iloc[:,[6,9]].values

#finding the optimal number of clusters required using the dendogram
import scipy.cluster.hierarchy as shc
dendro_inflation = shc.dendrogram(shc.linkage(X,method='ward'))
plt.title('Dendrogram_inflation')
plt.xlabel('Countries')
plt.ylabel('Euclidean Distances')
plt.show()

#Applying Hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc_inflation = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_pred_inflation = hc_inflation.fit_predict(X)

#Vsualising the clusters
plt.scatter(X[y_pred_inflation==0,0],X[y_pred_inflation==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_pred_inflation==1,0],X[y_pred_inflation==1,1],s=100,c='magenta',label='Cluster 2')
plt.scatter(X[y_pred_inflation==2,0],X[y_pred_inflation==2,1],s=100,c='blue',label='Cluster 3')
plt.scatter(X[y_pred_inflation==3,0],X[y_pred_inflation==3,1],s=100,c='green',label='Cluster 4')
plt.title('Clusters of Countries')
plt.xlabel('Inflation')
plt.ylabel('GDPP')
plt.legend()
plt.show()

"""
Clustering by using LIFE_EXPECTANCY feature

"""
#importing the required dataset
X = dataset.iloc[:,[7,9]].values

#finding the optimal number of clusters required using the dendogram
import scipy.cluster.hierarchy as shc
dendro_life_exp = shc.dendrogram(shc.linkage(X,method='ward'))
plt.title('Dendrogram_life_expectancy')
plt.xlabel('Countries')
plt.ylabel('Euclidean Distances')
plt.show()

#Applying Hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc_life_exp = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_pred_life_exp = hc_life_exp.fit_predict(X)

#Vsualising the clusters
plt.scatter(X[y_pred_life_exp==0,0],X[y_pred_life_exp==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_pred_life_exp==1,0],X[y_pred_life_exp==1,1],s=100,c='magenta',label='Cluster 2')
plt.scatter(X[y_pred_life_exp==2,0],X[y_pred_life_exp==2,1],s=100,c='blue',label='Cluster 3')
plt.scatter(X[y_pred_life_exp==3,0],X[y_pred_life_exp==3,1],s=100,c='green',label='Cluster 4')
plt.title('Clusters of Countries')
plt.xlabel('Life_Expectancy')
plt.ylabel('GDPP')
plt.legend()
plt.show()

"""
Clustering by using TOTAL FERTILITY feature

"""
#importing the required dataset
X = dataset.iloc[:,[8,9]].values

#finding the optimal number of clusters required using the dendogram
import scipy.cluster.hierarchy as shc
dendro_fertility = shc.dendrogram(shc.linkage(X,method='ward'))
plt.title('Dendrogram_Total_Fertilty')
plt.xlabel('Countries')
plt.ylabel('Euclidean Distances')
plt.show()

#Applying Hierarchical clustering to the dataset
from sklearn.cluster import AgglomerativeClustering
hc_fertility = AgglomerativeClustering(n_clusters=4, affinity='euclidean', linkage='ward')
y_pred_fertility = hc_fertility.fit_predict(X)

#Vsualising the clusters
plt.scatter(X[y_pred_fertility==0,0],X[y_pred_fertility==0,1],s=100,c='red',label='Cluster 1')
plt.scatter(X[y_pred_fertility==1,0],X[y_pred_fertility==1,1],s=100,c='magenta',label='Cluster 2')
plt.scatter(X[y_pred_fertility==2,0],X[y_pred_fertility==2,1],s=100,c='blue',label='Cluster 3')
plt.scatter(X[y_pred_fertility==3,0],X[y_pred_fertility==3,1],s=100,c='green',label='Cluster 4')
plt.title('Clusters of Countries')
plt.xlabel('Total_Fetility')
plt.ylabel('GDPP')
plt.legend()
plt.show()

