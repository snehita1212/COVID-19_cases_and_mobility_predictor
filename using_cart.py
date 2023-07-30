# -*- coding: utf-8 -*-

#Importing essential libraries
import pandas as pd
import numpy as np

df1 = pd.read_csv('/content/changes-visitors-covid_final - changes-visitors-covid_final.csv')
df2 = pd.read_csv('/content/covid-data - covid-data.csv')
#files read from google drive

"""**DATA PREPROCESSING**"""

india_df1 = df1[df1['Entity'] == "India"]
india_df2 = df2[df2['iso_code'] =='IND']
#since we only want to extract results about India

india_df1.reset_index(inplace = True)
india_df2.reset_index(inplace = True)
# to give proper indexing like 1,2,3... bcs the initial indices are shuffled

india_df1.drop(['index'], axis=1, inplace = True)
india_df2.drop(['index'], axis=1, inplace = True)
#removing column with shuffled index numbers

india_df2.drop(india_df2.columns[6:], axis=1, inplace = True)
#dropping columns from no 6 to end as they are not required

india_df2 = india_df2[18: -2]
india_df2.reset_index(inplace = True)

#Creating a new dataframe
final_df = pd.concat([india_df1, india_df2['new_cases']], axis=1, join='inner')

final_df = final_df.iloc[:, 3:]

#Normalization
for i in final_df.columns:
  final_df[i] = (final_df[i] - final_df[i].min())/(final_df[i].max() - final_df[i].min())

"""**CART**"""

class Node:
    def __init__(self, predicted_val, num_samples):
        self.num_samples = num_samples
        self.predicted_val = predicted_val
        self.feature_index = 0
        self.threshold = 0
        self.left = None
        self.right = None

def standard_deviation(target):
    avg = np.average(target)
    sd = 0
    for i in target:
        sd += np.square((i-avg))
    sd = np.sqrt(sd/len(target))
    return sd

class CART:
    def __init__(self, max_depth = None):
        self.max_depth = max_depth

    def fit(self, x, y):
        self.n_features_ = x.shape[1]
        self.tree = self.grow_tree(x,y)

    def grow_tree(self, X, y, depth=0):
        predicted_val = np.mean(y)
        node = Node(predicted_val = predicted_val, num_samples = y.size)

        if depth < self.max_depth:
            idx, thr = self._best_split(X, y)
            if idx is not None:
                indices_left = X.iloc[:, idx] < thr
                X_left, y_left = X[indices_left], y[indices_left]
                X_right, y_right = X[~indices_left], y[~indices_left]
                node.feature_index = int(idx)
                node.threshold = float(thr)
                node.left = self.grow_tree(X_left, y_left, depth + 1)
                node.right = self.grow_tree(X_right, y_right, depth + 1)

        return node

    def predict(self, X, pred=[]):
        for inputs in X:
            self._predict(inputs, pred = pred)
        return pred

    def _predict(self, inputs, pred):
        node = self.tree
        while node.left:
            if inputs[node.feature_index] < node.threshold:
                node = node.left
            else:
                node = node.right
        pred.append(node.predicted_val)

    def _best_split(self, X, y):
        # Need at least two elements to split a node.
        m = y.size
        if m <= 1:
            return None, None

        # Count of each class in the current node.
        sd_parent = standard_deviation(y)

        # Gini of current node.
        best_sd = sd_parent
        best_idx, best_thr = None, None

        # Loop through all features.
        for idx in range(self.n_features_):
            # Sort data along selected feature.
            thresholds, Val = zip(*sorted(zip(X.iloc[:, idx], y)))
            sd_left = [0]
            sd_right = Val
            sd_right = sd_right[1:]
            for i in range(1,m):
                split = (thresholds[i]+thresholds[i-1])/2

                sd_l = standard_deviation(sd_left)
                sd_r = standard_deviation(sd_right)

                sd_left.append(sd_right[0])
                sd_right = sd_right[1:]

                child_sd = (i*sd_l +(m-i)*sd_r)/m

                if thresholds[i] == thresholds[i - 1]:
                    continue
                if child_sd < best_sd:
                    best_sd=child_sd
                    best_idx=idx
                    best_thr=split

        return best_idx, best_thr

"""**Creating Training and Testing Data Sets**

**c) Using all mobilities to predict new cases**
"""

df_2 = final_df.copy()
train = df_2.sample(frac=0.80,axis='rows')
test = df_2.sample(frac=0.2,axis='rows')
train = train.reset_index()
test = test.reset_index()
train.drop(['index'],axis=1,inplace=True)
test.drop(['index'],axis=1,inplace=True)

X_train = train.iloc[:,:-1]
X_test = test.iloc[:,:-1]

y_train = train.iloc[:,-1]
y_test = test.iloc[:,-1]

model = CART(50)

test = []
# Iterate over each row
for index, rows in X_test.iterrows():
    # Create list for the current row
    my_list = [rows.retail_and_recreation, rows.grocery_and_pharmacy, rows.residential, rows.transit_stations, rows.parks, rows.workplaces]
    # append the list to the final list
    test.append(my_list)

model.fit(X_train,y_train)
Predictions = model.predict(test)
Predictions

RMSE = np.sqrt(np.sum(((y_test-Predictions)**2)/len(y_test)))  #Room Mean Squared Error
RMSE

MSE = np.sum(((y_test-Predictions)**2)/len(y_test))  #Mean Squared Error
MSE
