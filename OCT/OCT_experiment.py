import dataset
import tree as miptree
from sklearn import tree
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo, list_available_datasets

#From tree.py
from collections import namedtuple
import numpy as np
from scipy import stats
import gurobipy as gp
from gurobipy import GRB
from sklearn import tree



#Create function to convert target to binary
def binary_features(array):
    output = array.copy()
    for i in range(len(array)):
        if 0 < array.num[i]:
            output.num[i] = 1
    return output


# Load data and manipulate it
heart_disease = fetch_ucirepo(id=45)
x = heart_disease.data.features
y = heart_disease.data.targets
y = binary_features(y)

# Set arguments
max_depth = 3
min_samples_split = 2
timelimit = 600
seed = 42

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=seed)

#Create tree using mioptree
# oct_tree = miptree.optimalDecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, timelimit=timelimit)
# oct_tree.fit(X_train, y_train)

#Write up the functions in steps
# 1. Create a tree
self = miptree.optimalDecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split, timelimit=timelimit)
# 2. Fit the tree
self.fit(X_train, y_train)


"""
 fit(self, x, y): START
"""
# data size
self.n, self.p = x.shape
print('Training data include {} instances, {} features.'.format(self.n,self.p))

# labels
self.labels = np.unique(y)

# scale data
self.scales = np.max(x, axis=0)
self.scales[self.scales == 0] = 1

# solve MIP
m, a, b, c, d, l = self._buildMIP(x/self.scales, y)

"""
 _buildMIP(self, x, y): START


# # create a model
# m = gp.Model('m')

# # output
# m.Params.outputFlag = self.output
# m.Params.LogToConsole = self.output
# # time limit
# m.Params.timelimit = self.timelimit
# # parallel
# m.params.threads = 0

# # model sense
# m.modelSense = GRB.MINIMIZE

# # variables
# a = m.addVars(self.p, self.b_index, vtype=GRB.BINARY, name='a') # splitting feature
# b = m.addVars(self.b_index, vtype=GRB.CONTINUOUS, name='b') # splitting threshold
# c = m.addVars(self.labels, self.l_index, vtype=GRB.BINARY, name='c') # node prediction
# d = m.addVars(self.b_index, vtype=GRB.BINARY, name='d') # splitting option
# z = m.addVars(self.n, self.l_index, vtype=GRB.BINARY, name='z') # leaf node assignment
# l = m.addVars(self.l_index, vtype=GRB.BINARY, name='l') # leaf node activation
# L = m.addVars(self.l_index, vtype=GRB.CONTINUOUS, name='L') # leaf node misclassified
# M = m.addVars(self.labels, self.l_index, vtype=GRB.CONTINUOUS, name='M') # leaf node samples with label
# N = m.addVars(self.l_index, vtype=GRB.CONTINUOUS, name='N') # leaf node samples

# # calculate baseline accuracy
# baseline = self._calBaseline(y)

# # calculate minimum distance
# min_dis = self._calMinDist(x)

# # objective function
# obj = L.sum() / baseline + self.alpha * d.sum()
# m.setObjective(obj)

# # constraints
# # (20)
# m.addConstrs(L[t] >= N[t] - M[k,t] - self.n * (1 - c[k,t]) for t in self.l_index for k in self.labels)
# # (21)
# m.addConstrs(L[t] <= N[t] - M[k,t] + self.n * c[k,t] for t in self.l_index for k in self.labels)
# # (17)     
# for i in range(self.n):
#     m.addConstrs(gp.quicksum((y.iloc[i] == k) * z[i,t]) == M[k,t] for t in self.l_index for k in self.labels)
# # (16)
# m.addConstrs(z.sum('*', t) == N[t] for t in self.l_index)
# # (18)
# m.addConstrs(c.sum('*', t) == l[t] for t in self.l_index)
# # (13) and (14)
# for t in self.l_index:
#     left = (t % 2 == 0)
#     ta = t // 2
#     while ta != 0:
#         if left:
#             m.addConstrs(gp.quicksum(a[j,ta] * (x.iloc[i,j] + min_dis[j]) for j in range(self.p))
#                             +
#                             (1 + np.max(min_dis)) * (1 - d[ta])
#                             <=
#                             b[ta] + (1 + np.max(min_dis)) * (1 - z[i,t])
#                             for i in range(self.n))
#         else:
#             m.addConstrs(gp.quicksum(a[j,ta] * x.iloc[i,j] for j in range(self.p))
#                             >=
#                             b[ta] - (1 - z[i,t])
#                             for i in range(self.n))
#         left = (ta % 2 == 0)
#         ta //= 2

# # (8)
# m.addConstrs(z.sum(i, '*') == 1 for i in range(self.n))
# # (6)
# m.addConstrs(z[i,t] <= l[t] for t in self.l_index for i in range(self.n))
# # (7)
# m.addConstrs(z.sum('*', t) >= self.min_samples_split * l[t] for t in self.l_index)
# # (2)
# m.addConstrs(a.sum('*', t) == d[t] for t in self.b_index)
# # (3)
# m.addConstrs(b[t] <= d[t] for t in self.b_index)
# # (5)
# m.addConstrs(d[t] <= d[t//2] for t in self.b_index if t != 1)
"""


self._setStart(x, y, a, c, d, l)
m.optimize()
self.optgap = m.MIPGap

# get parameters
self._a = {ind:a[ind].x for ind in a}
self._b = {ind:b[ind].x for ind in b}
self._c = {ind:c[ind].x for ind in c}
self._d = {ind:d[ind].x for ind in d}
"""
 fit(self, x, y): END
"""