# Purpose: Create a decision tree using the CART algorithm

#Import libraries
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from ucimlrepo import fetch_ucirepo, list_available_datasets
from sklearn.tree import export_graphviz
import pydot

#Create function to convert target to binary
def binary_features(array):
    output = array.copy()
    for i in range(len(array)):
        if 0 < array.num[i]:
            output.num[i] = 1
    return output


# check which datasets can be imported
list_available_datasets()

# import dataset
heart_disease = fetch_ucirepo(id=45)

# access data
X = heart_disease.data.features
y = heart_disease.data.targets
y2 = binary_features(y)

#Create train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y2, random_state=42)

#Set parameters

max_depth = 3
min_samples_split = 2
#Create tree
tree = DecisionTreeClassifier(max_depth=max_depth, min_samples_split=min_samples_split)
tree.fit(X_train, y_train)

#Print accuracy of tree
print("accuracy on training set: %f" % tree.score(X_train, y_train))
print("accuracy on test set: %f" % tree.score(X_test, y_test))

#Create tree.dot file
export_graphviz(tree, out_file="tree.dot", class_names=["no heartfailure", "heartfailure"], feature_names = X.columns, impurity=False, filled=True)

#Convert tree.dot to tree.png
(graph,) = pydot.graph_from_dot_file('tree.dot')
graph.write_png('tree.png')
