# episode 2 script
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

# load dataset and print it
iris = load_iris()
print iris.feature_names
print iris.target_names
print iris.data[0]
print iris.target[0]
for i in range(len(iris.target)):
    print "Example %d: label %s, features %s" % (i, iris.target[i], iris.data[i])

# remove test data and define train data
test_idx = [0, 50, 100]
train_target = np.delete(iris.target, test_idx)
train_data = np.delete(iris.data, test_idx, axis=0)

# test data
test_target = iris.target[test_idx]
test_data = iris.data[test_idx]

# Define classifier
clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

# make sure the test data has one each of the flowers
print test_target
print clf.predict(test_data)

# Do some visualization for the tree
from sklearn.externals.six import StringIO
import pydot
from IPython.display import Image
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=iris.feature_names,
                         class_names=iris.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
#Image(graph.create_png())
