
import numpy as np
from sklearn.datasets import load_iris
from sklearn import tree

data = load_iris()

test_idx = [0,50,100]

#training data
train_target = np.delete(data.target , test_idx)
train_data = np.delete(data.data , test_idx,axis =0 )

#testing data
test_target = data.target[test_idx]
test_data = data.data[test_idx]

clf = tree.DecisionTreeClassifier()
clf.fit(train_data, train_target)

print (test_target)
print (clf.predict(test_data))

from sklearn.externals.six import StringIO
import pydotplus
dot_data = StringIO()
tree.export_graphviz(clf, out_file=dot_data,
                         feature_names=data.feature_names,
                         class_names=data.target_names,
                         filled=True, rounded=True,
                         special_characters=True)
graph = pydotplus.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf("iris.pdf")
