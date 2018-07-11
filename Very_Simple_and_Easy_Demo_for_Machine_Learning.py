from sklearn import tree

# Training dataset

# Taking the traing features arbitarily as a set of weight and bumpyness(0 for Bumpy and 1 for Smooth) of a fruit
# to output make output whether it's Apple(0) or Orange(1) 
features = [[140, 1], [130, 1], [150, 0], [170, 0]]
labels = [0, 0, 1, 1]


# Taking Decision Tree Classifier for the sake of classification which currently has no idea about apple and oranges.
# New Joinee should be trained to do desired work. Isn't it? :)
clf = tree.DecisionTreeClassifier()

# Training the classifier with input and output data which are features and labels respectively.
# New joinee is trained to find patterns in data
clf = clf.fit(features, labels)

# Now it's time to check how it has learned. So checking with new input and telling this guy to produce output
# Before running you can predict with the data to get an idea about the pattern and checking whether this new
# joinee is good enough to perform your task.
print(clf.predict([[151, 0]]))