import matplotlib.pyplot as plt
from sklearn import model_selection
from sklearn.linear_model import LogisticRegression

from sklearn.metrics import precision_recall_curve
from sklearn.metrics import confusion_matrix
import pandas as  pd
from collections import Counter
import pandas as  pd
from collections import Counter
import numpy as np
from sklearn import tree
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier
import joblib
import xgboost as xgb
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from xgboost.sklearn import XGBClassifier

from sklearn.metrics import average_precision_score

import matplotlib.pyplot as plt
from sklearn.datasets import make_classification

from imblearn.combine import SMOTETomek

from sklearn.model_selection import train_test_split,cross_val_predict
from pandas.plotting import scatter_matrix

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from mlxtend.classifier import StackingClassifier

# load dataset
df = shuffle(pd.read_csv('files/Credit.csv',sep=','))
features = df.iloc[0:len(df),1:12].values
labels = df.iloc[0:len(df):,0].values
bins = 25
#X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.20, random_state=42)

sm2 = SMOTETomek()
'''
An imbalanced dataset is a dataset where the number of examples in each class is not equal. In other words, one or more classes have significantly fewer examples than the others. This can be a problem in machine learning because many algorithms are designed to work best when the classes are balanced.

SMOTETomek is a technique used to address the problem of imbalanced datasets. It is a combination of two techniques: Synthetic Minority Over-sampling Technique (SMOTE) and Tomek links.

SMOTE is a technique that generates synthetic examples of the minority class by interpolating between existing examples. This helps to balance the classes and prevent the algorithm from being biased towards the majority class.

Tomek links, on the other hand, are pairs of examples from different classes that are very close to each other. Removing these pairs can help to improve the separation between the classes and make the algorithm more accurate.

By combining these two techniques, SMOTETomek can help to balance the classes in an imbalanced dataset and improve the accuracy of the algorithm. It works by first applying SMOTE to generate synthetic examples of the minority class, and then applying Tomek links to remove any pairs of examples that are very close to each other.

In the code you provided, SMOTETomek is used to balance the classes in the dataset before training the StackingClassifier. This can help to improve the accuracy of the classifier and prevent it from being biased towards the majority class.


'''
# Apply SMOTETomek to the features and labels to balance the classes
X_res, y_res = sm2.fit_resample(features, labels)

# Print the shape of the resampled dataset
print('Resampled dataset shape {}'.format(Counter(y_res)))

'''
Counter is a subclass of the Python dictionary used to count the occurrences of elements in a list. In this case, we are using it to count the number of occurrences of each class in the resampled dataset.

The output of the Counter function will be a dictionary-like object with the class labels as keys and the number of occurrences as values. For example, if there are 100 examples of class 0 and 200 examples of class 1 in the resampled dataset, the output of Counter(y_res) will be {0: 100, 1: 200}.
'''

# Assign the resampled features and labels to new variables
XX_res = X_res
yy_res = y_res
# Create a decision tree classifier using the Gini index as the criterion for splitting nodes
clf1 = tree.DecisionTreeClassifier(criterion='gini')


# Create an XGBoost classifier with a maximum depth of 6, a learning rate of 0.1, and 100 estimators
clf3 = XGBClassifier(max_depth=6,learning_rate=0.1,n_estimators=100,nthread=-1, objective='binary:logistic')
#clf3= XGBClassifier(max_depth=3, learning_rate=0.1, n_estimators=100, verbosity=1, silent=None, objective='binary:logistic', booster='gbtree', n_jobs=None, nthread=None, gamma=0, min_child_weight=1, max_delta_step=0, subsample=1, colsample_bytree=1, colsample_bylevel=1, colsample_bynode=1, reg_alpha=0, reg_lambda=1, scale_pos_weight=1, base_score=0.5, random_state=None, seed=None, missing=None, **kwargs)
'''
Setting objective='binary:logistic' means that the model will use binary logistic regression as the objective function, which is appropriate for binary classification problems where the goal is to predict a binary outcome (e.g. 0 or 1).

The model will try to minimize the logistic loss, which is a measure of the difference between the predicted probabilities and the true binary labels.
'''

# Create a random forest classifier with a maximum depth of 100, a random state of 7, and the Gini index as the criterion for splitting nodes
clf2= RandomForestClassifier(max_depth=100, random_state=7,criterion="gini")
#clf2= RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None, min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=None, random_state=None, verbose=0, warm_start=False, class_weight=None, ccp_alpha=0.0, max_samples=None)


# Create a logistic regression classifier with a regularization parameter of 10.0
lr = LogisticRegression(C=10.0)

# Create a stacking classifier that combines the predictions of the three base classifiers (clf1, clf2, and clf3) using logistic regression (lr) as the meta-classifier
#clf4 = StackingClassifier(classifiers=[clf1, clf2, clf3], meta_classifier=lr)
clf4 = StackingClassifier(classifiers=[clf1, clf2], meta_classifier=lr)

# Set the random seed for reproducibility
seed = 7

# Create a list of tuples containing the name and instance of each model to be evaluated
models = []
models.append(("Decision_Tree",clf1))
models.append(("Random_Forest",clf2))
#models.append(("XgBoost",clf3))
models.append(("Stacking",clf4))

# Set the scoring metric to be used for evaluation
scoring = 'accuracy'

# Create empty lists to store the results and names of each model
results = []
results_custom_stack = []
names = []

# Evaluate each model using cross-validation
for name, model in models:
    
    # Create a KFold cross-validation object with 10 splits, shuffling the data and using the random seed
    kfold = model_selection.KFold(n_splits=10, shuffle=True, random_state=seed)
    
    # Evaluate the model using cross-validation and store the results
    cv_results = model_selection.cross_val_score(model, X_res, y_res, cv=kfold, scoring=scoring)
    
    # Make predictions on the resampled data using the current model
    y_pred = cross_val_predict(model, X_res, y_res)
    
    # Append the predictions as a new feature to the resampled data
    XX_res = np.append(XX_res, y_pred.reshape(len(y_pred),1),axis=1) 
    
    # Store the results and name of the current model
    results.append(cv_results)
    names.append(name)
    
    # Print the mean and standard deviation of the cross-validation results for the current model
    msg = "%s: %f (%f)" % (name, cv_results.mean(), cv_results.std())
    print(msg)


# Create a boxplot to compare the performance of each model
fig = plt.figure()
fig.suptitle('Algorithm Comparison')
ax = fig.add_subplot(111)
plt.boxplot(results)
ax.set_xticklabels(names)
plt.show()

'''
The standard deviation (std) is an important metric in classification models because it provides a measure of the variability or spread of the cross-validation results. A low standard deviation indicates that the cross-validation results are consistent and reliable, while a high standard deviation indicates that the results are more variable and less reliable.

In the context of classification models, the standard deviation is often used to assess the stability and generalization performance of the model. A low standard deviation indicates that the model is likely to perform well on new, unseen data, while a high standard deviation indicates that the model may be overfitting to the training data and may not generalize well to new data.

Therefore, when evaluating the performance of classification models using cross-validation, it is important to consider both the mean and standard deviation of the results. A model with a high mean accuracy but a high standard deviation may not be as reliable as a model with a slightly lower mean accuracy but a lower standard deviation.
'''