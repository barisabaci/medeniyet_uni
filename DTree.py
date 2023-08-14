'''
Credit Scoring Module
'''
import pandas as pd
from sklearn import metrics, tree
from sklearn.utils import shuffle
from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sn
import matplotlib.colors as mcolors
from sklearn.preprocessing import LabelEncoder


def header(df):
    '''
    Prints the header of the DataFrame.
    
    Parameters:
    df (DataFrame): The DataFrame for which the header is to be printed.
    '''
    print(df.head())

# Read the data and shuffle it
data = shuffle(pd.read_csv('files/Bank.csv', sep=';'))

# Apply label encoding to the data
data = data.apply(LabelEncoder().fit_transform)
'''
LabelEncoder is a class in scikit-learn that is used to convert categorical data, represented as strings or integers, into numerical labels. It assigns a unique integer value to each category in the data, with the smallest integer assigned to the most frequently occurring category.

LabelEncoder is useful in machine learning because many algorithms require numerical input data. By converting categorical data into numerical labels, we can use these algorithms to analyze and make predictions on the data.

For example, in the code you provided, LabelEncoder is used to convert the categorical data in the data DataFrame into numerical labels. This allows the DecisionTreeClassifier algorithm to be trained on the data and make predictions on new data.

Note that LabelEncoder should only be used for nominal or ordinal data, where the categories have no inherent order or relationship. For data with an inherent order or relationship, such as dates or temperature values, other encoding techniques such as one-hot encoding or ordinal encoding may be more appropriate.
'''
# Print the header of the data
header(data)

# Size of the data
number = len(data)

# Split the data into train and test sets
trainSize = int(number * 0.8)
testSize = number - trainSize
train = data.iloc[0:trainSize, 0:16]
test = data.iloc[trainSize:number, 0:16]
train_target = data.iloc[0:trainSize, 16:17]
test_target = data.iloc[trainSize:number, 16:17]
# Decision Tree


def run_model(_tree,_clf):



      # Fit the decision tree classifier to the training data and make predictions on the test data
      _tree.fit(train, train_target)
      predictedTree = _tree.predict(test)

      # Print the feature importances of the decision tree classifier
      print(_tree.feature_importances_)

      # Print the classification report and confusion matrix for the decision tree classifier
      print("Classification report for classifier %s:\n%s\n"
            % (_tree, metrics.classification_report(test_target, predictedTree)))
      print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_target, predictedTree))

      # Create a pandas DataFrame to store the confusion matrix for the decision tree classifier
      df_cm = pd.DataFrame(metrics.confusion_matrix(test_target, predictedTree), range(2), range(2))

      # Fit the random forest classifier to the training data and make predictions on the test data
      _clf.fit(train, train_target)
      predictForest = _clf.predict(test)

      # Print the feature importances of the random forest classifier
      print("----------------Feature Importance-----------------")

      print(_clf.feature_importances_)

      # Print the classification report and confusion matrix for the random forest classifier
      print("Classification report for classifier %s:\n%s\n"
            % (_clf, metrics.classification_report(test_target, predictForest)))
      print("Confusion matrix:\n%s" % metrics.confusion_matrix(test_target, predictForest))

      # Create a pandas DataFrame to store the confusion matrix for the random forest classifier
      df_rm = pd.DataFrame(metrics.confusion_matrix(test_target, predictForest), range(2), range(2))

      # Set the font size for the heatmap labels
      sn.set(font_scale=1)

      # Create a figure with two subplots to display the confusion matrices for the two classifiers
      fig, axes = plt.subplots(ncols=2)

      # Create a colormap for the heatmap
      cmap = mcolors.LinearSegmentedColormap.from_list("n", ['#000066', '#000099', '#0000cc', '#1a1aff', '#6666ff', '#b3b3ff',
                                                            '#ffff00', '#ffcccc', '#ff9999', '#ff6666', '#ff3333',
                                                            '#ff0000'])

      # Create a heatmap of the confusion matrix for the decision tree classifier and add it to the first subplot
      sn.heatmap(df_cm, ax=axes[0], cmap=cmap, annot=True, annot_kws={"size": 12}, fmt='g')

      # Create a heatmap of the confusion matrix for the random forest classifier and add it to the second subplot
      sn.heatmap(df_rm, ax=axes[1], cmap=cmap, annot=True, annot_kws={"size": 12}, fmt='g')

      # Set the titles for the subplots
      axes[0].set_title('Decision Tree')
      axes[1].set_title('Random Forest')

      # Display the figure
      plt.show()
      
if __name__ == '__main__':
      rf = RandomForestClassifier(max_depth=100, random_state=7)
      
      treeClassifier = tree.DecisionTreeClassifier(criterion='gini')

      '''
      The Gini index is a measure of impurity or randomness used in decision tree algorithms to determine the optimal split for a given node. It measures the probability of misclassifying a randomly chosen element in the dataset if it were randomly labeled according to the class distribution in the node.

      In the context of decision trees, the Gini index is used as a criterion to evaluate the quality of a split. The Gini index ranges from 0 to 1, with 0 indicating a perfectly pure node (all elements belong to the same class) and 1 indicating a perfectly impure node (elements are evenly distributed across all classes).

      When building a decision tree, the algorithm tries to minimize the Gini index at each node by selecting the split that results in the lowest weighted sum of the Gini indices of the child nodes. This process is repeated recursively until all nodes are pure or a stopping criterion is met.
      '''
      run_model(treeClassifier,rf)
      
      rf2 = RandomForestClassifier(max_depth=100, random_state=7)
      
      treeClassifier2 = tree.DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=100, min_samples_split=7, min_samples_leaf=2)
      
      run_model(treeClassifier2,rf2)      
            