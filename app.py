import pandas as pd
import normalizer
from joblib import dump
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn import metrics
from collections import Counter


path_to_file = "path/to/file.xlsx"
# Test Base definition
df_train = normalizer.df_sample(pd.read_excel(path_to_file))
# Define the relevant features to use in the classification
features_col = ['VL_MENSAGEM', 'DS_INSTITUICAO_CREDITADA', 'ID_REGRA']
feature = df_train[features_col]
# Create the targeted value to be identified
target = df_train.ID_STATUS_MENSAGEM

# Read the information to test the prediction
path_to_test_file = "path/to/test/file.xlsx"
df_test = pd.read_excel(path_to_test_file)
x_test = df_test[features_col]
y_test = df_test.ID_STATUS_MENSAGEM

# Create and train the classifier
clf = DecisionTreeClassifier()
clf = clf.fit(feature, target)
y_pred = clf.predict(x_test)

# Metrics
print("Accuracy:", metrics.accuracy_score(y_test, y_pred))
print("Precision:", metrics.precision_score(y_test, y_pred, average='weighted'))
print("Confusion Matrix:", metrics.confusion_matrix(y_test, y_pred, labels=[2, 3]))

# Print the decision tree to evaluate if the path makes sense
r = export_text(clf, feature_names=features_col)
print(r)

# Save the model tu be used in the future
dump(clf, 'teds_decision_tree.joblib')
