# -------------------------------------------
# Decision Tree Classifier on Bank Marketing Dataset
# -------------------------------------------

# Step 1: Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Step 2: Load Dataset (Bank Marketing - GitHub Mirror)
url = "https://raw.githubusercontent.com/selva86/datasets/master/bank-additional-full.csv"
data = pd.read_csv(url, sep=";")
print("Dataset shape:", data.shape)
print(data.head())

# Step 3: Preprocessing
data_encoded = pd.get_dummies(data, drop_first=True)

# Target column (encoded as y_yes)
X = data_encoded.drop("y_yes", axis=1)
y = data_encoded["y_yes"]

# Step 4: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Step 5: Train Model
dtree = DecisionTreeClassifier(max_depth=5, random_state=42)
dtree.fit(X_train, y_train)

# Step 6: Predict & Evaluate
y_pred = dtree.predict(X_test)

print("\nAccuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()

# Step 7: Visualize Decision Tree
plt.figure(figsize=(20,10))
plot_tree(dtree, filled=True, feature_names=X.columns,
          class_names=["No", "Yes"], rounded=True, fontsize=8)
plt.show()
