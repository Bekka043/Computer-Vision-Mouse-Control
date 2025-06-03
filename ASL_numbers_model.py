import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# load and split the data
df = pd.read_csv(r"C:\\Users\\bekka\\Desktop\\Y2S2\\AILab\\Project\\Sign language dataset\\ASL_numbers_dataset.csv")
X = df.drop("Label", axis=1)
y = df["Label"]
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.6)

# model = RandomForestClassifier()
# model.fit(X_train, y_train)

model = KNeighborsClassifier(n_neighbors=70)
model.fit(X_train, y_train)


# to save the model
joblib.dump(model, "gesture_model.pkl")

## model evaluation

y_pred = model.predict(X_test)

print (accuracy_score(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

conf_matrix = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(10, 7))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()