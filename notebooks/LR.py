import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PolynomialFeatures
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

#Data loading and preprocessing
current_dir = os.path.dirname(os.path.abspath(__file__))

train_df = pd.read_csv(os.path.join(current_dir, 'train_data.csv'))
val_df = pd.read_csv(os.path.join(current_dir, 'val_data.csv'))
test_df = pd.read_csv(os.path.join(current_dir, 'test_data.csv'))

target = 'Attack Type'

# Text-labels for number encoding
le = LabelEncoder()
y_train = le.fit_transform(train_df[target])
y_val = le.transform(val_df[target])
y_test = le.transform(test_df[target])

X_train = train_df.drop(target, axis=1)
X_val = val_df.drop(target, axis=1)
X_test = test_df.drop(target, axis=1)



# Build the pipeline with the specified steps
pipeline = ImbPipeline([
    ('scaler', StandardScaler()),
    ('poly', PolynomialFeatures()),
    ('dim_red', PCA()), 
    ('smote', SMOTE(random_state=42)),
    ('classifier', LogisticRegression(max_iter=3000, solver='saga', tol=0.01))
])

# Set up the parameter grid for GridSearchCV
param_grid = {
    'scaler': [StandardScaler(), MinMaxScaler()],
    'poly__degree': [1, 2],
    'dim_red': [PCA(n_components=5), 'passthrough'], # LDA left out due to class imbalance and potential issues with the number of components
    'smote': [SMOTE(random_state=42), 'passthrough'],
    'classifier__C': [0.1, 1, 10]
}

print("Aloitetaan GridSearchCV...")
grid_search = GridSearchCV(pipeline, param_grid, cv=3, scoring='accuracy', n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

#Result summary
print(f"\nParhaat parametrit: {grid_search.best_params_}")

#Test validation data with the best model
y_val_pred = grid_search.predict(X_val)
print("\nValidaatiodatan tulokset:")
print(classification_report(y_val, y_val_pred, target_names=le.classes_, zero_division=0))

best_model = grid_search.best_estimator_
y_test_pred = best_model.predict(X_test)

print("\nFinal results (LR)")
print(classification_report(y_test, y_test_pred, target_names=le.classes_, zero_division=0))

#VISUALISATION - Confusion Matrix
import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

fig, ax = plt.subplots(figsize=(10, 8))
ConfusionMatrixDisplay.from_predictions(
    y_test, 
    y_test_pred, 
    display_labels=le.classes_, 
    xticks_rotation='vertical',
    ax=ax,
    cmap='Greens'
)
plt.title("Logistic Regression: Lopullinen testi (Confusion Matrix)")
plt.tight_layout()
plt.show()