# # This project is all about predicting customer churn using various machine learning models.
# # We will perform exploratory data analysis (EDA), preprocess the data, handle class imbalance, train models, and evaluate their performance.
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, roc_auc_score

# # Step 1: EDA (Exploratory Data Analysis)
# # Loading the dataset
df = pd.read_csv('data_1.csv')

# # Displaying the first few rows of the dataset
print(df.head())

# # Checking for missing values
print(df.isnull().sum())

# # Explore each feature (column) to understand distributions and patterns.
print(df.describe())

# # Visualize churn vs. non-churn (e.g., bar plots, pie charts).
# # Title: Churn Distribution

# Add plot labels and title
plt.pie(df['Churn'].value_counts()[::-1], labels = df['Churn'].unique()[::-1], autopct='%1.1f%%', startangle=270)
plt.title("Churn Distribution")
plt.xlabel("Churn")
plt.ylabel("Count")
# Show the plot
plt.show()

# # Step 2: Data Preprocessing & Feature Engineering.
# # This step includes:
#     # Encoding categorical variables:
# #                                   It means converting categorical variables into numerical format. e.g: Machine learning models can't understand text or labels etc,
# #                                   so we convert them into numbers through tecqniques like Label Encoding or One-Hot Encoding.
#     # Feature scaling (if needed)
#     # Creating or modifying useful features
#     # Preparing data for the model

# # ----------------------------- Encoding Binary Categorical Variables -----------------------------
label_enc = LabelEncoder() # Label encoding is a technique to convert categorical variables into numerical format. e.g: Male = 1, Female = 0

# # Gender: Male/Female
df['Gender'] = label_enc.fit_transform(df['Gender'])  # Male = 1, Female = 0

# # AutoRenew: Yes/No
df['AutoRenew'] = label_enc.fit_transform(df['AutoRenew'])  # Yes = 1, No = 0

# # Churn (Target variable): Yes/No
df['Churn'] = label_enc.fit_transform(df['Churn'])  # Yes = 1, No = 0


# # ----------------------------- One-Hot Encoding Multiclass Categorical Variables -----------------------------
# # One Hot Encoding is a technique to convert categorical variables into a format that can be provided to ML algorithms to do a better job in prediction.
# # This line is applying One-Hot Encoding to the two columns:
# # 'SubscriptionType'
# # 'PaymentMethod'
# # It converts each category in those columns into separate new columns with 0 or 1 values.

df = pd.get_dummies(df, columns=['SubscriptionType', 'PaymentMethod'], drop_first=True) 

# # ----------------------------- Feature Scaling -----------------------------
# # Feature Scaling is the process of standardizing or normalizing numerical features so that they are on the same scale and have similar distributions.
scaler = StandardScaler() #StandardScaler is a function from sklearn.preprocessing that scales numeric data using something called standardization.
# # Standardization transforms your data so that each feature has: Mean = 0 , Standard Deviation = 1
numerical_features = ['Age', 'MonthlyCharges', 'TenureMonths', 'NumSupportTickets']

# # Apply scaling only on numerical columns 
df[numerical_features] = scaler.fit_transform(df[numerical_features])

# # ----------------------------- Spliting Features and Labels -----------------------------
X = df.drop(['CustomerID', 'Churn'], axis=1)  # Remove ID and target column which is Churn
# # X contains all the features except 'CustomerID' and 'Churn'
y = df['Churn']

# # ----------------------------- Train-Test Split -----------------------------
# # Train-test split is a technique to split the dataset into two parts: one for training the model and one for testing its performance.
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# # Checking dimensions
print("Training Features:", X_train.shape)
print("Training Labels:", y_train.shape)
print("Testing Features:", X_test.shape)
print("Testing Labels:", y_test.shape)

# # Step 3: Handling Class Imbalance?
# # Handling Class Imbalance is a technique to deal with situations where one class (e.g., churn) has significantly fewer samples than the other class (e.g., non-churn).
# # Checking class distribution
print("Class distribution in training set before applying SMOTE:")
print(y_train.value_counts())
# # Visualizing the distribution 
sns.countplot(x=y_train)
plt.title("Class Distribution in Training Set Before Applying SMOTE")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()
# # Applying SMOTE to handle class imbalance
# # SMOTE (Synthetic Minority Over-sampling Technique) is a technique to create synthetic samples for the minority class to balance the dataset.
smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

# # Checking the new class distribution after SMOTE (Numeric)
print("Class distribution after applying SMOTE:")
print(y_train_res.value_counts())
# # Visualizing the new distribution 
sns.countplot(x=y_train_res)
plt.title("Class Distribution After Applying SMOTE")
plt.xlabel("Churn")
plt.ylabel("Count")
plt.show()

# # Step 4: Model Training
# # 🧪 Store results here
model_scores = {}

# Define models
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
    # Logistic Regression means A statistical and linear model used for binary classification problems (like churn or not churn). It is mostly used for Binary classification problems.
    "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
    # Random Forest Means A tree-based ensemble model that uses many decision trees to make a final decision. It is modtly used for General usage, and to avoid overfitting.
    "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42),
    # XGBoost means A powerful, optimized implementation of Gradient Boosting, known for high speed and performance. It is mostly used for competitions and to handle complex problems.
    "LightGBM": LGBMClassifier(random_state=42)
    # LightGBM meansA faster and more efficient version of gradient boosting created by Microsoft. It is mostly used to handle Large data, efficient boosting.
}

# Train, predict, and evaluate each model
for name, model in models.items():
    print(f"\n🚀 Training {name}...")
    model.fit(X_train_res, y_train_res)
    y_pred = model.predict(X_test)
    
    print(f"\n📊 Evaluation for {name}:")
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    
    # Store key metrics
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_pred)
    
    model_scores[name] = {
        "Accuracy": acc,
        "F1 Score": f1,
        "ROC AUC": roc
    }

# 🔍 Compare all model results
print("\n📋 Model Comparison Summary:")
results_df = pd.DataFrame(model_scores).T
print(results_df)

# 📊 Optional: Plot comparison
results_df.plot(kind='bar', figsize=(10, 6))
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=45)
plt.grid(axis='y')
plt.tight_layout()
plt.show()


# Step 5: Feature Importance & Interpretation
# ✅ Train Random Forest again (just to ensure it's ready here)
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_res, y_train_res)

# ✅ Get feature names and importance scores
feature_names = X_train.columns  # original features before SMOTE
importances = rf_model.feature_importances_

# ✅ Combine into a DataFrame
feat_importance_df = pd.DataFrame({
    'Feature': feature_names,
    'Importance': importances
})

# ✅ Sort features by importance
feat_importance_df = feat_importance_df.sort_values(by='Importance', ascending=False)

# ✅ Visualize top 10 features
plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Feature', data=feat_importance_df.head(10))
plt.title('Top 10 Important Features for Churn Prediction (Random Forest)')
plt.xlabel('Importance Score')
plt.ylabel('Feature Name')
plt.tight_layout()
plt.show()

# Step 6: Save the Models
# # Saving the trained models for future use
# Save all trained models
for name, model in models.items():
    filename = f'{name.replace(" ", "_").lower()}_model.pkl'
    joblib.dump(model, filename)
    print(f"✅ {name} saved as {filename}")
