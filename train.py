import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import SMOTE
import logging

logging.basicConfig(level=logging.DEBUG)

# Load dataset
df = pd.read_csv(r"E:\Projects\Smart Health Predictor\Smart-Health-Predictor\dataset\Symptom-severity.csv")

logging.debug(f"Master dataset shape: {df.shape}")

# Drop very rare classes (with <=5 samples)
label_col = "disease"  # replace with your label column
counts = df[label_col].value_counts()
rare_classes = counts[counts <= 5].index
df = df[~df[label_col].isin(rare_classes)]
logging.debug(f"Dataset shape after removing rare classes: {df.shape}")

# Separate features and target
X = df.drop(columns=[label_col])
y = df[label_col]

# Encode target labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)
logging.debug(f"Encoded label '{label_col}' with classes: {le.classes_}")

# Apply SMOTE safely
# Set k_neighbors=1 to handle classes with very few samples
smote = SMOTE(random_state=42, k_neighbors=1)
X_res, y_res = smote.fit_resample(X, y_encoded)
logging.debug(f"Dataset shape after SMOTE: {X_res.shape}")

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(
    X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
)

logging.debug(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")

# Now you can continue with your model training...
