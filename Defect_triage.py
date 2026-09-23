# ==========================================================
# AUTOMOTIVE DEFECT TRIAGE - CATBOOST VERSION
# Predict Suggested_Team
# ==========================================================

import pandas as pd
import numpy as np
import joblib
import logging

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from catboost import CatBoostClassifier
from sentence_transformers import SentenceTransformer

import os

# Remove previous log

if os.path.exists("vehicle.log"):
    os.remove("vehicle.log")

# Create fresh log

logging.basicConfig(
    filename="vehicle.log",
    filemode="w",
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

# ==========================================================
# LOAD DATA
# ==========================================================

df = pd.read_excel("automotive_defect_triage_balanced.xlsx")
print(df.shape)

logging.info(
    f"Dataset Loaded | Shape={df.shape}"
)

print("Dataset Shape:", df.shape)


embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)
# ==========================================================
# REMOVE NULL VALUES
# ==========================================================

required_cols = [
    "Title",
    "Description",
    "Module",
    "Subsystem",
    "Severity",
    "Priority",
    "Vehicle_Model",
    "Variant",
    "Software_Version",
    "Suggested_Team"
]

df = df.dropna(subset=required_cols)

# ==========================================================
# FEATURE ENGINEERING
# ==========================================================

df["Issue_Text"] = (
    df["Title"].astype(str)
    + " "
    + df["Description"].astype(str)
)

print("Generating embeddings...")

embeddings = embedding_model.encode(
    df["Issue_Text"].tolist(),
    show_progress_bar=True
)

embedding_df = pd.DataFrame(
    embeddings,
    columns=[
        f"emb_{i}"
        for i in range(
            embeddings.shape[1]
        )
    ]
)

df["Module_Subsystem"] = (
    df["Module"].astype(str)
    + "_" +
    df["Subsystem"].astype(str)
)

df["Severity_Priority"] = (
    df["Severity"].astype(str)
    + "_" +
    df["Priority"].astype(str)
)

df["Version_Module"] = (
    df["Software_Version"].astype(str)
    + "_" +
    df["Module"].astype(str)
)

# ==========================================================
# FEATURES
# ==========================================================

FEATURES = [
    "Issue_Text",
    "Module",
    "Subsystem",
    "Module_Subsystem",
    "Severity",
    "Priority",
    "Severity_Priority",
    "Vehicle_Model",
    "Variant",
    "Software_Version",
    "Version_Module"
]

TARGET = "Suggested_Team"

structured_features = [
    "Module",
    "Subsystem",
    "Module_Subsystem",
    "Severity",
    "Priority",
    "Severity_Priority",
    "Vehicle_Model",
    "Variant",
    "Software_Version",
    "Version_Module"
]

X = pd.concat(
    [
        embedding_df,
        df[structured_features].reset_index(drop=True)
    ],
    axis=1
)

# ==========================================================
# LABEL ENCODING
# ==========================================================

label_encoder = LabelEncoder()

y = label_encoder.fit_transform(df[TARGET])

logging.info(
    f"Classes={list(label_encoder.classes_)}"
)

print("\nTarget Classes:")
print(label_encoder.classes_)

# ==========================================================
# SPLIT
# ==========================================================

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    stratify=y,
    random_state=42
)

print("Train Samples :", len(X_train))
print("Test Samples :", len(X_test))

logging.info(
    f"Train Shape={X_train.shape}"
)

logging.info(
    f"Test Shape={X_test.shape}"
)
print(X_train.shape)

# ==========================================================
# CATBOOST CATEGORICAL FEATURES
# ==========================================================

cat_features = list(
    range(
        embedding_df.shape[1],
        X.shape[1]
    )
)

# ==========================================================
# TRAIN CATBOOST
# ==========================================================

model = CatBoostClassifier(
    iterations=2000,
    depth=10,
    learning_rate=0.03,
    loss_function="MultiClass",
    eval_metric="Accuracy",
    random_seed=42,
    verbose=100,
    auto_class_weights="Balanced",
    early_stopping_rounds=100
)


logging.info("Training Started...")

try:

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        cat_features=cat_features
    )

    logging.info(
        "Model Training Successful"
    )

except Exception as e:

    logging.exception(
        f"Training Failed: {e}"
    )

# ==========================================================
# PREDICTIONS
# ==========================================================

predictions = model.predict(X_test)

predictions = predictions.flatten().astype(int)

probabilities = model.predict_proba(X_test)

confidence_scores = probabilities.max(axis=1)

# ==========================================================
# EVALUATION
# ==========================================================

accuracy = accuracy_score(
    y_test,
    predictions
)

print("\n================================================")
print("ACCURACY")
print("================================================")

logging.info(
    f"Accuracy : {accuracy:.4f}"
)


print("\n================================================")
print("CLASSIFICATION REPORT")
print("================================================")

print(
    classification_report(
        y_test,
        predictions,
        target_names=label_encoder.classes_
    )
)

print("\n================================================")
print("CONFUSION MATRIX")
print("================================================")

print(
    confusion_matrix(
        y_test,
        predictions
    )
)

# ==========================================================
# ACTUAL VS PREDICTED
# ==========================================================

comparison = pd.DataFrame({

    "Actual_Team":
        label_encoder.inverse_transform(y_test),

    "Predicted_Team":
        label_encoder.inverse_transform(predictions),

    "Confidence":
        np.round(confidence_scores, 4)

})

comparison["Correct"] = (
    comparison["Actual_Team"]
    ==
    comparison["Predicted_Team"]
)

print("\n================================================")
print("TOP 50 PREDICTIONS")
print("================================================")

print(
    comparison.head(50)
)

# ==========================================================
# ERROR ANALYSIS
# ==========================================================

errors = comparison[
    comparison["Correct"] == False
]

print("\n================================================")
print("WRONG PREDICTIONS")
print("================================================")

print(
    errors["Actual_Team"].value_counts()
)

# Log every wrong prediction

for _, row in errors.iterrows():

    logging.warning(
        f"Wrong Prediction | "
        f"Actual={row['Actual_Team']} | "
        f"Predicted={row['Predicted_Team']} | "
        f"Confidence={row['Confidence']}"
    )

# ==========================================================
# FEATURE IMPORTANCE
# ==========================================================

print("\n================================================")
print("FEATURE IMPORTANCE")
print("================================================")

feature_names = (
    list(embedding_df.columns)
    +
    structured_features
)

feature_importance = pd.DataFrame(
    {
        "Feature":
        feature_names,
        "Importance":
        model.get_feature_importance()
    }
)

feature_importance = (
    feature_importance
    .sort_values(
        by="Importance",
        ascending=False
    )
)

print(feature_importance)

feature_importance.to_excel(
    "feature_importance.xlsx",
    index=False
)

# ==========================================================
# SAVE COMPARISON
# ==========================================================

comparison.to_excel(
    "prediction_comparison.xlsx",
    index=False
)

print(
    "\nSaved prediction_comparison.xlsx"
)

# ==========================================================
# SAVE MODEL
# ==========================================================

joblib.dump(
    model,
    "catboost_defect_triage_model.pkl"
)

logging.info(
    "Model Saved Successfully"
)

joblib.dump(
    label_encoder,
    "label_encoder.pkl"
)

joblib.dump(
    {
        "embedding_model":
        "all-MiniLM-L6-v2"
    },
    "embedding_config.pkl"
)


logging.info(
    "Label Encoder Saved Successfully"
)

print(
    "\nSaved catboost_defect_triage_model.pkl"
)

# ==========================================================
# MANUAL PREDICTION
# ==========================================================

sample = pd.DataFrame([
    {
        "Issue_Text":
        """
        Fuel gauge freezes at 45 percent.
        Customer reports incorrect range estimate.
        CAN sensor timeout observed.
        """,

        "Module":
        "HMI",

        "Subsystem":
        "Fuel Gauge",

        "Module_Subsystem":
        "HMI_Fuel Gauge",

        "Severity":
        "High",

        "Priority":
        "P2",

        "Severity_Priority":
        "High_P2",

        "Vehicle_Model":
        "SUV Z",

        "Variant":
        "Top",

        "Software_Version":
        "v4.2",

        "Version_Module":
        "v4.2_HMI"
    }
])

sample_embedding = (
    embedding_model.encode(
        sample["Issue_Text"].tolist()
    )
)

sample_embedding_df = pd.DataFrame(
    sample_embedding,
    columns=[
        f"emb_{i}"
        for i in range(
            sample_embedding.shape[1]
        )
    ]
)

sample_structured = sample[
    structured_features
].reset_index(drop=True)

sample_final = pd.concat(
    [
        sample_embedding_df,
        sample_structured
    ],
    axis=1
)

pred = model.predict(
    sample_final
)

prob = model.predict_proba(
    sample_final
)


team = label_encoder.inverse_transform(
    pred.flatten().astype(int)
)[0]

confidence = prob.max()
# Automotive Event Logging

if team == "CAN Team":

    logging.error(
        "CAN_TIMEOUT ID=0x1A0"
    )

elif team == "Diagnostic Team":

    logging.error(
        "UDS_NEG_RESPONSE SID=22"
    )

elif team == "HMI Team":

    logging.error(
        "DISPLAY_FREEZE"
    )

else:

    logging.error(
        "FIRMWARE_TASK_FAILURE"
    )

logging.info(
    f"Manual Prediction | "
    f"Team={team} | "
    f"Confidence={confidence:.4f}"
)



print("\n================================================")
print("MANUAL TEST")
print("================================================")

print("Predicted Team :", team)
print("Confidence     :", round(confidence, 4))

logging.info(
    f"Manual Prediction | Team={team}"
)

logging.info(
    f"Confidence={confidence:.4f}"
)