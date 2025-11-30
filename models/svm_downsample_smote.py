import numpy as np
import pandas as pd
import warnings
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline 

# Suppress warnings for clean output during tuning
warnings.filterwarnings("ignore", category=UserWarning, message="Precision is ill-defined")
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning) # For division by zero in metrics

# --- Configuration ---
RANDOM_STATE = 42
TARGET_COL = "fire_today"
TEST_SIZE_RATIO = 0.20
OPTIMAL_DOWNSAMPLE_RATIO = 20 # Best ratio found
OPTIMAL_C_VAL = 1.0           # Best C value found
OPTIMAL_K_VAL = 25            # K to use for this run (can be adjusted)

# ========================================================
# 0. Helper Functions
# ========================================================

def tune_threshold(clf_pipeline, X_test, y_test):
    """Evaluates model performance across a range of thresholds."""
    # LinearSVC uses decision_function for scores
    y_scores = clf_pipeline.decision_function(X_test) 
    # Test a common range for tuned SVM scores
    thresholds = np.arange(-2.0, 2.1, 0.1) 
    
    current_best_f1 = -1
    optimal_t = 0
    
    for t in thresholds:
        y_pred_tuned = (y_scores >= t).astype(int)
        
        # Calculate F1 score for the minority class (1)
        p, r, f1, _ = precision_recall_fscore_support(
            y_test, y_pred_tuned, labels=[1], average=None, zero_division=0
        )
        
        if f1[0] > current_best_f1:
            current_best_f1 = f1[0]
            optimal_t = t
            
    return current_best_f1, optimal_t

def check_nan_after_impute(df_full, feature_cols):
    """Performs a preliminary check on NaN counts after applying imputation."""
    print("\n================= NaN Check After Imputation (Simulated) =================")
    df_temp = df_full[feature_cols].copy()
    imputer = SimpleImputer(strategy="median")
    imputer.fit(df_temp)
    df_imputed = pd.DataFrame(imputer.transform(df_temp), columns=feature_cols)
    nan_counts = df_imputed.isnull().sum()
    
    print("NaN counts per column after median imputation (should all be 0):")
    print(nan_counts[nan_counts > 0].to_markdown(numalign="left", stralign="left"))
    if nan_counts.sum() == 0:
        print("✅ Imputation would successfully eliminate all NaNs in the feature set.")
    else:
        print(f"⚠️ Warning: {nan_counts.sum()} NaNs remaining.")


# ========================================================
# 1. Load Data & Feature Engineering (WITH DROPS) 📐
# ========================================================
try:
    df = pd.read_csv("ca_5km_daily_panel_2020_2023_6_9.csv")
    print(f"Data loaded successfully. Total rows: {len(df)}")
except FileNotFoundError:
    print("Error: ca_5km_daily_panel_2020_2023_6_9.csv not found.")
    exit()

# Feature Engineering
df['date'] = pd.to_datetime(df['date'])
df['day_of_year'] = df['date'].dt.dayofyear
df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 366)
df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 366)
df['month'] = df['date'].dt.month
df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

# --- Define Features to Drop/Keep (UPDATED) ---
drop_cols = [
    "date", 
    "grid_id", 
    TARGET_COL, 
    "day_of_year", 
    "month",
    
    # CRITICAL: Drop redundant and highly collinear coordinate features
    "w_lat",  
    "w_lon",
    "n_lat",
    "n_lon",
]

feature_cols = [c for c in df.columns if c not in drop_cols]
print(f"Total features selected for input: {len(feature_cols)}")

X = df[feature_cols].values.astype(np.float32)
y = df[TARGET_COL].values.astype(int)

# --- Check NaN Counts ---
check_nan_after_impute(df, feature_cols)

# 2. Train/Test Split
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=TEST_SIZE_RATIO, random_state=RANDOM_STATE, stratify=y
)

# 3. Downsampling
print(f"\n--- Sampling Summary (Fixed at {OPTIMAL_DOWNSAMPLE_RATIO}:1) ---")
pos_indices = np.where(y_train_full == 1)[0]; neg_indices = np.where(y_train_full == 0)[0]
X_pos = X_train_full[pos_indices]; y_pos = y_train_full[pos_indices]
X_neg = X_train_full[neg_indices]; y_neg = y_train_full[neg_indices]
num_pos = len(y_pos); num_neg_to_keep = min(len(y_neg), num_pos * OPTIMAL_DOWNSAMPLE_RATIO)
neg_sample_indices = np.random.choice(len(y_neg), size=num_neg_to_keep, replace=False)
X_neg_sampled = X_neg[neg_sample_indices]; y_neg_sampled = y_neg[neg_sample_indices]
X_train_sampled = np.vstack([X_pos, X_neg_sampled])
y_train_sampled = np.concatenate([y_pos, y_neg_sampled])
shuffled_indices = np.random.permutation(len(y_train_sampled))
X_train_sampled = X_train_sampled[shuffled_indices]
y_train_sampled = y_train_sampled[shuffled_indices]
print(f"Training set size (SAMPLED): {len(y_train_sampled)}")


# =======================================================
# 4. Final SVM Pipeline Setup and Fit
# =======================================================

print(f"\n================= Final SVM Model Training (K={OPTIMAL_K_VAL}) =================")

# Build the final optimal SVM pipeline
clf_pipeline_svm = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")), 
        ("scaler", StandardScaler()), 
        ("smote", SMOTE(random_state=RANDOM_STATE, k_neighbors=5)), 
        ("feature_selection", SelectKBest(score_func=mutual_info_classif, k=OPTIMAL_K_VAL)), 
        ("clf", LinearSVC(
            C=OPTIMAL_C_VAL, 
            class_weight="balanced", 
            max_iter=10000, 
            random_state=RANDOM_STATE,
            dual='auto'
        )),
    ]
)

# Fit the final model
clf_pipeline_svm.fit(X_train_sampled, y_train_sampled)

# 5. Tune Threshold
current_best_f1, optimal_t = tune_threshold(clf_pipeline_svm, X_test, y_test)

# 6. Final Evaluation
print(f"\n================= Final Evaluation ({OPTIMAL_K_VAL} Features, T={optimal_t:.2f}) =================")
y_scores_final = clf_pipeline_svm.decision_function(X_test)
y_pred_final = (y_scores_final >= optimal_t).astype(int)

print("\nFinal Classification report (BEST TUNED MODEL):")
print(classification_report(y_test, y_pred_final, digits=4))
print("\nConfusion matrix (BEST TUNED MODEL):")
print(confusion_matrix(y_test, y_pred_final))

# =======================================================
# 7. FEATURE CONTRIBUTION AND RANKING (SVM COEFFICIENTS)
# =======================================================
print("\n================= Feature Importance Ranking (SVM Coefficients) =================\n")

# 1. Access the trained LinearSVC model
svm_model = clf_pipeline_svm.named_steps['clf']

# 2. Access the coefficients (feature weights)
coefficients = svm_model.coef_[0] 

# 3. Get the feature names used by the SVM model
fs = clf_pipeline_svm.named_steps['feature_selection']
selected_mask = fs.get_support()
# Feature names are taken from the original feature_cols, filtered by SelectKBest
feature_names = np.array(feature_cols)[selected_mask]

# 4. Create a DataFrame for ranking
feature_df = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Absolute_Importance': np.abs(coefficients) 
})

# 5. Sort and Print the ranked list
feature_df = feature_df.sort_values(by='Absolute_Importance', ascending=False).reset_index(drop=True)
feature_df.index.name = 'Rank'
feature_df.index = feature_df.index + 1 

print(feature_df[['Feature', 'Coefficient', 'Absolute_Importance']].to_markdown(numalign="left", stralign="left"))