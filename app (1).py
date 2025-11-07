import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
from io import BytesIO

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, RocCurveDisplay, auc, roc_curve
)
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier

# =========================
# Helpers
# =========================

@st.cache_data
def load_csv(upload) -> pd.DataFrame:
    return pd.read_csv(upload)

def infer_column_types(df: pd.DataFrame):
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    cat_cols = [c for c in df.columns if c not in num_cols]
    return num_cols, cat_cols

def fill_missing(df: pd.DataFrame):
    df = df.copy()
    num_cols, cat_cols = infer_column_types(df)
    for c in num_cols:
        if df[c].isna().any():
            df[c] = df[c].fillna(df[c].mean())
    for c in cat_cols:
        if df[c].isna().any():
            mode_vals = df[c].mode(dropna=True)
            df[c] = df[c].fillna(mode_vals.iloc[0] if not mode_vals.empty else "Unknown")
    return df

def label_encode_dataframe(df: pd.DataFrame):
    """Encode all categorical columns and return encoders map for inverse transform later."""
    df_enc = df.copy()
    num_cols, cat_cols = infer_column_types(df_enc)
    encoders = {}
    for c in cat_cols:
        le = LabelEncoder()
        df_enc[c] = le.fit_transform(df_enc[c].astype(str))
        encoders[c] = le
    return df_enc, encoders

def ensure_binary_or_warn(y: pd.Series):
    classes = sorted(y.unique())
    if len(classes) == 2:
        return True, classes
    else:
        st.warning(f"Label has {len(classes)} classes. Some metrics (ROC AUC) are only shown for binary classification.")
        return False, classes

def confusion_matrix_plot(y_true, y_pred, class_names):
    labels_sorted = sorted(np.unique(y_true))
    cm = confusion_matrix(y_true, y_pred, labels=labels_sorted)
    fig, ax = plt.subplots(figsize=(6,5))
    im = ax.imshow(cm)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    ax.set_xticks(range(len(labels_sorted)))
    ax.set_yticks(range(len(labels_sorted)))
    ax.set_xticklabels([str(c) for c in labels_sorted], rotation=45, ha="right")
    ax.set_yticklabels([str(c) for c in labels_sorted])
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    st.pyplot(fig)

def model_block(X, y, model, model_name, pos_label=None):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    model.fit(X_train, y_train)
    y_pred_tr = model.predict(X_train)
    y_pred_te = model.predict(X_test)

    train_acc = accuracy_score(y_train, y_pred_tr)
    test_acc = accuracy_score(y_test, y_pred_te)

    is_binary, classes = ensure_binary_or_warn(y)
    # For binary metrics
    if is_binary:
        average = "binary"
        # Best-effort pick positive label
        if pos_label is None:
            pos_label = classes[-1]
        prec = precision_score(y_test, y_pred_te, pos_label=pos_label, zero_division=0)
        rec = recall_score(y_test, y_pred_te, pos_label=pos_label, zero_division=0)
        f1 = f1_score(y_test, y_pred_te, pos_label=pos_label, zero_division=0)
        # Try ROC AUC
        try:
            if hasattr(model, "predict_proba"):
                y_proba = model.predict_proba(X_test)[:, list(model.classes_).index(pos_label)]
                auc_val = roc_auc_score(y_test == pos_label, y_proba)
            elif hasattr(model, "decision_function"):
                scores = model.decision_function(X_test)
                auc_val = roc_auc_score(y_test == pos_label, scores)
            else:
                auc_val = np.nan
        except Exception:
            auc_val = np.nan
    else:
        average = "macro"
        prec = precision_score(y_test, y_pred_te, average=average, zero_division=0)
        rec  = recall_score(y_test, y_pred_te, average=average, zero_division=0)
        f1   = f1_score(y_test, y_pred_te, average=average, zero_division=0)
        auc_val = np.nan

    # Cross-validation (accuracy)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_scores = cross_val_score(model, X, y, scoring="accuracy", cv=cv, n_jobs=None)
    cv_mean = float(np.mean(cv_scores))

    st.subheader(model_name)
    st.write(pd.DataFrame([{
        "Train Accuracy": train_acc,
        "Test Accuracy": test_acc,
        "Precision": prec,
        "Recall": rec,
        "F1": f1,
        "ROC AUC": auc_val,
        "CV Acc (5-fold)": cv_mean
    }]).T.rename(columns={0:"Score"}))

    st.markdown("**Confusion Matrix (Test)**")
    confusion_matrix_plot(y_test, y_pred_te, classes)

    # Feature importance (if available)
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        order = np.argsort(importances)[::-1]
        topn = min(25, len(order))
        imp_df = pd.DataFrame({
            "Feature": X.columns[order][:topn],
            "Importance": importances[order][:topn]
        })
        fig = px.bar(imp_df, x="Feature", y="Importance", title=f"{model_name} - Feature Importances")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"{model_name} does not expose feature importances.")

def download_df_button(df: pd.DataFrame, filename: str, label: str):
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(label=label, data=csv, file_name=filename, mime="text/csv")

# =========================
# Sidebar: Data loading
# =========================
st.set_page_config(page_title="Attrition & Retention Dashboard", layout="wide")

st.sidebar.title("Data")
st.sidebar.write("Upload your CSV. You can use your earlier dataset as well.")
uploaded = st.sidebar.file_uploader("Choose CSV", type=["csv"])

# Fallback: path text box (for local dev)
path_text = st.sidebar.text_input("Or enter a path (optional):", "")

df = None
if uploaded is not None:
    df = load_csv(uploaded)
elif path_text:
    try:
        df = pd.read_csv(path_text)
    except Exception as e:
        st.sidebar.error(f"Failed to load path: {e}")

if df is None:
    st.info("Please upload a CSV to get started.")
    st.stop()

st.success(f"Loaded data with shape: {df.shape}")
df_orig = df.copy()

# =========================
# Schema selection block
# =========================
st.sidebar.subheader("Column Selection")
all_cols = df.columns.tolist()
num_cols, cat_cols = infer_column_types(df)

# Label column (what you want to predict)
label_col = st.sidebar.selectbox("Label/Target column (e.g., Attrition or POLICY_STATUS)", options=all_cols, index=min(0, len(all_cols)-1))

# Job role-like column for filters
job_col = st.sidebar.selectbox("Job Role-like column (categorical)", options=cat_cols if cat_cols else all_cols)

# Satisfaction-like column for slider
satisfaction_col = st.sidebar.selectbox("Satisfaction-like column (numeric)", options=num_cols if num_cols else all_cols)

# Optional second categorical for heatmap
second_cat_col = st.sidebar.selectbox("Second categorical (for heatmap)", options=[c for c in cat_cols if c != job_col] if len(cat_cols)>1 else all_cols)

# Positive class selection (for binary metrics)
label_values = df[label_col].astype(str).unique().tolist()
pos_label_val = st.sidebar.selectbox("Positive class (for binary metrics)", options=sorted(label_values), index=0)

# =========================
# Missing-value handling
# =========================
df = fill_missing(df)

# =========================
# Tabs
# =========================
tab1, tab2, tab3, tab4 = st.tabs(["Insights Dashboard", "Filters", "Modeling", "Batch Predict"])

# ------------------------------------
# Tab 1: INSIGHTS DASHBOARD (5 charts)
# ------------------------------------
with tab1:
    st.header("HR Retention Insights")
    st.caption("Interactive visuals to explore drivers of attrition/label and plan interventions.")

    # KPI cards
    colA, colB, colC = st.columns(3)
    total = len(df)
    rate = df[label_col].astype(str).eq(pos_label_val).mean() * 100
    unique_roles = df[job_col].nunique() if job_col in df.columns else 0
    colA.metric("Total Records", f"{total}")
    colB.metric(f"Rate of '{pos_label_val}'", f"{rate:.1f}%")
    colC.metric(f"Unique {job_col}", f"{unique_roles}")

    # 1) Attrition/label rate by Job Role
    st.subheader("1) Rate by Job Role")
    tmp = df.groupby(job_col)[label_col].apply(lambda s: (s.astype(str)==pos_label_val).mean()).reset_index(name="rate")
    fig1 = px.bar(tmp.sort_values("rate", ascending=False), x=job_col, y="rate", title=f"Rate of '{pos_label_val}' by {job_col}")
    st.plotly_chart(fig1, use_container_width=True)

    # 2) Satisfaction vs label distribution (box)
    st.subheader("2) Satisfaction Distribution by Label")
    fig2 = px.box(df, x=label_col, y=satisfaction_col, points="all", title=f"{satisfaction_col} by {label_col}")
    st.plotly_chart(fig2, use_container_width=True)

    # 3) Satisfaction buckets vs rate (stacked bars)
    st.subheader("3) Satisfaction Buckets vs Label Rate")
    df["sat_bucket"] = pd.qcut(df[satisfaction_col].rank(method="first"), q=5, labels=["Very Low","Low","Medium","High","Very High"])
    tmp3 = df.groupby(["sat_bucket"])[label_col].apply(lambda s: (s.astype(str)==pos_label_val).mean()).reset_index(name="rate")
    fig3 = px.bar(tmp3, x="sat_bucket", y="rate", title=f"Rate of '{pos_label_val}' across {satisfaction_col} buckets")
    st.plotly_chart(fig3, use_container_width=True)

    # 4) Heatmap: Job Role vs second categorical → rate
    st.subheader(f"4) Heatmap: {job_col} vs {second_cat_col}")
    pivot = df.pivot_table(index=job_col, columns=second_cat_col, values=label_col,
                           aggfunc=lambda s: (s.astype(str)==pos_label_val).mean(), fill_value=0.0)
    fig4 = px.imshow(pivot, text_auto=True, aspect="auto", title=f"Rate of '{pos_label_val}' by {job_col} × {second_cat_col}")
    st.plotly_chart(fig4, use_container_width=True)

    # 5) Feature importance proxy: RandomForest on encoded data
    st.subheader("5) Feature Importance (RandomForest proxy)")
    # Prepare encoded copy excluding label
    features = df.drop(columns=[label_col])
    lab = df[label_col].astype(str)
    features_enc, encs = label_encode_dataframe(features)
    lab_enc = LabelEncoder().fit_transform(lab)
    rf = RandomForestClassifier(n_estimators=300, random_state=42)
    rf.fit(features_enc, lab_enc)
    importances = rf.feature_importances_
    order = np.argsort(importances)[::-1]
    topn = min(20, len(order))
    imp_df = pd.DataFrame({"Feature": features_enc.columns[order][:topn], "Importance": importances[order][:topn]})
    fig5 = px.bar(imp_df, x="Feature", y="Importance", title="RandomForest Feature Importances (Top 20)")
    st.plotly_chart(fig5, use_container_width=True)

# ------------------------
# Tab 2: FILTERS (global)
# ------------------------
with tab2:
    st.header("Global Filters")
    st.caption("These filters apply to the charts in Tab 1 when re-run below.")

    # Multiselect on Job Role
    job_vals = sorted(df[job_col].unique().tolist())
    selected_roles = st.multiselect(f"Filter {job_col}", options=job_vals, default=job_vals)

    # Satisfaction slider
    s_min = float(df[satisfaction_col].min())
    s_max = float(df[satisfaction_col].max())
    s_range = st.slider(f"{satisfaction_col} range", min_value=s_min, max_value=s_max, value=(s_min, s_max))

    # Apply filters
    if selected_roles:
        df = df[df[job_col].isin(selected_roles)]
    df = df[(df[satisfaction_col] >= s_range[0]) & (df[satisfaction_col] <= s_range[1])]

    st.success(f"Filtered dataset shape: {df.shape}. Go back to **Insights Dashboard** (Tab 1) to see charts for the filtered data.")

# ------------------------
# Tab 3: MODELING
# ------------------------
with tab3:
    st.header("Apply 3 Algorithms & Metrics")
    st.caption("Click the button to train Decision Tree, Random Forest, and Gradient Boosted Trees with an 80:20 stratified split.")

    # Prepare data (encode all non-numeric)
    df_model = fill_missing(df_orig)
    X_all = df_model.drop(columns=[label_col])
    y_all = df_model[label_col].astype(str)

    X_enc, encs = label_encode_dataframe(X_all)
    y_enc = LabelEncoder().fit_transform(y_all)

    # Positive label mapping
    # Ensure pos_label exists in encoded vector
    le_y = LabelEncoder().fit(y_all)
    y_codes = le_y.transform(y_all)
    if pos_label_val in le_y.classes_:
        pos_label_code = int(np.where(le_y.classes_ == pos_label_val)[0][0])
    else:
        pos_label_code = int(np.argmax(np.bincount(y_codes)))

    if st.button("Run Models"):
        models = {
            "Decision Tree": DecisionTreeClassifier(random_state=42),
            "Random Forest": RandomForestClassifier(n_estimators=300, random_state=42),
            "Gradient Boosted Trees": GradientBoostingClassifier(random_state=42),
        }
        for name, model in models.items():
            model_block(X_enc, le_y.transform(y_all), model, name, pos_label=pos_label_code)

# ------------------------
# Tab 4: BATCH PREDICT
# ------------------------
with tab4:
    st.header("Upload New Data & Predict")
    st.caption("Upload a CSV with the **same columns as training features** (i.e., all columns except the label). We'll encode using the current dataset's encoders and return predictions.")

    df_current = fill_missing(df_orig)
    X_train_like = df_current.drop(columns=[label_col])
    y_train_like = df_current[label_col].astype(str)

    # Fit encoders & model on full current data
    X_enc_full, encs_full = label_encode_dataframe(X_train_like)
    le_y_full = LabelEncoder().fit(y_train_like)
    y_enc_full = le_y_full.transform(y_train_like)

    base_model = RandomForestClassifier(n_estimators=300, random_state=42)
    base_model.fit(X_enc_full, y_enc_full)

    uploaded_new = st.file_uploader("Upload CSV for prediction", type=["csv"], key="pred_csv")
    if uploaded_new is not None:
        new_df = pd.read_csv(uploaded_new)
        st.write("Preview of uploaded data:", new_df.head())

        # Align columns
        missing_cols = [c for c in X_train_like.columns if c not in new_df.columns]
        extra_cols = [c for c in new_df.columns if c not in X_train_like.columns]
        if missing_cols:
            st.error(f"Missing columns: {missing_cols}. Please provide all training feature columns.")
        else:
            new_df = new_df[X_train_like.columns]  # reorder

            # Fill missing & encode using fitted encoders
            new_df = fill_missing(new_df).copy()
            new_enc = new_df.copy()
            for c in new_enc.columns:
                if c in encs_full:
                    # map unseen labels to a placeholder (extend classes)
                    le = encs_full[c]
                    vals = new_enc[c].astype(str).values
                    mapped = []
                    for v in vals:
                        if v in le.classes_:
                            mapped.append(le.transform([v])[0])
                        else:
                            # unseen -> add dynamically: fall back to 0
                            mapped.append(0)
                    new_enc[c] = mapped

            preds_code = base_model.predict(new_enc)
            preds_label = le_y_full.inverse_transform(preds_code)

            out = new_df.copy()
            out[label_col] = preds_label
            st.write("Predictions (preview):", out.head())

            download_df_button(out, filename="predictions_with_label.csv", label="Download Predictions CSV")
            st.success("Prediction complete. Download the file for your records.")


st.caption("© Your Streamlit Cloud Dashboard — HR Retention & Attrition / General Label Analytics")
