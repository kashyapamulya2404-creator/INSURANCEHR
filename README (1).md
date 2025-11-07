
# Streamlit Cloud: Attrition / Label Analytics Dashboard

**How to deploy (GitHub → Streamlit Cloud):**
1. Create a new **GitHub repo**.
2. Upload these three files to the repo root:
   - `app.py`
   - `requirements.txt` (no versions used as requested)
   - `README.md`
3. Go to https://share.streamlit.io/ and connect your repo.
4. Set **main file** to `app.py` and deploy.

**How to use:**
- In the sidebar, upload your CSV. Select the columns:
  - Label/Target column (e.g., `Attrition` or `POLICY_STATUS`)
  - A Job Role-like categorical column (for filters and charts)
  - A numeric Satisfaction-like column (for slider)
  - A second categorical column for the heatmap
  - Choose the "Positive class" (e.g., `Yes` for attrition)
- **Tab 1 – Insights Dashboard**: 5 charts with actionable insights.
- **Tab 2 – Filters**: Filter by Job Role multiselect and satisfaction slider (applies to Tab 1 charts when you revisit them).
- **Tab 3 – Modeling**: Click the button to run Decision Tree, Random Forest, and Gradient Boosted Trees. Shows Train/Test Accuracy, Precision, Recall, F1, ROC AUC (binary), 5-fold CV, and Confusion Matrix, plus feature importances when available.
- **Tab 4 – Batch Predict**: Upload a *new* dataset (same feature columns) to get predictions and download the output with the predicted label appended.

**Notes:**
- Missing values handled automatically (numeric→mean, categorical→mode).
- All non-numeric features are label-encoded for tree-based models.
- If your task isn't binary, ROC AUC will be hidden and other metrics use macro-average.
- Keep column names consistent when using Batch Predict.
