# --- Report: Summary of Results ---

import matplotlib.pyplot as plt
import pandas as pd

# 1. Model Performance Summary
print("Model Performance Summary")
print(f"Final Model: {model_name}")   # (LLM: Replace with actual mod name)
print(f"Evaluation Score: {score}")   # (LLM: Replace with actual score)

# 2. Important Variables (e.g., Tree-based model)
if 'feature_importances_' in dir(model):
    importance_df = pd.DataFrame({
        'feature': X.columns,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    display(importance_df)
    importance_df.plot.bar(x='feature', y='importance', legend=False)
    plt.title('Feature Importances')
    plt.show()

# 3. Predicted vs Actual (Regression)
if 'y_test' in locals() and 'y_pred' in locals():
    plt.scatter(y_test, y_pred, alpha=0.3)
    plt.xlabel('Actual')
    plt.ylabel('Predicted')
    plt.title('Predicted vs Actual')
    plt.show()

# 4. Confusion Matrix (Classification)
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
if 'y_test' in locals() and 'y_pred' in locals() and len(set(y_test)) <= 10:
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()

# 5. LLM-generated automatic interpretation/business insights
print("""[LLM_SUMMARY_PLACEHOLDER]
Here is the text of the LLM-generated automatic interpretation/business insights.
""") 