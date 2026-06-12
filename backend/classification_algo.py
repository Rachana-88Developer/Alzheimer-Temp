import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from dataset import load_and_preprocess

# Import standard Machine Learning algorithms from Scikit-Learn
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

# Import evaluation metrics for result analysis
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, classification_report, confusion_matrix)

# Use 'Agg' backend for matplotlib to avoid GUI issues on remote servers/drives
matplotlib.use('Agg')

def initialize_models():
    """
    Initialize the 6 Classification Algorithms with standard hyperparameters.
    These models range from simple linear models to complex ensemble methods.
    """
    return {
        "Random Forest":       RandomForestClassifier(n_estimators=100, random_state=42),
        "SVM (RBF Kernel)":    SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42),
        "KNN (k=5)":           KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression":  LogisticRegression(max_iter=1000, random_state=42),
        "Naive Bayes":         GaussianNB(),
        "Decision Tree":       DecisionTreeClassifier(max_depth=10, random_state=42),
    }

def train_and_evaluate(models, data):
    """
    Core logic to train each model on the training set and 
    evaluate its performance on the unseen test set.
    """
    X_train, X_test, y_train, y_test = data
    results = []

    print("\n" + "=" * 60)
    print("  PHASE 1: MODEL TRAINING & PER-CLASS PERFORMANCE")
    print("=" * 60)

    for name, model in models.items():
        
        model.fit(X_train, y_train)
        
     
        y_pred = model.predict(X_test)

       
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec  = recall_score(y_test, y_pred, zero_division=0)
        f1   = f1_score(y_test, y_pred, zero_division=0)
        cm   = confusion_matrix(y_test, y_pred)

        results.append({
            "name": name, 
            "acc": acc, 
            "prec": prec, 
            "rec": rec, 
            "f1": f1, 
            "cm": cm
        })

        # Provide a detailed human-readable report for this specific model
        print(f"\n[ANALYSIS] Algorithm: {name}")
        print("-" * 55)
        print(classification_report(y_test, y_pred, target_names=["Control (0)", "Alzheimer's (1)"]))
        print(f"Confusion Matrix (Actual vs Predicted):\n{cm}")

    return results

def display_comparison_table(results):
    """
    Displays a final sorted comparison table of all algorithms.
    Helps in identifying the best performing model at a glance.
    """
    print("\n" + "=" * 75)
    print("  PHASE 2: CROSS-ALGORITHM PERFORMANCE COMPARISON")
    print("=" * 75)
    
    header = f"{'Algorithm':<25} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1-Score':>10}"
    print(header)
    print("-" * 75)

    # Sort algorithms by Accuracy descending. High accuracy models appear first.
    results_sorted = sorted(results, key=lambda x: x["acc"], reverse=True)

    for r in results_sorted:
        print(f"{r['name']:<25} {r['acc']:>9.2%} {r['prec']:>9.2%} {r['rec']:>9.2%} {r['f1']:>9.2%}")

    print("-" * 75)

    best = results_sorted[0]
    print(f"\n[CONCLUSION] The best performing model is {best['name']} ")
    print(f"with an impressive accuracy of {best['acc']:.2%}.")
    
    return results_sorted

def save_visualizations(results, results_sorted):
    """
    Generate and save statistical plots for the research findings.
    - Plot 1: Mixed Confusion Matrices for all models.
    - Plot 2: Comparative Bar Chart of all performance metrics.
    """
    # Visualization 1: Heatmaps of Confusion Matrices
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Diagnostic Accuracy Analysis: Confusion Matrices", fontsize=20, fontweight='bold', y=0.98)

    for idx, r in enumerate(results):
        ax = axes[idx // 3][idx % 3]
        sns.heatmap(r["cm"], annot=True, fmt="d", cmap="YlGnBu", ax=ax,
                    xticklabels=["Normal", "AD"], yticklabels=["Normal", "AD"])
        ax.set_title(f"Model: {r['name']}", fontsize=14, loc='center', pad=10)
        ax.set_xlabel("Predicted Diagnosis")
        ax.set_ylabel("Actual Condition")

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig("confusion_matrices_report.png", dpi=300)
    print("\n[EXPORT] Detailed Confusion Matrices saved to 'confusion_matrices_report.png'")

    # Visualization 2: Multi-Metric Comparative Bar Chart
    fig2, ax2 = plt.subplots(figsize=(14, 7))
    names = [r["name"] for r in results_sorted]
    
    metrics_data = {
        "Accuracy":  [r["acc"]  for r in results_sorted],
        "Precision": [r["prec"] for r in results_sorted],
        "Recall":    [r["rec"]  for r in results_sorted],
        "F1-Score":  [r["f1"]   for r in results_sorted],
    }

    x = np.arange(len(names))
    width = 0.2
    # Professional color palette for research publication
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

    for i, (label, values) in enumerate(metrics_data.items()):
        ax2.bar(x + i * width, values, width, label=label, color=colors[i], edgecolor='black', alpha=0.8)

    ax2.set_xlabel("AI Algorithm", fontsize=12)
    ax2.set_ylabel("Performance Score (0.0 - 1.0)", fontsize=12)
    ax2.set_title("Comparative Analysis of Prediction Metrics", fontsize=16, fontweight='bold')
    ax2.set_xticks(x + width * 1.5)
    ax2.set_xticklabels(names, rotation=20, ha='right')
    ax2.legend(loc='lower left', bbox_to_anchor=(1, 0.5))
    ax2.set_ylim(0, 1.1)
    ax2.grid(axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("algorithm_performance_summary.png", dpi=300)
    print("[EXPORT] Overall Performance Summary chart saved to 'algorithm_performance_summary.png'")

def main():
    """
    Main execution flow for the Alzheimer's Diagnostic Study.
    """
    # Step 1: Load and Preprocess the Dataset using the dedicated module
    # Returns standardized data ready for machine learning
    X_train, X_test, y_train, y_test, features = load_and_preprocess()
    data_split = (X_train, X_test, y_train, y_test)

    # Step 2: Initialize the set of algorithms
    models = initialize_models()

    # Step 3: Run the Training and Evaluation cycle
    results = train_and_evaluate(models, data_split)

    # Step 4: Display the summary results in a readable table
    results_sorted = display_comparison_table(results)

    # Step 5: Save high-resolution plots for reports/papers
    save_visualizations(results, results_sorted)

    print("\n" + "=" * 60)
    print("  STUDY COMPLETED: All analysis files have been generated.")
    print("=" * 60)

if __name__ == "__main__":
    main()
