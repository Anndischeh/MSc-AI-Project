# run_comparison_fixed.py
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import (
    roc_auc_score, 
    average_precision_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    accuracy_score,
    roc_curve
)

RESULTS_DIR = "./results"

# Model names exactly as they appear in your file names
MODELS = [
    "Autoencoder",
    "VAE", 
    "GAN",
    "GANomaly",
    "ViT_AE_CD"
]

def load_model_data():
    """Load scores and labels for all models"""
    model_data = {}
    
    print("=" * 70)
    print("LOADING MODEL DATA")
    print("=" * 70)
    
    for model in MODELS:
        try:
            # Try exact filename first
            scores_file = f"{model}_scores.npy"
            labels_file = f"{model}_labels.npy"
            
            scores_path = os.path.join(RESULTS_DIR, scores_file)
            labels_path = os.path.join(RESULTS_DIR, labels_file)
            
            if os.path.exists(scores_path) and os.path.exists(labels_path):
                scores = np.load(scores_path)
                labels = np.load(labels_path)
                
                print(f"✓ {model}: Loaded {len(scores)} scores and {len(labels)} labels")
                
                # Check data quality
                if len(scores) != len(labels):
                    print(f"Warning: Mismatch in array lengths")
                
                if np.any(np.isnan(scores)):
                    print(f"Warning: NaN values in scores, replacing with median")
                    median_val = np.nanmedian(scores)
                    scores = np.nan_to_num(scores, nan=median_val)
                
                model_data[model] = {
                    'scores': scores,
                    'labels': labels
                }
            else:
                print(f"✗ {model}: Missing score or label file")
                
        except Exception as e:
            print(f"✗ {model}: Error loading data - {e}")
    
    return model_data

def calculate_all_metrics(model_data):
    """Calculate performance metrics for all models"""
    metrics_list = []
    
    print("\n" + "=" * 70)
    print("CALCULATING METRICS")
    print("=" * 70)
    
    for model_name, data in model_data.items():
        scores = data['scores']
        labels = data['labels']
        
        print(f"\n{model_name}:")
        print("-" * 40)
        
        try:
            # Check if we have both classes
            unique_labels = np.unique(labels)
            if len(unique_labels) < 2:
                print(f"Only one class present: {unique_labels}")
                continue
            
            # Calculate ROC curve and find optimal threshold
            fpr, tpr, thresholds = roc_curve(labels, scores)
            
            # Calculate Youden's J statistic for optimal threshold
            j_scores = tpr - fpr
            optimal_idx = np.argmax(j_scores)
            optimal_threshold = thresholds[optimal_idx]
            
            # Make predictions
            predictions = (scores >= optimal_threshold).astype(int)
            
            # Calculate confusion matrix
            tn, fp, fn, tp = confusion_matrix(labels, predictions, labels=[0, 1]).ravel()
            
            # Calculate all metrics
            roc_auc = roc_auc_score(labels, scores)
            ap = average_precision_score(labels, scores)
            precision = precision_score(labels, predictions, zero_division=0)
            recall = recall_score(labels, predictions, zero_division=0)
            f1 = f1_score(labels, predictions, zero_division=0)
            accuracy = accuracy_score(labels, predictions)
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            
            print(f"  ROC-AUC: {roc_auc:.4f}")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  Accuracy: {accuracy:.4f}")
            print(f"  Optimal threshold: {optimal_threshold:.4f}")
            
            metrics_list.append({
                'model': model_name,
                'roc_auc': roc_auc,
                'average_precision': ap,
                'precision': precision,
                'recall': recall,
                'f1': f1,
                'accuracy': accuracy,
                'specificity': specificity,
                'tn': int(tn),
                'fp': int(fp),
                'fn': int(fn),
                'tp': int(tp),
                'optimal_threshold': optimal_threshold
            })
            
        except Exception as e:
            print(f"  ✗ Error calculating metrics: {e}")
    
    return pd.DataFrame(metrics_list)

def plot_comprehensive_comparison(df, model_data):
    """Create all comparison plots"""
    
    print("\n" + "=" * 70)
    print("GENERATING VISUALIZATIONS")
    print("=" * 70)
    
    # 1. ROC CURVES
    print("\n1. Generating ROC Curves...")
    plt.figure(figsize=(12, 8))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for idx, (_, row) in enumerate(df.iterrows()):
        model_name = row['model']
        
        # Try to load pre-computed ROC curve
        fpr_path = os.path.join(RESULTS_DIR, f"{model_name}_fpr.npy")
        tpr_path = os.path.join(RESULTS_DIR, f"{model_name}_tpr.npy")
        
        if os.path.exists(fpr_path) and os.path.exists(tpr_path):
            fpr = np.load(fpr_path)
            tpr = np.load(tpr_path)
        else:
            # Calculate from scores
            data = model_data[model_name]
            fpr, tpr, _ = roc_curve(data['labels'], data['scores'])
        
        auc_val = row['roc_auc']
        plt.plot(fpr, tpr, linewidth=3, alpha=0.8, color=colors[idx],
                label=f"{model_name} (AUC = {auc_val:.3f})")
    
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.5, linewidth=2, label="Random")
    plt.xlabel("False Positive Rate", fontsize=14)
    plt.ylabel("True Positive Rate", fontsize=14)
    plt.title("ROC Curves - Brain Tumor Anomaly Detection", fontsize=16, fontweight='bold')
    plt.grid(True, alpha=0.3, linestyle='--')
    plt.legend(loc="lower right", fontsize=12)
    plt.tight_layout()
    
    roc_path = os.path.join(RESULTS_DIR, "all_models_roc_curves.png")
    plt.savefig(roc_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to: {roc_path}")
    plt.show()
    
    # 2. METRIC COMPARISON BAR CHART
    print("\n2. Generating Metric Comparison Bar Chart...")
    metrics_to_plot = ['roc_auc', 'precision', 'recall', 'f1', 'accuracy']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    
    for idx, metric in enumerate(metrics_to_plot):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        df_sorted = df.sort_values(metric, ascending=True)
        
        y_pos = np.arange(len(df_sorted))
        colors_bars = plt.cm.viridis(np.linspace(0.2, 0.8, len(df_sorted)))
        
        bars = ax.barh(y_pos, df_sorted[metric], color=colors_bars)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(df_sorted['model'], fontsize=10)
        ax.set_xlabel(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_title(f"{metric.replace('_', ' ').title()} Comparison", fontsize=12, fontweight='bold')
        ax.set_xlim(0, 1)
        
        # Add values
        for bar, value in zip(bars, df_sorted[metric]):
            width = bar.get_width()
            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
                   f'{value:.3f}', ha='left', va='center', fontsize=10)
    
    # 6th subplot: Model ranking
    ax = axes[5]
    df['avg_score'] = df[['roc_auc', 'precision', 'recall', 'f1', 'accuracy']].mean(axis=1)
    df_sorted = df.sort_values('avg_score', ascending=True)
    
    y_pos = np.arange(len(df_sorted))
    colors_bars = plt.cm.plasma(np.linspace(0.2, 0.8, len(df_sorted)))
    
    bars = ax.barh(y_pos, df_sorted['avg_score'], color=colors_bars)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted['model'], fontsize=10)
    ax.set_xlabel("Average Score", fontsize=11)
    ax.set_title("Overall Model Ranking", fontsize=12, fontweight='bold')
    ax.set_xlim(0, 1)
    
    for bar, value in zip(bars, df_sorted['avg_score']):
        width = bar.get_width()
        ax.text(width + 0.01, bar.get_y() + bar.get_height()/2,
               f'{value:.3f}', ha='left', va='center', fontsize=10)
    
    plt.suptitle("Model Performance Comparison", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    bar_path = os.path.join(RESULTS_DIR, "all_models_metric_comparison.png")
    plt.savefig(bar_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to: {bar_path}")
    plt.show()
    
    # 3. CONFUSION MATRICES
    print("\n3. Generating Confusion Matrices...")
    n_models = len(df)
    n_cols = 3
    n_rows = (n_models + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_models == 1:
        axes = np.array([axes])
    axes = axes.flatten()
    
    for i, (_, row) in enumerate(df.iterrows()):
        ax = axes[i]
        
        cm = np.array([[row['tn'], row['fp']],
                      [row['fn'], row['tp']]])
        
        im = ax.imshow(cm, cmap='YlOrRd', aspect='auto')
        
        for r in range(2):
            for c in range(2):
                text_color = 'white' if cm[r, c] > cm.max() / 2 else 'black'
                ax.text(c, r, f"{cm[r, c]:,}", 
                       ha='center', va='center', 
                       color=text_color, fontsize=12, fontweight='bold')
        
        ax.set_xticks([0, 1])
        ax.set_yticks([0, 1])
        ax.set_xticklabels(['Normal', 'Tumor'])
        ax.set_yticklabels(['Normal', 'Tumor'])
        ax.set_xlabel('Predicted', fontsize=10)
        ax.set_ylabel('Actual', fontsize=10)
        ax.set_title(f"{row['model']}\n"
                    f"P: {row['precision']:.3f}, R: {row['recall']:.3f}, F1: {row['f1']:.3f}", 
                    fontsize=11)
        ax.grid(False)
    
    # Hide unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])
    
    plt.suptitle("Confusion Matrices - All Models", fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    cm_path = os.path.join(RESULTS_DIR, "all_models_confusion_matrices.png")
    plt.savefig(cm_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to: {cm_path}")
    plt.show()
    
    # 4. PERFORMANCE SUMMARY DASHBOARD
    print("\n4. Generating Performance Summary Dashboard...")
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Top-left: Precision-Recall scatter
    ax = axes[0, 0]
    for idx, (_, row) in enumerate(df.iterrows()):
        ax.scatter(row['recall'], row['precision'], 
                  s=300, alpha=0.7, label=row['model'],
                  edgecolors='black', linewidth=2)
        ax.annotate(row['model'], (row['recall'], row['precision']),
                   xytext=(8, 8), textcoords='offset points', fontsize=10)
    ax.set_xlabel("Recall", fontsize=12)
    ax.set_ylabel("Precision", fontsize=12)
    ax.set_title("Precision-Recall Trade-off", fontsize=13, fontweight='bold')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    
    # Top-right: Score distributions
    ax = axes[0, 1]
    for model_name, data in model_data.items():
        scores = data['scores']
        labels = data['labels']
        
        normal_scores = scores[labels == 0]
        tumor_scores = scores[labels == 1]
        
        ax.hist(normal_scores, bins=30, alpha=0.5, density=True, 
               label=f"{model_name} - Normal" if model_name == MODELS[0] else "")
        ax.hist(tumor_scores, bins=30, alpha=0.5, density=True,
               label=f"{model_name} - Tumor" if model_name == MODELS[0] else "")
    
    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title("Anomaly Score Distributions", fontsize=13, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    # Bottom-left: Radar chart
    ax = axes[1, 0]
    metrics_radar = ['precision', 'recall', 'f1', 'accuracy', 'specificity']
    n_metrics = len(metrics_radar)
    
    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]
    
    ax = plt.subplot(2, 2, 3, polar=True)
    
    colors_radar = plt.cm.Set2(np.linspace(0, 1, len(df)))
    
    for idx, (_, row) in enumerate(df.iterrows()):
        values = [row[metric] for metric in metrics_radar]
        values += values[:1]
        
        ax.plot(angles, values, 'o-', linewidth=2, label=row['model'], color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.1, color=colors_radar[idx])
    
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.capitalize() for m in metrics_radar], fontsize=10)
    ax.set_ylim(0, 1)
    ax.set_title("Performance Radar Chart", fontsize=13, fontweight='bold', pad=20)
    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1), fontsize=10)
    
    # Bottom-right: Model ranking
    ax = axes[1, 1]
    df['composite_score'] = (df['roc_auc'] + df['f1'] + df['accuracy']) / 3
    df_ranked = df.sort_values('composite_score', ascending=False)
    
    x = np.arange(len(df_ranked))
    width = 0.2
    
    metrics_bars = ['roc_auc', 'f1', 'accuracy']
    colors_bars = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    
    for i, (metric, color) in enumerate(zip(metrics_bars, colors_bars)):
        ax.bar(x + i*width - width, df_ranked[metric], width, 
              label=metric.upper(), color=color, alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels(df_ranked['model'], rotation=45, ha='right', fontsize=11)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Model Ranking by Composite Score", fontsize=13, fontweight='bold')
    ax.set_ylim(0, 1)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.suptitle("Brain Tumor Anomaly Detection - Performance Dashboard", 
                fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    dashboard_path = os.path.join(RESULTS_DIR, "all_models_performance_dashboard.png")
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved to: {dashboard_path}")
    plt.show()
    
    print("\n✓ All visualizations generated successfully!")

def save_results(df):
    """Save comprehensive results"""
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)
    
    # Save to CSV
    csv_path = os.path.join(RESULTS_DIR, "model_comparison_results.csv")
    df.to_csv(csv_path, index=False)
    print(f"✓ CSV saved to: {csv_path}")
    
    # Save to Excel with formatting
    excel_path = os.path.join(RESULTS_DIR, "model_comparison_results.xlsx")
    with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
        # Main metrics
        df.to_excel(writer, sheet_name='Metrics', index=False)
        
        # Create ranking sheet
        df_ranked = df.copy()
        for metric in ['roc_auc', 'precision', 'recall', 'f1', 'accuracy']:
            df_ranked[f'{metric}_rank'] = df_ranked[metric].rank(ascending=False, method='min')
        
        df_ranked.to_excel(writer, sheet_name='Rankings', index=False)
        
        # Create summary statistics
        summary = pd.DataFrame({
            'Metric': ['ROC-AUC', 'Precision', 'Recall', 'F1-Score', 'Accuracy'],
            'Best Model': [df.loc[df['roc_auc'].idxmax(), 'model'],
                          df.loc[df['precision'].idxmax(), 'model'],
                          df.loc[df['recall'].idxmax(), 'model'],
                          df.loc[df['f1'].idxmax(), 'model'],
                          df.loc[df['accuracy'].idxmax(), 'model']],
            'Best Value': [df['roc_auc'].max(),
                          df['precision'].max(),
                          df['recall'].max(),
                          df['f1'].max(),
                          df['accuracy'].max()],
            'Average': [df['roc_auc'].mean(),
                       df['precision'].mean(),
                       df['recall'].mean(),
                       df['f1'].mean(),
                       df['accuracy'].mean()],
            'Std Dev': [df['roc_auc'].std(),
                       df['precision'].std(),
                       df['recall'].std(),
                       df['f1'].std(),
                       df['accuracy'].std()]
        })
        summary.to_excel(writer, sheet_name='Summary', index=False)
    
    print(f"✓ Excel file saved to: {excel_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("PERFORMANCE SUMMARY")
    print("=" * 70)
    
    print("\n🏆 Model Rankings:")
    for metric in ['roc_auc', 'precision', 'recall', 'f1', 'accuracy']:
        best_idx = df[metric].idxmax()
        best_model = df.loc[best_idx, 'model']
        best_value = df.loc[best_idx, metric]
        print(f"  {metric.upper():12s}: {best_model:15s} = {best_value:.4f}")
    
    print("\n📊 Average Performance:")
    for metric in ['roc_auc', 'precision', 'recall', 'f1', 'accuracy']:
        avg = df[metric].mean()
        std = df[metric].std()
        print(f"  {metric.upper():12s}: {avg:.4f} ± {std:.4f}")

def main():
    """Main function"""
    print("=" * 80)
    print("BRAIN TUMOR ANOMALY DETECTION - MODEL COMPARISON")
    print("=" * 80)
    
    # Load model data
    model_data = load_model_data()
    
    if not model_data:
        print("No model data loaded. Check your results directory.")
        return
    
    print(f"\n✓ Successfully loaded data for {len(model_data)} models")
    
    # Calculate metrics
    df_metrics = calculate_all_metrics(model_data)
    
    if df_metrics.empty:
        print("No metrics calculated. Check your data.")
        return
    
    # Generate visualizations
    plot_comprehensive_comparison(df_metrics, model_data)
    
    # Save results
    save_results(df_metrics)
    
    print("\n" + "=" * 80)
    print("COMPARISON COMPLETE!")
    print("=" * 80)
    print(f"\nResults saved to: {os.path.abspath(RESULTS_DIR)}")
    print("\nGenerated files:")
    print("  - all_models_roc_curves.png")
    print("  - all_models_metric_comparison.png")
    print("  - all_models_confusion_matrices.png")
    print("  - all_models_performance_dashboard.png")
    print("  - model_comparison_results.csv")
    print("  - model_comparison_results.xlsx")

if __name__ == "__main__":
    main()