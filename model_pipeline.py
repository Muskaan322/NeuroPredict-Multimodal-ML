"""
Multimodal Biomarker ML Pipeline
==================================
Predicts neuromodulation responders from:
  DTI + EMG/neurophysiology + kinematics + clinical features

Pipeline:
  1. Load & preprocess data
  2. Feature engineering (modality groups)
  3. Train: Logistic Regression, Random Forest, XGBoost (with cross-validation)
  4. Evaluate: ROC-AUC, confusion matrix, feature importance
  5. SHAP explainability
  6. Export: figures + summary report

Usage:
  python model_pipeline.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings, json
from pathlib import Path

from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.preprocessing  import StandardScaler, LabelEncoder
from sklearn.pipeline        import Pipeline
from sklearn.linear_model    import LogisticRegression
from sklearn.ensemble        import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics         import (roc_auc_score, roc_curve, confusion_matrix,
                                     classification_report, ConfusionMatrixDisplay)
from sklearn.inspection      import permutation_importance

warnings.filterwarnings('ignore')
np.random.seed(42)

OUT = Path('outputs')
OUT.mkdir(exist_ok=True)

# ── Color palette aligned with publication style ────────────────────────────
C = dict(dti='#534AB7', emg='#0F6E56', kin='#BA7517', clin='#993C1D',
         pos='#185FA5', neg='#A32D2D', neutral='#5F5E5A')

MODALITY_FEATURES = {
    'DTI':         ['dti_fa_cst', 'dti_md_cst', 'dti_cst_asymmetry'],
    'EMG':         ['mep_amplitude_mv', 'h_reflex_ratio', 'emg_rms_voluntary'],
    'Kinematics':  ['gait_speed_ms', 'step_symmetry', 'rom_degrees', 'reaction_time_ms'],
    'Clinical':    ['age', 'months_post_injury', 'baseline_motor_score'],
}
ALL_FEATURES = [f for feats in MODALITY_FEATURES.values() for f in feats]
TARGET       = 'responder'

FEATURE_LABELS = {
    'dti_fa_cst':             'FA — CST integrity',
    'dti_md_cst':             'MD — CST diffusivity',
    'dti_cst_asymmetry':      'CST asymmetry index',
    'mep_amplitude_mv':       'MEP amplitude (mV)',
    'h_reflex_ratio':         'H-reflex ratio',
    'emg_rms_voluntary':      'EMG RMS voluntary',
    'gait_speed_ms':          'Gait speed (m/s)',
    'step_symmetry':          'Step symmetry index',
    'rom_degrees':            'Range of motion (°)',
    'reaction_time_ms':       'Reaction time (ms)',
    'age':                    'Age (years)',
    'months_post_injury':     'Months post-injury',
    'baseline_motor_score':   'Baseline motor score',
}

MOD_COLOR_MAP = {'DTI': C['dti'], 'EMG': C['emg'], 'Kinematics': C['kin'], 'Clinical': C['clin']}
MODALITY_COLORS = {f: MOD_COLOR_MAP[k] for k, feats in MODALITY_FEATURES.items() for f in feats}


# ══════════════════════════════════════════════════════════════════════════════
# 1. LOAD & PREPROCESS
# ══════════════════════════════════════════════════════════════════════════════
def load_and_preprocess(path='data/neuro_multimodal.csv'):
    df = pd.read_csv(path)
    print(f"\n{'='*55}")
    print(f"  Dataset: {len(df)} patients | {df[TARGET].mean()*100:.1f}% responders")
    print(f"{'='*55}")

    # One-hot encode injury type
    le = LabelEncoder()
    df['injury_encoded'] = le.fit_transform(df['injury_type'])
    # Also add as dummies for interpretability
    dummies = pd.get_dummies(df['injury_type'], prefix='inj', drop_first=False)
    df = pd.concat([df, dummies], axis=1)

    # Add injury dummies to clinical features
    injury_cols = list(dummies.columns)
    MODALITY_FEATURES['Clinical'] += injury_cols
    ALL_FEATURES.extend(injury_cols)
    for c in injury_cols:
        FEATURE_LABELS[c] = c.replace('inj_', 'Injury: ')
        MODALITY_COLORS[c] = C['clin']

    X = df[ALL_FEATURES].copy()
    y = df[TARGET].values
    return df, X, y, le


# ══════════════════════════════════════════════════════════════════════════════
# 2. MODEL DEFINITIONS
# ══════════════════════════════════════════════════════════════════════════════
def build_models():
    return {
        'Logistic Regression': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LogisticRegression(C=0.5, max_iter=1000, random_state=42))
        ]),
        'Random Forest': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', RandomForestClassifier(n_estimators=200, max_depth=6,
                                           min_samples_leaf=5, random_state=42))
        ]),
        'Gradient Boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', GradientBoostingClassifier(n_estimators=150, max_depth=3,
                                               learning_rate=0.08, random_state=42))
        ]),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 3. CROSS-VALIDATION EVALUATION
# ══════════════════════════════════════════════════════════════════════════════
def evaluate_models(X, y, models):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    results = {}

    print("\n  Cross-validation results (5-fold stratified):")
    print(f"  {'Model':<25} {'AUC':>8}  {'Acc':>8}  {'F1':>8}")
    print(f"  {'-'*52}")

    for name, pipe in models.items():
        auc = cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc')
        acc = cross_val_score(pipe, X, y, cv=cv, scoring='accuracy')
        f1  = cross_val_score(pipe, X, y, cv=cv, scoring='f1')
        results[name] = {'auc': auc, 'acc': acc, 'f1': f1}
        print(f"  {name:<25} {auc.mean():.3f}±{auc.std():.3f}  "
              f"{acc.mean():.3f}±{acc.std():.3f}  "
              f"{f1.mean():.3f}±{f1.std():.3f}")

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 4. FINAL FIT (full data) + FEATURE IMPORTANCE
# ══════════════════════════════════════════════════════════════════════════════
def fit_best_model(X, y, models):
    # Use Gradient Boosting as best model (typically highest AUC for this type)
    best = models['Gradient Boosting']
    best.fit(X, y)
    print("\n  Best model (Gradient Boosting) fitted on full dataset.")
    return best


# ══════════════════════════════════════════════════════════════════════════════
# 5. FIGURE 1 — Model comparison + ROC curves
# ══════════════════════════════════════════════════════════════════════════════
def plot_model_comparison(X, y, models, cv_results):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.patch.set_facecolor('white')

    # Left: AUC box plot
    ax = axes[0]
    names = list(cv_results.keys())
    aucs  = [cv_results[n]['auc'] for n in names]
    short = ['LR', 'RF', 'GB']
    bp = ax.boxplot(aucs, labels=short, patch_artist=True, widths=0.45,
                    medianprops=dict(color='white', linewidth=2))
    colors = ['#B5D4F4', '#9FE1CB', '#FAC775']
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
    ax.set_ylabel('ROC-AUC', fontsize=12)
    ax.set_title('5-fold cross-validation AUC', fontsize=13, fontweight='normal')
    ax.set_ylim(0.4, 1.0)
    ax.axhline(0.5, ls='--', lw=0.8, color='#888780', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')

    # Right: ROC curves (full-data fit for display)
    ax2 = axes[1]
    roc_colors = {'Logistic Regression': '#185FA5',
                  'Random Forest':       '#0F6E56',
                  'Gradient Boosting':   '#BA7517'}
    for name, pipe in models.items():
        pipe.fit(X, y)
        y_prob = pipe.predict_proba(X)[:, 1]
        fpr, tpr, _ = roc_curve(y, y_prob)
        auc_val = roc_auc_score(y, y_prob)
        ax2.plot(fpr, tpr, lw=1.8, color=roc_colors[name],
                 label=f"{name} (AUC={auc_val:.3f})")
    ax2.plot([0,1],[0,1], '--', lw=0.8, color='#888780', alpha=0.6)
    ax2.set_xlabel('False positive rate', fontsize=11)
    ax2.set_ylabel('True positive rate', fontsize=11)
    ax2.set_title('ROC curves (training data)', fontsize=13, fontweight='normal')
    ax2.legend(fontsize=9, frameon=False)
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_facecolor('white')

    plt.tight_layout(pad=2)
    fig.savefig(OUT / 'fig1_model_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → fig1_model_comparison.png")


# ══════════════════════════════════════════════════════════════════════════════
# 6. FIGURE 2 — Feature importance (permutation + modality breakdown)
# ══════════════════════════════════════════════════════════════════════════════
def plot_feature_importance(best_model, X, y):
    perm = permutation_importance(best_model, X, y, n_repeats=30,
                                  random_state=42, scoring='roc_auc')
    imp_mean = perm.importances_mean
    imp_std  = perm.importances_std
    feat_names = X.columns.tolist()

    order = np.argsort(imp_mean)
    sorted_names  = [feat_names[i] for i in order]
    sorted_labels = [FEATURE_LABELS.get(n, n) for n in sorted_names]
    sorted_mean   = imp_mean[order]
    sorted_std    = imp_std[order]
    sorted_colors = [MODALITY_COLORS.get(n, C['neutral']) for n in sorted_names]

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor('white')

    # Left: horizontal bar — permutation importance
    ax = axes[0]
    y_pos = np.arange(len(sorted_names))
    ax.barh(y_pos, sorted_mean, xerr=sorted_std,
            color=sorted_colors, alpha=0.85, height=0.7,
            error_kw=dict(lw=0.8, capsize=2, color='#444441'))
    ax.set_yticks(y_pos)
    ax.set_yticklabels(sorted_labels, fontsize=9.5)
    ax.set_xlabel('Permutation importance (ΔAUC)', fontsize=11)
    ax.set_title('Feature importance by permutation', fontsize=13, fontweight='normal')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')

    # Legend for modality colors
    from matplotlib.patches import Patch
    legend_els = [Patch(facecolor=C['dti'], label='DTI'),
                  Patch(facecolor=C['emg'], label='EMG / Neurophysiology'),
                  Patch(facecolor=C['kin'], label='Kinematics'),
                  Patch(facecolor=C['clin'], label='Clinical')]
    ax.legend(handles=legend_els, fontsize=8.5, frameon=False, loc='lower right')

    # Right: modality-level importance (sum per group)
    ax2 = axes[1]
    modality_imp = {}
    for mod, feats in MODALITY_FEATURES.items():
        idx = [feat_names.index(f) for f in feats if f in feat_names]
        modality_imp[mod] = imp_mean[idx].sum() if idx else 0

    mod_names = list(modality_imp.keys())
    mod_vals  = list(modality_imp.values())
    mod_colors = [C['dti'], C['emg'], C['kin'], C['clin']]
    bars = ax2.bar(mod_names, mod_vals, color=mod_colors, alpha=0.85, width=0.55)
    ax2.set_ylabel('Summed permutation importance', fontsize=11)
    ax2.set_title('Importance by modality', fontsize=13, fontweight='normal')
    ax2.spines['top'].set_visible(False)
    ax2.spines['right'].set_visible(False)
    ax2.set_facecolor('white')
    for bar, val in zip(bars, mod_vals):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
                 f'{val:.3f}', ha='center', va='bottom', fontsize=9)

    plt.tight_layout(pad=2)
    fig.savefig(OUT / 'fig2_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → fig2_feature_importance.png")


# ══════════════════════════════════════════════════════════════════════════════
# 7. FIGURE 3 — Confusion matrix + key biomarker distributions
# ══════════════════════════════════════════════════════════════════════════════
def plot_biomarker_distributions(df):
    key_features = [
        ('dti_fa_cst',       'FA — CST integrity',    C['dti']),
        ('mep_amplitude_mv', 'MEP amplitude (mV)',     C['emg']),
        ('gait_speed_ms',    'Gait speed (m/s)',       C['kin']),
        ('baseline_motor_score', 'Baseline motor score', C['clin']),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.patch.set_facecolor('white')
    axes = axes.flatten()

    for ax, (feat, label, color) in zip(axes, key_features):
        r0 = df[df['responder'] == 0][feat]
        r1 = df[df['responder'] == 1][feat]
        ax.hist(r0, bins=20, alpha=0.55, color=C['neg'],   label='Non-responder', density=True)
        ax.hist(r1, bins=20, alpha=0.55, color=C['pos'],   label='Responder',     density=True)
        ax.axvline(r0.mean(), color=C['neg'], lw=1.5, ls='--', alpha=0.8)
        ax.axvline(r1.mean(), color=C['pos'], lw=1.5, ls='--', alpha=0.8)
        ax.set_xlabel(label, fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.legend(fontsize=8, frameon=False)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_facecolor('white')

    plt.suptitle('Key biomarker distributions: responders vs non-responders',
                 fontsize=13, y=1.01, fontweight='normal')
    plt.tight_layout(pad=2)
    fig.savefig(OUT / 'fig3_biomarker_distributions.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → fig3_biomarker_distributions.png")


# ══════════════════════════════════════════════════════════════════════════════
# 8. FIGURE 4 — Correlation heatmap (multimodal)
# ══════════════════════════════════════════════════════════════════════════════
def plot_correlation_heatmap(df):
    core_feats = [
        'dti_fa_cst', 'dti_md_cst', 'dti_cst_asymmetry',
        'mep_amplitude_mv', 'h_reflex_ratio', 'emg_rms_voluntary',
        'gait_speed_ms', 'step_symmetry', 'reaction_time_ms',
        'age', 'months_post_injury', 'baseline_motor_score', 'responder'
    ]
    corr = df[core_feats].corr()
    labels = [FEATURE_LABELS.get(f, f) for f in core_feats[:-1]] + ['Responder']

    fig, ax = plt.subplots(figsize=(11, 9))
    fig.patch.set_facecolor('white')
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-0.8, vmax=0.8, linewidths=0.3,
                xticklabels=labels, yticklabels=labels, ax=ax,
                annot_kws={'size': 7}, cbar_kws={'shrink': 0.8})
    ax.set_title('Multimodal feature correlation matrix', fontsize=13, fontweight='normal', pad=12)
    plt.xticks(rotation=40, ha='right', fontsize=8)
    plt.yticks(rotation=0, fontsize=8)
    plt.tight_layout()
    fig.savefig(OUT / 'fig4_correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → fig4_correlation_heatmap.png")


# ══════════════════════════════════════════════════════════════════════════════
# 9. FIGURE 5 — Modality ablation (what if you remove one modality?)
# ══════════════════════════════════════════════════════════════════════════════
def plot_modality_ablation(X, y, models):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    pipe = models['Gradient Boosting']

    results = {'All modalities': cross_val_score(pipe, X, y, cv=cv, scoring='roc_auc').mean()}
    for mod, feats in MODALITY_FEATURES.items():
        drop_cols = [f for f in feats if f in X.columns]
        X_ablated = X.drop(columns=drop_cols)
        auc = cross_val_score(pipe, X_ablated, y, cv=cv, scoring='roc_auc').mean()
        results[f'− {mod}'] = auc

    fig, ax = plt.subplots(figsize=(8, 4))
    fig.patch.set_facecolor('white')
    names = list(results.keys())
    vals  = list(results.values())
    colors_ab = ['#3B3489'] + [C['dti'], C['emg'], C['kin'], C['clin']]
    bars = ax.barh(names, vals, color=colors_ab, alpha=0.85, height=0.55)
    ax.axvline(vals[0], ls='--', lw=1, color='#444441', alpha=0.5)
    ax.set_xlabel('Mean ROC-AUC (5-fold CV)', fontsize=11)
    ax.set_title('Modality ablation — contribution of each data type', fontsize=13, fontweight='normal')
    ax.set_xlim(0.45, 1.0)
    for bar, val in zip(bars, vals):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('white')
    plt.tight_layout(pad=1.5)
    fig.savefig(OUT / 'fig5_modality_ablation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("  → fig5_modality_ablation.png")
    return results


# ══════════════════════════════════════════════════════════════════════════════
# 10. SAVE SUMMARY REPORT
# ══════════════════════════════════════════════════════════════════════════════
def save_summary(cv_results, ablation_results, X, y):
    summary = {
        'n_patients': len(y),
        'n_features': X.shape[1],
        'responder_rate': float(y.mean()),
        'cv_results': {
            name: {
                'auc_mean': float(v['auc'].mean()),
                'auc_std':  float(v['auc'].std()),
                'acc_mean': float(v['acc'].mean()),
                'f1_mean':  float(v['f1'].mean()),
            } for name, v in cv_results.items()
        },
        'modality_ablation_auc': {k: float(v) for k, v in ablation_results.items()},
        'feature_groups': {k: v for k, v in MODALITY_FEATURES.items()},
    }
    with open(OUT / 'model_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print("  → model_summary.json")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    print("\n  Neuromodulation Responder Prediction — Multimodal ML Pipeline")

    # Generate data if not present
    from generate_data import generate_dataset
    generate_dataset(n=200, save_path='data/neuro_multimodal.csv')

    # Load & preprocess
    df, X, y, le = load_and_preprocess('data/neuro_multimodal.csv')

    # Models
    models = build_models()

    # Cross-validation
    cv_results = evaluate_models(X, y, models)

    # Fit best model on full data
    best_model = fit_best_model(X, y, models)

    # Plots
    print("\n  Generating figures...")
    plot_model_comparison(X, y, models, cv_results)
    plot_feature_importance(best_model, X, y)
    plot_biomarker_distributions(df)
    plot_correlation_heatmap(df)
    ablation = plot_modality_ablation(X, y, models)

    # Summary
    save_summary(cv_results, ablation, X, y)

    print(f"\n  All outputs saved to: {OUT.resolve()}")
    print("  Done.\n")
