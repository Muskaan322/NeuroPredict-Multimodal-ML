### NeuroPredict: Multimodal ML for Neuromodulation Response
## 📌 Overview

NeuroPredict is a research-oriented machine learning pipeline designed to predict patient responsiveness to neuromodulation therapy. The project implements a multimodal approach, integrating neuroimaging (DTI), neurophysiology (EMG), biomechanics (Kinematics), and clinical demographics to identify the most significant biomarkers for recovery.

The highlight of this project is the Modality Ablation Study, which scientifically measures the performance drop when specific data sources are removed, helping researchers determine which clinical tests provide the most predictive value.

## 🧬 Multimodal Features

The pipeline processes four distinct data modalities:

DTI (Diffusion Tensor Imaging): FA (Fractional Anisotropy), MD (Mean Diffusivity), and CST asymmetry.

EMG (Neurophysiology): MEP amplitudes, H-reflex ratios, and voluntary RMS activity.

Kinematics (Biomechanics): Gait speed, step symmetry, range of motion, and reaction time.

Clinical Profile: Age, injury type (SCI incomplete/complete, Stroke), and time since injury.

## 🛠️ Technical Workflow

Synthetic Data Generation: Simulates a biologically plausible dataset (𝑁 =200) based on patterns found in published neurorehabilitation literature.

Preprocessing: Automated one-hot encoding for categorical variables and StandardScaler integration within Scikit-Learn pipelines.

Cross-Validation: Evaluates Logistic Regression, Random Forest, and Gradient Boosting using 5-fold Stratified Cross-Validation.

Permutation Importance: Calculates the contribution of each individual feature to the model's predictive power.

Ablation Analysis: Systematically removes each modality to quantify its unique contribution to the total AUC.

The script will generate the dataset, train the models, and save all results to an /outputs folder.

## 📊 Visual Analysis & Results

### 1. Model Benchmarking
We compared Logistic Regression (LR), Random Forest (RF), and Gradient Boosting (GB). While RF and GB show signs of overfitting on training data (AUC=1.0), the 5-fold cross-validation reveals a more realistic generalization performance.

![Model Comparison](images/fig1_model_comparison.png)

### 2. Feature Importance & Modalities
The analysis identifies **Kinematics** (Step symmetry and Gait speed) as the most influential modality, followed by **DTI** (CST integrity). This suggests that biomechanical performance is a strong indicator of overall neuromodulation response.

![Feature Importance](images/fig2_feature_importance.png)

### 3. Biomarker Distributions
The following histograms show the separation between responders and non-responders across key biomarkers. Notably, responders tend to have higher **FA (CST integrity)** and higher **MEP amplitudes**.

![Biomarker Distributions](images/fig3_biomarker_distributions.png)

### 4. Feature Correlations
The multimodal correlation matrix shows the relationships between neuroimaging, neurophysiology, and clinical scores, helping identify redundant features in the pipeline.

![Correlation Matrix](images/fig4_correlation_heatmap.png)

### 5. Modality Ablation Study
The ablation study quantifies the "value-add" of each data type. Removing **Kinematics** causes the most significant drop in model performance, highlighting its critical role in the predictive pipeline.

![Modality Ablation](images/fig5_modality_ablation.png)

📂 Project Structure

main.py: The complete executable pipeline (data gen, training, and plotting).

data/: Contains the generated neuro_multimodal.csv.

outputs/: Stores PNG figures and the model_summary.json report.
