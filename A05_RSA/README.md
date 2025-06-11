# A05_RSA - Representational Similarity Analysis

This directory contains MATLAB code for conducting Representational Similarity Analysis (RSA) on neuroimaging data, specifically analyzing the relationship between neural patterns and computational models of semantic representation.

## Overview

The RSA analysis pipeline is designed to:
1. Conduct both ROI-based and searchlight-based neural analyses (core components)
2. Compare neural representational dissimilarity matrices (RDMs) with computational model RDMs
3. Perform additional correlation analysis and model clustering
4. Evaluate model consensus through partial correlation analysis

## Core Analysis Components

## Directory Structure

### 1. `/ROI/` - Region of Interest Analysis (Core Component)
**Purpose**: Extracts RSA results from predefined brain regions to examine model-brain correlations within specific anatomical areas.

**Key Scripts**:
- `ROIAnalysis_RSA_Concept.m`: Main analysis for concept-level representations
  - Extracts spmT values from specified ROI masks
  - Computes brain RDMs using Pearson correlation
  - Calculates Spearman correlations with 30 computational models
  - Applies Fisher z-transformation to correlation coefficients
  - Performs both standard and partial correlation analysis (controlling for feature RDM)
- `ROIAnalysis_RSA_CA.m`: Similar analysis for category-level (CA) representations
- `Stat_test.ipynb`: Statistical testing and visualization of ROI results

**Output Files**:
- `RSA_ROI_Concept_results.csv`: Concept-level correlation results
- `RSA_ROI_CA_results.csv`: Category-level correlation results
- `featureRDM.mat`: Feature-based representational dissimilarity matrix

### 2. `/Searchlight/` - Whole-brain Searchlight Analysis (Core Component)
**Purpose**: Performs whole-brain RSA analysis using spherical searchlight neighborhoods to create spatial maps of model-brain correlations.

**Analysis Pipeline**:
- `A0_RSA_YuModel.m`: Main searchlight RSA function
  - Configurable searchlight radius (default: 10mm)
  - Processes multiple computational models simultaneously
  - Supports subject-specific analysis with flexible parameters
- `A0_RSA_one_subj_*.m`: Subject-specific analysis scripts for different model ranges (21-30, 31-40, 41-50)
- `A1_fisher_transform_all_*.m`: Fisher z-transformation of correlation maps
- `A2_Smooth_*.m`: Spatial smoothing of RSA correlation maps
- `A3_OneSampleModels.m`: Group-level one-sample t-tests for statistical inference

**Key Features**:
- Whole-brain coverage with voxel-wise analysis
- Spherical searchlight neighborhoods for local pattern analysis
- Multiple model processing with efficient memory management
- Statistical processing pipeline from individual to group level

## Additional Analysis Tools

### 3. `BinderRSA.m` - RSA Correlation Function
**Purpose**: Standalone function for computing RSA correlations between concept and semantic RDMs with flexible parameter control.

**Usage**: `[corr_results, p_values] = BinderRSA('RandomConcept', false, 'Alpha', 0.05)`

**Key Features**:
- Supports both real and random concept analysis for control conditions
- Uses Spearman correlation for RSA model-brain comparisons
- Processes models 1-30 with configurable parameters
- Outputs correlation results to CSV files in `outputs_wt95/` directory
- Includes statistical significance testing with configurable alpha levels

### 4. `kmeans_rsa_clustering.m` - Model Clustering Analysis
**Purpose**: Performs k-means clustering on model RDMs to identify consensus groups among computational models.

**Key Features**:
- Loads 30 computational models from TransE embeddings
- Computes partial correlations between models while controlling for feature RDMs
- Applies k-means clustering (k=2) to identify model consensus groups
- Classifies models into "high-consensus" and "low-consensus" clusters
- Outputs cluster assignments to `kmeans_clusters.csv`
- Saves inter-model distance matrix as `InterModelRDM.mat`

## Data Requirements

### Input Data
- **fMRI Data**: SPM T-statistic images (`spmT*.nii`) for each condition
- **Model RDMs**: Computational model dissimilarity matrices
  - `Yumodel_RDM_eval_TransE_21to30.mat`
  - `Yumodel_RDM_eval_TransE_31to40.mat`
  - `Yumodel_RDM_eval_TransE_41to50.mat`
- **Semantic Features**: `wt_65dim_rearrange.csv` containing 65-dimensional semantic features
- **Brain Masks**: ROI masks for region-specific analysis

### Dependencies
- SPM12 (Statistical Parametric Mapping)
- DPABI (Data Processing & Analysis for Brain Imaging)
- REST (RESTing-state fMRI data analysis toolkit)
- MATLAB Statistics and Machine Learning Toolbox

## Analysis Pipeline

### 1. Model Preparation
- Load computational models (TransE embeddings, models 21-50)
- Create representational dissimilarity matrices for each model
- Apply indexing to filter relevant concepts (excluding items 37, 41 from 95 total)

### 2. Brain Data Processing
- Extract neural activity patterns from fMRI data
- Compute brain RDMs using Pearson correlation
- Apply Fisher z-transformation to correlation values

### 3. RSA Computation
- **Searchlight Analysis**: Compute model-brain correlations within spherical neighborhoods
- **ROI Analysis**: Extract correlations within predefined brain regions
- Use Spearman correlation for model-brain comparisons
- Control for low-level features using partial correlation

### 4. Statistical Analysis
- Apply Fisher z-transformation to stabilize correlations
- Perform spatial smoothing on correlation maps
- Conduct group-level statistical tests
- Correct for multiple comparisons

### 5. Model Clustering
- Compute inter-model similarity using partial correlations
- Apply k-means clustering to identify model consensus groups
- Evaluate model agreement patterns

## Output Files

### Core Analysis Results
- `RSA_ROI_Concept_results.csv`: ROI-based concept-level correlation results
- `RSA_ROI_CA_results.csv`: ROI-based category-level correlation results
- Individual model correlation maps (`.nii` files): Searchlight RSA results

### Additional Analysis Results
- `correlation_results.csv`: BinderRSA correlation results
- `kmeans_clusters.csv`: Model cluster assignments
- `InterModelRDM.mat`: Inter-model distance matrix

## Usage Examples

### Core ROI Analysis
```matlab
% Run ROI-based RSA analysis (modify paths in script as needed)
run('ROI/ROIAnalysis_RSA_Concept.m');
run('ROI/ROIAnalysis_RSA_CA.m');
```

### Core Searchlight Analysis
```matlab
% Run searchlight RSA for subject 1
A0_RSA_YuModel(1, './lib/', 'SearchlightRadius', 10);

% Run with custom parameters
A0_RSA_YuModel(1, './lib/', 'SearchlightRadius', 8, 'ModelRange', 1:20);
```

### Additional RSA Analysis
```matlab
% Run RSA correlation analysis with default parameters
[corr_results, p_values] = BinderRSA();

% Run with random concepts as control
[corr_results, p_values] = BinderRSA('RandomConcept', true);
```

### Model Clustering Analysis
```matlab
% Run model clustering analysis
run('kmeans_rsa_clustering.m');
```

## Key Parameters

- **Searchlight Radius**: 10mm (configurable)
- **Correlation Type**: Spearman for model-brain comparisons, Pearson for brain RDMs
- **Model Range**: 30 computational models (indexed 1-30, corresponding to original models 21-50)
- **Concept Filter**: 93 concepts (95 total minus items 37, 41)
- **Clustering**: K-means with k=2 (high/low consensus groups)

## References

This analysis framework implements representational similarity analysis techniques for comparing neural and computational representations of semantic knowledge. The approach builds on established RSA methodologies while incorporating model clustering and consensus analysis.

## Notes

- Ensure all required toolboxes are installed and paths are correctly set
- Modify hardcoded paths in scripts to match your local directory structure
- Results are automatically saved to designated output directories
- Consider computational requirements for whole-brain searchlight analysis 