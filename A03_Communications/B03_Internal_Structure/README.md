# B03_Internal_Structure

This module analyzes the internal representational structure of speaker and listener networks in the CATS-Net communication framework. It focuses on understanding how concepts are organized and clustered within the neural representations, and how these structures compare between speakers and listeners.

## Overview

The module performs three main types of analyses:
1. **Concept-to-Graph Conversion**: Transforms neural representations into graph structures for network analysis
2. **Representational Distance Matrix (RDM) Analysis**: Compares similarity structures between speakers and listeners
3. **Statistical Testing**: Validates communication accuracy and representational similarity

## Files Description

### Main Analysis Scripts

#### `C01_cifar100_concept2graph.m`
**Purpose**: Converts concept representations into graph structures for network visualization and analysis.

**Functionality**:
- Performs hierarchical clustering on concept representations using cosine distance
- Creates adjacency matrices based on clustering linkage
- Generates node and edge information files for Gephi visualization
- Can analyze either speaker representations or specific listener representations
- Uses average linkage clustering with minimum distance criterion

**Key Parameters**:
- `listener_idx`: Set to 0 for speaker analysis, 1-100 for specific listeners
- `method_linkage`: Clustering linkage method (default: 'average')
- `link_crit`: Link selection criterion (default: 'min')

**Outputs**:
- `Nods_information.csv`: Node information with cluster assignments
- `Edge_information.csv`: Edge weights and connections
- Dendrogram visualization
- Graph files for Gephi (`.gephi` format)

#### `C02_cifar100_RDM_speaker_listener.m`
**Purpose**: Compares representational distance matrices (RDMs) between speakers and listeners to measure structural similarity.

**Functionality**:
- Computes RDMs for both speaker and listener representations
- Supports both cosine distance and Pearson correlation metrics
- Calculates Spearman correlations between speaker and listener RDMs
- Performs statistical testing (one-sample t-test) against similarity threshold
- Processes all 100 listeners systematically

**Key Features**:
- Cross-validation approach: excludes target listener from speaker data
- Statistical significance testing with Cohen's d effect size
- Customizable similarity threshold (default: 0.35)
- Batch processing for all listener models

**Outputs**:
- `speaker_listener_correlations.csv`: Correlation values for all listeners
- Statistical test results with confidence intervals

#### `C03_cifar100_comm_acc_ttest.m`
**Purpose**: Performs statistical analysis on communication accuracy data.

**Functionality**:
- Reads communication accuracy data from CSV files
- Performs one-sample t-test against chance level (0.5)
- Calculates effect size (Cohen's d)
- Provides detailed statistical interpretation

**Input Data**: `../../Results/single_ct_acc.csv` (best_acc_2 column)

### Utility Functions

#### `cluster_search.m`
**Purpose**: Recursive function to identify leaf nodes in hierarchical clustering trees.

**Functionality**:
- Searches clustering linkage matrices to find terminal nodes
- Used by the graph construction algorithm
- Handles nested cluster structures recursively

#### `Matrix2List.m`
**Purpose**: Converts square matrices to vector format by extracting lower triangular elements.

**Usage**: Commonly used for RDM analysis to convert distance matrices to vectors for correlation analysis.

### Data Files

#### Core Data
- `cifar100_ss20_ni1e-1_ychen_trail1.mat`: Speaker concept representations (CIFAR-100 classes)
- `meta.mat`: Metadata including class label names
- `vik.mat`: Color map for visualizations

#### Results and Outputs
- `speaker_listener_correlations.csv`: RDM correlation results (100 values, one per listener)
- `Nods_information.csv`: Node information for graph visualization (ID, Label, Modularity Class)
- `Edge_information.csv`: Edge information for graph visualization (Source, Target, Weight)

#### Network Visualizations
- `graph_speaker.gephi`: Speaker concept network for Gephi visualization
- `graph_listener_1.gephi`: Example listener concept network

## Usage Workflow

### 1. Basic Concept Graph Analysis
```matlab
% Run concept-to-graph conversion for speaker
C01_cifar100_concept2graph.m

% This will generate:
% - Nods_information.csv
% - Edge_information.csv
% - Dendrogram plot
```

### 2. RDM Similarity Analysis
```matlab
% Compare speaker-listener representational structures
C02_cifar100_RDM_speaker_listener.m

% This will:
% - Process all 100 listeners
% - Generate correlation statistics
% - Perform significance testing
```

### 3. Communication Accuracy Testing
```matlab
% Test communication performance
C03_cifar100_comm_acc_ttest.m

% Requires: ../../Results/single_ct_acc.csv
```

## Key Analysis Parameters

### Clustering Parameters
- **Distance Metric**: Cosine distance (default for concept similarity)
- **Linkage Method**: Average linkage clustering
- **Clustering Cutoff**: 
  - Speaker: 0.3785
  - Listeners: 0.31

### Statistical Testing
- **RDM Similarity Threshold**: 0.35
- **Communication Accuracy Threshold**: 0.5 (chance level)
- **Test Type**: One-sample t-test (right-tailed)
- **Significance Level**: α = 0.05

## Dependencies

### MATLAB Toolboxes Required
- Statistics and Machine Learning Toolbox (for clustering and statistical tests)
- Bioinformatics Toolbox (for dendrogram visualization)

### External Dependencies
- Gephi (for network visualization of generated `.gephi` files)
- Speaker representations: `cifar100_ss20_ni1e-1_ychen_trail1.mat`
- Listener models: `../B02_Communication_Game/Symbol_and_Model_of_Listener/contexts/context_id_*_e_1999.mat`

## Output Interpretation

### Graph Analysis
- **Nodes**: Represent CIFAR-100 concept classes
- **Edges**: Weighted by representational similarity (1 - cosine distance)
- **Clusters**: Groups of conceptually similar representations
- **Modularity Classes**: Community structure in the concept space

### RDM Correlations
- **Range**: [0, 1] where higher values indicate more similar representational structures
- **Significance**: Values > 0.35 suggest meaningful structural similarity
- **Cross-listener Variability**: Indicates consistency of learned representations

### Statistical Results
- **t-statistics**: Measure of effect magnitude
- **p-values**: Statistical significance
- **Cohen's d**: Effect size interpretation
- **Confidence Intervals**: Precision of estimates

## Research Applications

This module supports research into:
- **Emergent Communication**: How structured representations arise through communication
- **Representational Alignment**: Similarity between speaker and listener concept spaces
- **Network Analysis**: Graph-theoretic properties of concept representations
- **Communication Efficiency**: Relationship between representational structure and communication success

## Notes

- The analysis excludes self-communication (listener cannot be the same as the speaker index)
- Cosine distance is preferred for high-dimensional neural representations
- Statistical tests account for multiple comparisons through appropriate thresholding
- Graph visualizations can be imported into Gephi for interactive exploration 