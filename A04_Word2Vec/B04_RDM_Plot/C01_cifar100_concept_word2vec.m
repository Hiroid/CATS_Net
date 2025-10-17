clear all;
clc;

load('vik.mat');

% Load class labels for proper labeling
class_label = load("meta.mat");

rdm_type = 'cosine'; % 'cosine' 'pearson'

wv = load('cifar100_ss20_wordvec.mat');
wv_data = struct2array(wv);

ct = load('cifar100_ss20_ni1e-1_ychen_trail1.mat');
ct_data = struct2array(ct);

% Clustering parameters
method_linkage = 'average';
cluster_num = 10;

st = 1;
ed = 100;

if strcmp(rdm_type, 'pearson')
    WordvecRDM = ones(100,100) - corr(wv_data','type','Pearson','rows','all','tail','both');
else
    Wordvec_dist_orig = pdist(wv_data, 'cosine');
    WordvecRDM = squareform(Wordvec_dist_orig);
end

if strcmp(rdm_type, 'pearson')
    ConceptRDM = ones(100,100) - corr(ct_data','type','Pearson','rows','all','tail','both');
else
    Concept_dist_orig = pdist(ct_data, 'cosine');
    ConceptRDM = squareform(Concept_dist_orig);
end

% Perform hierarchical clustering on concept data to get reordering
ct_link = linkage(ct_data, method_linkage, 'cosine');
cluster_ct = cluster(ct_link,'maxclust',cluster_num);
cutoff_ct = median([ct_link(end-cluster_num+1,3), ct_link(end-cluster_num+2,3)]);

% Get dendrogram ordering for reordering the matrices
[H_ct, T_ct, outperm_ct] = dendrogram(ct_link,0,'Orientation','left','ColorThreshold',cutoff_ct);
close; % Close the dendrogram figure as we only need the ordering

% Create class labels for reordered indices
tickLabel_reorder = cell(1,100);
for i=1:100
   fine_label = strrep(class_label.fine_label_names{outperm_ct(i)},'_',' ');
   tickLabel_reorder{i} = fine_label;
end

% Reorder both RDM matrices using the concept-based clustering ordering
ConceptRDM_reordered = ConceptRDM(outperm_ct, outperm_ct);
WordvecRDM_reordered = WordvecRDM(outperm_ct, outperm_ct);

% ========== Percentile Transformation for Enhanced Visualization ==========
% Function to apply percentile transformation to enhance visual contrast
percentile_transform = @(matrix) tiedrank(matrix(:)) / numel(matrix);

% Apply percentile transformation to concept-based reordered matrices
ConceptRDM_reordered_pct = reshape(percentile_transform(ConceptRDM_reordered), size(ConceptRDM_reordered));
WordvecRDM_reordered_pct = reshape(percentile_transform(WordvecRDM_reordered), size(WordvecRDM_reordered));

% ========== Word2Vec-based Clustering ==========
% Perform hierarchical clustering on word2vec data to get alternative reordering
wv_link = linkage(wv_data, method_linkage, 'cosine');
cluster_wv = cluster(wv_link,'maxclust',cluster_num);
cutoff_wv = median([wv_link(end-cluster_num+1,3), wv_link(end-cluster_num+2,3)]);

% Get dendrogram ordering for word2vec-based reordering
[H_wv, T_wv, outperm_wv] = dendrogram(wv_link,0,'Orientation','left','ColorThreshold',cutoff_wv);
close; % Close the dendrogram figure as we only need the ordering

% Create class labels for word2vec-based reordered indices
tickLabel_reorder_wv = cell(1,100);
for i=1:100
   fine_label = strrep(class_label.fine_label_names{outperm_wv(i)},'_',' ');
   tickLabel_reorder_wv{i} = fine_label;
end

% Reorder both RDM matrices using the word2vec-based clustering ordering
ConceptRDM_reordered_wv = ConceptRDM(outperm_wv, outperm_wv);
WordvecRDM_reordered_wv = WordvecRDM(outperm_wv, outperm_wv);

% Apply percentile transformation to word2vec-based reordered matrices
ConceptRDM_reordered_wv_pct = reshape(percentile_transform(ConceptRDM_reordered_wv), size(ConceptRDM_reordered_wv));
WordvecRDM_reordered_wv_pct = reshape(percentile_transform(WordvecRDM_reordered_wv), size(WordvecRDM_reordered_wv));

% ========== Enhanced Visualization with Percentile Transformation ==========
figure('Position', [100, 100, 1200, 500]);

subplot(1, 2, 1);
imagesc(ConceptRDM_reordered_pct);
colorbar;
axis square;
title('Concept RDM (Concept Clustered + Percentile)', 'FontSize', 12, 'FontWeight', 'bold');
% Add tick labels for better visualization
xticks(1:10:100);
xticklabels(tickLabel_reorder(1:10:100));
xtickangle(45);
yticks(1:10:100);
yticklabels(tickLabel_reorder(1:10:100));
caxis([0 1]); % Percentile range is 0-1

subplot(1, 2, 2);
imagesc(WordvecRDM_reordered_pct);
colorbar;
axis square;
title('Word Vector RDM (Concept Clustered + Percentile)', 'FontSize', 12, 'FontWeight', 'bold');
% Add tick labels for better visualization
xticks(1:10:100);
xticklabels(tickLabel_reorder(1:10:100));
xtickangle(45);
yticks(1:10:100);
yticklabels(tickLabel_reorder(1:10:100));
caxis([0 1]); % Percentile range is 0-1

colormap(vik);

% ========== Word2Vec-based Clustering Visualization with Percentile ==========
figure('Position', [150, 150, 1200, 500]);

subplot(1, 2, 1);
imagesc(ConceptRDM_reordered_wv_pct);
colorbar;
axis square;
title('Concept RDM (Word2Vec Clustered + Percentile)', 'FontSize', 12, 'FontWeight', 'bold');
% Add tick labels for better visualization
xticks(1:10:100);
xticklabels(tickLabel_reorder_wv(1:10:100));
xtickangle(45);
yticks(1:10:100);
yticklabels(tickLabel_reorder_wv(1:10:100));
caxis([0 1]); % Percentile range is 0-1

subplot(1, 2, 2);
imagesc(WordvecRDM_reordered_wv_pct);
colorbar;
axis square;
title('Word Vector RDM (Word2Vec Clustered + Percentile)', 'FontSize', 12, 'FontWeight', 'bold');
% Add tick labels for better visualization
xticks(1:10:100);
xticklabels(tickLabel_reorder_wv(1:10:100));
xtickangle(45);
yticks(1:10:100);
yticklabels(tickLabel_reorder_wv(1:10:100));
caxis([0 1]); % Percentile range is 0-1

colormap(vik);

% % ========== Additional Visual Enhancement: Side-by-Side Comparison ==========
% figure('Position', [200, 200, 1400, 600]);

% % Original vs Percentile comparison for Concept RDM (concept clustered)
% subplot(2, 2, 1);
% imagesc(ConceptRDM_reordered);
% colorbar;
% axis square;
% title('Original Concept RDM (Concept Clustered)', 'FontSize', 10, 'FontWeight', 'bold');
% xticks(1:20:100);
% yticks(1:20:100);
% colormap(gca, vik);

% subplot(2, 2, 2);
% imagesc(ConceptRDM_reordered_pct);
% colorbar;
% axis square;
% title('Percentile Concept RDM (Concept Clustered)', 'FontSize', 10, 'FontWeight', 'bold');
% xticks(1:20:100);
% yticks(1:20:100);
% caxis([0 1]);
% colormap(gca, vik);

% % Original vs Percentile comparison for Word Vector RDM (concept clustered)
% subplot(2, 2, 3);
% imagesc(WordvecRDM_reordered);
% colorbar;
% axis square;
% title('Original WordVec RDM (Concept Clustered)', 'FontSize', 10, 'FontWeight', 'bold');
% xticks(1:20:100);
% yticks(1:20:100);
% colormap(gca, vik);

% subplot(2, 2, 4);
% imagesc(WordvecRDM_reordered_pct);
% colorbar;
% axis square;
% title('Percentile WordVec RDM (Concept Clustered)', 'FontSize', 10, 'FontWeight', 'bold');
% xticks(1:20:100);
% yticks(1:20:100);
% caxis([0 1]);
% colormap(gca, vik);

% % ========== Alternative Colormap Visualization ==========
% figure('Position', [250, 250, 1200, 500]);

% subplot(1, 2, 1);
% imagesc(ConceptRDM_reordered_pct);
% colorbar;
% axis square;
% title('Concept RDM (Percentile + Hot Colormap)', 'FontSize', 12, 'FontWeight', 'bold');
% xticks(1:10:100);
% xticklabels(tickLabel_reorder(1:10:100));
% xtickangle(45);
% yticks(1:10:100);
% yticklabels(tickLabel_reorder(1:10:100));
% caxis([0 1]);
% colormap(gca, hot);

% subplot(1, 2, 2);
% imagesc(WordvecRDM_reordered_pct);
% colorbar;
% axis square;
% title('Word Vector RDM (Percentile + Hot Colormap)', 'FontSize', 12, 'FontWeight', 'bold');
% xticks(1:10:100);
% xticklabels(tickLabel_reorder(1:10:100));
% xtickangle(45);
% yticks(1:10:100);
% yticklabels(tickLabel_reorder(1:10:100));
% caxis([0 1]);
% colormap(gca, hot);

% ========== Correlation Analysis ==========
% Calculate correlation using concept-based reordered matrices (original)
[corr_Wordvec_Concept, p_Wordvec_Concept] = corr(Matrix2List(ConceptRDM_reordered(st:ed, st:ed)), Matrix2List(WordvecRDM_reordered(st:ed, st:ed)),'type','Spearman','rows','all','tail','both');

% Calculate correlation using word2vec-based reordered matrices (original)
[corr_Wordvec_Concept_wv, p_Wordvec_Concept_wv] = corr(Matrix2List(ConceptRDM_reordered_wv(st:ed, st:ed)), Matrix2List(WordvecRDM_reordered_wv(st:ed, st:ed)),'type','Spearman','rows','all','tail','both');

% Calculate correlation using concept-based reordered matrices (percentile)
[corr_Wordvec_Concept_pct, p_Wordvec_Concept_pct] = corr(Matrix2List(ConceptRDM_reordered_pct(st:ed, st:ed)), Matrix2List(WordvecRDM_reordered_pct(st:ed, st:ed)),'type','Spearman','rows','all','tail','both');

% Calculate correlation using word2vec-based reordered matrices (percentile)
[corr_Wordvec_Concept_wv_pct, p_Wordvec_Concept_wv_pct] = corr(Matrix2List(ConceptRDM_reordered_wv_pct(st:ed, st:ed)), Matrix2List(WordvecRDM_reordered_wv_pct(st:ed, st:ed)),'type','Spearman','rows','all','tail','both');

% Display correlation results
fprintf('=== Correlation Results ===\n');
fprintf('Original Matrices:\n');
fprintf('  Concept-based clustering: Spearman correlation = %.4f (p = %.4f)\n', corr_Wordvec_Concept, p_Wordvec_Concept);
fprintf('  Word2Vec-based clustering: Spearman correlation = %.4f (p = %.4f)\n', corr_Wordvec_Concept_wv, p_Wordvec_Concept_wv);
fprintf('\nPercentile-transformed Matrices:\n');
fprintf('  Concept-based clustering: Spearman correlation = %.4f (p = %.4f)\n', corr_Wordvec_Concept_pct, p_Wordvec_Concept_pct);
fprintf('  Word2Vec-based clustering: Spearman correlation = %.4f (p = %.4f)\n', corr_Wordvec_Concept_wv_pct, p_Wordvec_Concept_wv_pct);
