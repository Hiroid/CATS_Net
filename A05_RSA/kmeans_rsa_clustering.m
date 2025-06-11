%% Kmeans clustering 

clear all;
clc;

random_concept = false;

addpath(genpath('lib'));
load("lib\vik.mat");


for i=1:30
    if i == 1
        load("lib\Yumodel_RDM_eval_TransE_21to30.mat");
    elseif i == 11
        load("lib\Yumodel_RDM_eval_TransE_31to40.mat");
    elseif i == 21
        load("lib\Yumodel_RDM_eval_TransE_41to50.mat");
    end
    WT95_concept_RDM.(sprintf('model_%02d', i)) = YuRDM.(sprintf('model_%02d', i+20));
end
clear YuRDM;
featureRDM = load('./featureRDM.mat');
feature_RDM = featureRDM.feature_RDM;


filename = "lib\wt_65dim_rearrange.csv";
data = readtable(filename);
opts = detectImportOptions(filename);
header = opts.VariableNames;
labels = header(2:end);
base_folder = 'outputs_wt95/';

semantic_matrix = single(data{:, 2:end});
n_filtered_sapmes = size(semantic_matrix, 1);

% Calc Spearman
pcorr_RDM_allmodel = zeros(30,30);


for i = 1:30
    var_name = sprintf('model_%02d', i);
    model_i_concept_RDM = WT95_concept_RDM.(var_name);
    if random_concept
        model_i_concept = rand(size(semantic_matrix));
        model_i_concept_RDM = ones(n_filtered_sapmes, n_filtered_sapmes) - corr(model_i_concept','type','Pearson','rows','all','tail','both');
    end
    
    for j = i+1:30
        var_name = sprintf('model_%02d', j);
        model_j_concept_RDM = WT95_concept_RDM.(var_name);
        if random_concept
            model_j_concept = rand(size(semantic_matrix));
            model_j_concept_RDM = ones(n_filtered_sapmes, n_filtered_sapmes) - corr(model_j_concept','type','Pearson','rows','all','tail','both');
        end
        
        pcorr_tmp = partialcorr(Matrix2List(model_i_concept_RDM),Matrix2List(model_j_concept_RDM),Matrix2List(feature_RDM),'type','Pearson','rows','all','tail','both');

        pcorr_RDM_allmodel(i,j) = pcorr_tmp;
        pcorr_RDM_allmodel(j,i) = pcorr_tmp;


    end
end

%%
dist_matrix = 1-pcorr_RDM_allmodel;
save('InterModelRDM.mat','dist_matrix');
k=2;
[idx, C] = kmeans(dist_matrix, k);

% Reorder dist_matrix by cluster assignments
order_idx = [find(idx == 2); find(idx == 1)];
reordered_dist_matrix = pcorr_RDM_allmodel(order_idx, order_idx);

fprintf('Saving results...\n');
% Save cluster assignments with text labels (alternative method)
cluster_labels = strings(length(idx), 1);
cluster_labels(idx == 1) = "low-consensus";
cluster_labels(idx == 2) = "high-consensus";

results_table = table([1:30]', cluster_labels, ...
    'VariableNames', {'ModelID', 'Kmeans_partial_group'});
writetable(results_table, './kmeans_clusters.csv');