function [corr_results, p_values] = BinderRSA(varargin)
% RSA correlation analysis between concept and semantic RDMs
%
% Usage:
%   [corr_results, p_values] = rsa_correlation_analysis()
%   [corr_results, p_values] = rsa_correlation_analysis('RandomConcept', true, 'Alpha', 1e-3)

%% Setup
p = inputParser;
addParameter(p, 'RandomConcept', false);
addParameter(p, 'Alpha', 5e-2);
addParameter(p, 'ModelRange', 1:30);
parse(p, varargin{:});

% Add paths
addpath(genpath('E:\BaiduSyncdisk\CASIA_Sync\CAS\Research\symbolic\for_YULAB\DPABI_V8.1_240101'));
addpath(genpath('E:\BaiduSyncdisk\CASIA_Sync\CAS\Research\symbolic\for_YULAB\REST_V1.8_130615'));
addpath(genpath('E:\BaiduSyncdisk\CASIA_Sync\CAS\Research\symbolic\for_YULAB\spm12'));
addpath(genpath('lib'));

%% Load data
% Index and semantic data
load("lib\vik.mat");
index = setdiff(1:95, [37,41]);
data = readtable("lib\wt_65dim_rearrange.csv");
semantic_matrix = double(data{:, 2:end});

if p.Results.RandomConcept
    semantic_matrix = rand(size(semantic_matrix));
end

semantic_rdm = ones(size(semantic_matrix,1), size(semantic_matrix,1)) - ...
    corr(semantic_matrix', 'type', 'Pearson');

% Load concept RDMs
concept_rdms = struct();
for i = p.Results.ModelRange
    if i == 1
        load("lib\Yumodel_RDM_eval_TransE_21to30.mat");
    elseif i == 11
        load("lib\Yumodel_RDM_eval_TransE_31to40.mat");
    elseif i == 21
        load("lib\Yumodel_RDM_eval_TransE_41to50.mat");
    end
    
    model_name = sprintf('model_%02d', i);
    concept_rdms.(model_name) = YuRDM.(sprintf('model_%02d', i+20))(index, index);
end
clear YuRDM;

%% Compute correlations
models = fieldnames(concept_rdms);
corr_results = zeros(length(models), 1);
p_values = zeros(length(models), 1);

semantic_list = Matrix2List(semantic_rdm);

for i = 1:length(models)
    concept_list = Matrix2List(concept_rdms.(models{i}));
    [corr_results(i), p_values(i)] = corr(concept_list, semantic_list, ...
        'type', 'Spearman', 'rows', 'all', 'tail', 'both');
end

%% Results
fprintf('RSA Correlation Results:\n');
for i = 1:length(models)
    fprintf('%s: r=%.3f, p=%.3f n', models{i}, corr_results(i), p_values(i));
end

% Save results
if ~exist('outputs_wt95', 'dir'), mkdir('outputs_wt95'); end
results = table(models, corr_results, p_values, 'VariableNames', {'Model','Correlation','PValue'});
writetable(results, 'outputs_wt95/correlation_results.csv');
end