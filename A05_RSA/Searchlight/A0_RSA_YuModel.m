function A0_RSA_YuModel(subject_id, model_lib_path, varargin)
% RSA_ANALYSIS_OPTIMIZED - Optimized multi-model RSA analysis
%
% Syntax:
%   rsa_analysis_optimized(subject_id, model_lib_path)
%   rsa_analysis_optimized(subject_id, model_lib_path, 'ParameterName', ParameterValue)
%
% Input Arguments:
%   subject_id      - Subject ID/index
%   model_lib_path  - Path to directory containing model RDM files
%
% Optional Parameters:
%   'SearchlightRadius' - Searchlight radius (default: 10)
%   'TImagePath'       - T-statistic images path (default: 'E:\YuLab\A0_WT95_picture_spmT\')
%   'OutputPath'       - Output path (default: 'E:\YuLab\A1_results\')
%   'LibPath'          - Library path (default: 'E:\YuLab\lib')
%   'ModelRange'       - Models to process (default: 1:30)

%% Parse inputs
p = inputParser;
addRequired(p, 'subject_id');
addRequired(p, 'model_lib_path');
addParameter(p, 'SearchlightRadius', 10);
addParameter(p, 'TImagePath', 'E:\YuLab\fMRI_results/A0_WT95_picture_spmT\');
addParameter(p, 'OutputPath', 'E:\YuLab\fMRI_results/A1_results\');
addParameter(p, 'LibPath', 'E:\YuLab\fMRI_results/lib');
addParameter(p, 'ModelRange', 1:30);
parse(p, subject_id, model_lib_path, varargin{:});
config = p.Results;

%% Initialize
fprintf('Starting RSA analysis for subject %d...\n', subject_id);
addpath(config.LibPath);

% Load models
fprintf('Loading models...\n');
model_rdms = load_multiple_models(config);

% Setup paths
subjects = dir([config.TImagePath 'SUB*']);
subject_name = subjects(subject_id).name;
subject_input = fullfile(config.TImagePath, subject_name);
subject_output = fullfile(config.OutputPath, subject_name);
if ~exist(subject_output, 'dir'), mkdir(subject_output); end

% Load data
fprintf('Loading subject data: %s\n', subject_name);
[mask, ~, header] = rest_readfile('E:\matlabtool\DPABI_V8.2_240510\Templates\GreyMask_02_61x73x61.img');
t_files = dir(fullfile(subject_input, 'spmT*.nii'));
brain_images = zeros([header.dim, length(t_files)]);
for i = 1:length(t_files)
    [img, ~, ~] = rest_readfile(fullfile(subject_input, t_files(i).name));
    brain_images(:,:,:,i) = img;
end

%% RSA Analysis
fprintf('Running RSA analysis for %d models...\n', length(model_rdms.names));
results = run_searchlight_rsa(mask, brain_images, model_rdms, header, config);

%% Save results
fprintf('Saving results...\n');
for i = 1:length(model_rdms.names)
    model_name = model_rdms.names{i};
    output_file = fullfile(subject_output, [model_name '.nii']);
    rest_writefile(results.(model_name), output_file, header.dim, [1 1 1], header, 'double');
end

fprintf('Analysis completed!\n');
end

%% ========== Helper Functions ==========

function model_rdms = load_multiple_models(config)
% Load multiple model RDMs from files

% Define model files based on your pattern
model_files = {
    'Yumodel_RDM_eval_TransE_21to30.mat', [1, 10];
    'Yumodel_RDM_eval_TransE_31to40.mat', [11, 20];
    'Yumodel_RDM_eval_TransE_41to50.mat', [21, 30]
};

% Load index if available
index_file = fullfile(config.LibPath, 'index.mat');
if exist(index_file, 'file')
    idx_data = load(index_file);
    index_map = idx_data.index;
else
    index_map = [];
end

model_rdms = struct();
model_rdms.data = struct();
model_rdms.names = {};

% Load models from each file
for file_idx = 1:size(model_files, 1)
    file_name = model_files{file_idx, 1};
    range = model_files{file_idx, 2};
    
    file_path = fullfile(config.model_lib_path, file_name);
    if ~exist(file_path, 'file'), continue; end
    
    data = load(file_path);
    
    for i = range(1):range(2)
        if ~any(i == config.ModelRange), continue; end
        
        % Load main model (i+20 based on your pattern)
        model_field = sprintf('model_%02d', i+20);
        if isfield(data.YuRDM, model_field)
            rdm = data.YuRDM.(model_field);
            if ~isempty(index_map)
                rdm = rdm(index_map, index_map);
            end
            
            model_name = sprintf('model_%02d', i);
            model_rdms.data.(model_name) = Matrix2List(rdm);
            model_rdms.names{end+1} = model_name;
            
            % Load variants (e.g., model_XX_cdp_1)
            variant_fields = fieldnames(data.YuRDM);
            pattern = sprintf('^model_%02d_', i+20);
            for v = 1:length(variant_fields)
                if ~isempty(regexp(variant_fields{v}, pattern, 'once'))
                    variant_rdm = data.YuRDM.(variant_fields{v});
                    if ~isempty(index_map)
                        variant_rdm = variant_rdm(index_map, index_map);
                    end
                    
                    suffix = variant_fields{v}((length(sprintf('model_%02d_', i+20))+1):end);
                    variant_name = sprintf('model_%02d_%s', i, suffix);
                    model_rdms.data.(variant_name) = Matrix2List(variant_rdm);
                    model_rdms.names{end+1} = variant_name;
                end
            end
        end
    end
end

fprintf('Loaded %d models\n', length(model_rdms.names));
end

function results = run_searchlight_rsa(mask, brain_images, model_rdms, header, config)
% Run searchlight RSA analysis

% Initialize results
results = struct();
for i = 1:length(model_rdms.names)
    results.(model_rdms.names{i}) = zeros(header.dim);
end

% Setup searchlight
v_ref = spm_vol('E:\matlabtool\DPABI_V8.2_240510\Templates\GreyMask_02_61x73x61.img');
[~, coords] = spm_read_vols(v_ref);

% Find valid voxels
valid_idx = find(mask > 1/3);
[x_coords, y_coords, z_coords] = ind2sub(size(mask), valid_idx);

fprintf('Processing %d voxels...\n', length(valid_idx));

% Process each voxel
for v = 1:length(valid_idx)
    x = x_coords(v); y = y_coords(v); z = z_coords(v);
    
    % Get ROI indices
    roi_idx = gen_ROI_fast(v_ref, coords, [x,y,z], config.SearchlightRadius);
    if isempty(roi_idx), continue; end
    
    % Extract ROI data
    roi_data = zeros(length(roi_idx), size(brain_images, 4));
    for c = 1:size(brain_images, 4)
        img = brain_images(:,:,:,c);
        roi_data(:,c) = img(roi_idx);
    end
    
    % Compute brain RDM
    corr_mat = corr(roi_data, 'type', 'Pearson');
    brain_rdm = ones(size(corr_mat)) - corr_mat;
    brain_list = Matrix2List(brain_rdm);
    
    % Correlate with model RDMs
    for i = 1:length(model_rdms.names)
        model_name = model_rdms.names{i};
        model_list = model_rdms.data.(model_name);
        corr_val = corr(brain_list, model_list, 'type', 'Spearman');
        results.(model_name)(x,y,z) = corr_val;
    end
    
    % Progress
    if mod(v, 1000) == 0
        fprintf('  Progress: %d/%d (%.1f%%)\n', v, length(valid_idx), 100*v/length(valid_idx));
    end
end
end

