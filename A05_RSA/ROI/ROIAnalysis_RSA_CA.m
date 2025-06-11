%%% 提取指定路径下Control ROI下的RSA分析结果，当前为提取CA层的结果，计算了是否控制featureRDM的两部分结果，结果已经过**FisherZ**转换

clear;clc;

addpath(genpath('/data0/user/lxguo/Downloads/DPABI_V8.1_240101'));
addpath(genpath('/data0/user/lxguo/Downloads/REST_V1.8_130615'));
addpath('/data0/user/lxguo/Downloads/spm');

load("./featureRDM.mat");
addpath('./lib');
%%

% T Images Path: 放你们自己A0的Path
TImagePath = '/data0/user/lxguo/Data/BNU/A0_WT95_picture_spmT_DIDed';
Subject=dir([TImagePath filesep 'SUB*']);
ROIPath = './ROI/';
ROIFiles = {'Reslice3mm_Binary_Jackson2021_semantic_control_ALE_result.nii','Reslice3mm_Binary_Federenko2013_MultiDemand_ROI.nii'};


nSub = length(Subject);
nCond = 95;
%% load spmTFiles
TotalRows = nSub.*length(ROIFiles).*3.*30;
results = cell(TotalRows,6);
row_index = 1;
for subj = 1:nSub
    disp(['starting ' Subject(subj).name]);
    spmTFiles = cell(nCond,1);
    spmTFolder = dir([Subject(subj).folder filesep Subject(subj).name filesep 'spmT*.nii']);
    for i = 1:nCond
        spmTFiles{i} = rest_readfile([spmTFolder(i).folder filesep spmTFolder(i).name]);
    end
    
    for r=1:length(ROIFiles)
        ROI_mask = rest_readfile([ROIPath filesep ROIFiles{r}]);
        raw_data = zeros(length(ROI_mask(ROI_mask>0)),nCond);
        
        for i = 1:nCond
            raw_data(:,i) = spmTFiles{i}(ROI_mask>0);
        end
        % if change to extract, remember to tranpose ROI_data
        BrainRDM = ones(nCond,nCond)-corr(raw_data,'type','Pearson','rows','all','tail','both');
        BrainList = Matrix2List(BrainRDM);
        
        % Second-level Correlation: Correlate the Brain RDM with the Model RDM of Each Voxel
        % defining matrix
        % cdp1
        corrT_model = zeros(1,30);
        pcorrT_model = zeros(1,30);
        for m=1:30
            temp_model = Matrix2List(WT95_cdp_RDM.(sprintf('model_%02d_cdp_1', m)));
            feature_list = Matrix2List(feature_RDM);
            corrT_model(m) = fishZ_transformed(corr(BrainList,temp_model,'type','Spearman','rows','all','tail','both'));
            pcorrT_model(m) = fishZ_transformed(partialcorr(BrainList,temp_model,feature_list,'type','Spearman','rows','all','tail','both'));
        end
        % 将 cdp1 结果存储到 results 中
        for m = 1:30
            results{row_index, 1} = Subject(subj).name; % 存储被试名
            results{row_index, 2} = 'CA1'; % 层数
            results{row_index, 3} = m; % 模型编号
            results{row_index, 4} = ROIFiles{r}(1:end-4); % ROI名字
            results{row_index, 5} = corrT_model(m); % 相关系数
            results{row_index, 6} = pcorrT_model(m); % 偏相关系数
            row_index = row_index + 1;
        end

        % cdp2
        corrT_model = zeros(1,30);
        pcorrT_model = zeros(1,30);
        for m=1:30
            temp_model = Matrix2List(WT95_cdp_RDM.(sprintf('model_%02d_cdp_2', m)));
            feature_list = Matrix2List(feature_RDM);
            corrT_model(m) = fishZ_transformed(corr(BrainList,temp_model,'type','Spearman','rows','all','tail','both'));
            pcorrT_model(m) = fishZ_transformed(partialcorr(BrainList,temp_model,feature_list,'type','Spearman','rows','all','tail','both'));
        end
        % 将 cdp2 结果存储到 results 中
        for m = 1:30
            results{row_index, 1} = Subject(subj).name; % 存储被试名
            results{row_index, 2} = 'CA2';  % 层数
            results{row_index, 3} = m; % 模型编号
            results{row_index, 4} = ROIFiles{r}(1:end-4); % ROI名字
            results{row_index, 5} = corrT_model(m); % 相关系数
            results{row_index, 6} = pcorrT_model(m); % 偏相关系数
            row_index = row_index + 1;
        end
        % cdp3
        corrT_model = zeros(1,30);
        pcorrT_model = zeros(1,30);
        for m=1:30
            temp_model = Matrix2List(WT95_cdp_RDM.(sprintf('model_%02d_cdp_3', m)));
            feature_list = Matrix2List(feature_RDM);
            corrT_model(m) = fishZ_transformed(corr(BrainList,temp_model,'type','Spearman','rows','all','tail','both'));
            pcorrT_model(m) = fishZ_transformed(partialcorr(BrainList,temp_model,feature_list,'type','Spearman','rows','all','tail','both'));
        end
        % 将 cdp3 结果存储到 results 中
        for m = 1:30
            results{row_index, 1} = Subject(subj).name; % 存储被试名
            results{row_index, 2} = 'CA3';  % 层数
            results{row_index, 3} = m; % 模型编号
            results{row_index, 4} = ROIFiles{r}(1:end-4); % ROI名字
            results{row_index, 5} = corrT_model(m); % 相关系数
            results{row_index, 6} = pcorrT_model(m); % 偏相关系数
            row_index = row_index + 1;
        end
    end
    disp([Subject(subj).name 'have been finished']);
end

%% OutPut and Save
% results = results(1:4680,:);
results = results(1:row_index-1,:);
T = cell2table(results, 'VariableNames', {'Subject', 'Layer', 'ModelNumber', 'ROIName', 'Correlation', 'PartialCorrelation'});

% 输出为 CSV 文件
writetable(T,'RSA_ROI_CA_results.csv');


%%
function result = fishZ_transformed(rho_data)
    % compute_log_expression 计算0.5 .* log((1 + rho_data) ./ (1 - rho_data))
    %
    % 输入:
    %   rho_data_cdp1 - 一个标量、向量或矩阵，值域需满足 -1 < rho_data < 1
    %
    % 输出:
    %   result
    
    % 检查输入范围
    if any(rho_data <= -1 | rho_data >= 1, 'all')
        error('输入rho_data的值必须满足 -1 < rho_data_cdp1 < 1');
    end
    
    % 计算结果
    result = 0.5 .* log((1 + rho_data) ./ (1 - rho_data));
end
