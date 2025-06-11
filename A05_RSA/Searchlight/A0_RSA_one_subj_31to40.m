function A0_RSA_one_subj_31to40(subj)

    % % Add software, add DPABI before spm
    if ismac
        addpath(genpath(fullfile('..', 'DPABI_V8.1_240101')));
        addpath(genpath(fullfile('..', 'REST_V1.8_130615')));
        addpath(fullfile('..', 'spm'));
        greymask_path = fullfile('..', 'DPABI_V8.1_240101', 'Templates', 'GreyMask_02_61x73x61.img');
    elseif ispc
        addpath(genpath(fullfile('..', 'DPABI_V8.1_240101')));
        addpath(genpath(fullfile('..', 'REST_V1.8_130615')));
        addpath(fullfile('..', 'spm'));
        greymask_path = fullfile('..', 'DPABI_V8.1_240101', 'Templates', 'GreyMask_02_61x73x61.img');
    elseif isunix
        addpath(genpath('/data0/user/lxguo/Downloads/DPABI_V8.1_240101'));
        addpath(genpath('/data0/user/lxguo/Downloads/REST_V1.8_130615'));
        addpath('/data0/user/lxguo/Downloads/spm');
        greymask_path = '/data0/user/lxguo/Downloads/DPABI_V8.1_240101/Templates/GreyMask_02_61x73x61.img';
    end
    addpath(genpath('lib'));

    Radium = 10;
    % % Enter the File Path of the Model RDMs here:
    load(['lib' filesep 'Yumodel_RDM_eval_TransE_31to40.mat']);
    load(['lib' filesep 'feature_fe_RDM_eval_TransE_31to40.mat']);

    % % Enter the File Path of the T Images here:
    TImagePath = '/data0/user/lxguo/Data/BNU/A0_WT95_picture_spmT_DIDed';
    % % Enter the Output File Path here:
    OutputPath = ['A1_Searchlight' filesep 'A0_all_results_eval_TransE_31to40'];
    Subject=dir([TImagePath filesep 'SUB*']);

    % Load Structure Image of Each Subject
    [Mask,VoxDim,Header] = rest_readfile(greymask_path);
    PrintSum = sum(sum(sum(int8(Mask > 1/3)))); % logic

    % load the ref MNI space and transform
    Vref = spm_vol(greymask_path);
    [~, mmCoords] = spm_read_vols(Vref); % return all the voxel's xyz

    % Load Each Model RDM
    % ---------------------------------------------------------------
    RN50_conv1 = Matrix2List(feature_fe_RDM.model_31_fe_conv1);
    RN50_layer1 = Matrix2List(feature_fe_RDM.model_31_fe_layer1);
    RN50_layer2 = Matrix2List(feature_fe_RDM.model_31_fe_layer2);
    RN50_layer3 = Matrix2List(feature_fe_RDM.model_31_fe_layer3);
    RN50_layer4 = Matrix2List(feature_fe_RDM.model_31_fe_layer4);
    RN50_last = Matrix2List(feature_fe_RDM.model_31_fe_last);
    
    Yumodel_31_ts1 = Matrix2List(YuRDM.model_31_ts_1);
    Yumodel_31_ts2 = Matrix2List(YuRDM.model_31_ts_2);
    Yumodel_31_ts3 = Matrix2List(YuRDM.model_31_ts_3);
    Yumodel_31_cdp1 = Matrix2List(YuRDM.model_31_cdp_1);
    Yumodel_31_cdp2 = Matrix2List(YuRDM.model_31_cdp_2);
    Yumodel_31_cdp3 = Matrix2List(YuRDM.model_31_cdp_3);
    Yumodel_31_conv1 = Matrix2List(feature_fe_RDM.model_31_fe_conv1);
    Yumodel_31_layer1 = Matrix2List(feature_fe_RDM.model_31_fe_layer1);
    Yumodel_31_layer2 = Matrix2List(feature_fe_RDM.model_31_fe_layer2);
    Yumodel_31_layer3 = Matrix2List(feature_fe_RDM.model_31_fe_layer3);
    Yumodel_31_layer4 = Matrix2List(feature_fe_RDM.model_31_fe_layer4);
    Yumodel_31_last = Matrix2List(feature_fe_RDM.model_31_fe_last);
    Yumodel_31 = Matrix2List(YuRDM.model_31);

    Yumodel_32_ts1 = Matrix2List(YuRDM.model_32_ts_1);
    Yumodel_32_ts2 = Matrix2List(YuRDM.model_32_ts_2);
    Yumodel_32_ts3 = Matrix2List(YuRDM.model_32_ts_3);
    Yumodel_32_cdp1 = Matrix2List(YuRDM.model_32_cdp_1);
    Yumodel_32_cdp2 = Matrix2List(YuRDM.model_32_cdp_2);
    Yumodel_32_cdp3 = Matrix2List(YuRDM.model_32_cdp_3);
    Yumodel_32_conv1 = Matrix2List(feature_fe_RDM.model_32_fe_conv1);
    Yumodel_32_layer1 = Matrix2List(feature_fe_RDM.model_32_fe_layer1);
    Yumodel_32_layer2 = Matrix2List(feature_fe_RDM.model_32_fe_layer2);
    Yumodel_32_layer3 = Matrix2List(feature_fe_RDM.model_32_fe_layer3);
    Yumodel_32_layer4 = Matrix2List(feature_fe_RDM.model_32_fe_layer4);
    Yumodel_32_last = Matrix2List(feature_fe_RDM.model_32_fe_last);
    Yumodel_32 = Matrix2List(YuRDM.model_32);

    Yumodel_33_ts1 = Matrix2List(YuRDM.model_33_ts_1);
    Yumodel_33_ts2 = Matrix2List(YuRDM.model_33_ts_2);
    Yumodel_33_ts3 = Matrix2List(YuRDM.model_33_ts_3);
    Yumodel_33_cdp1 = Matrix2List(YuRDM.model_33_cdp_1);
    Yumodel_33_cdp2 = Matrix2List(YuRDM.model_33_cdp_2);
    Yumodel_33_cdp3 = Matrix2List(YuRDM.model_33_cdp_3);
    Yumodel_33_conv1 = Matrix2List(feature_fe_RDM.model_33_fe_conv1);
    Yumodel_33_layer1 = Matrix2List(feature_fe_RDM.model_33_fe_layer1);
    Yumodel_33_layer2 = Matrix2List(feature_fe_RDM.model_33_fe_layer2);
    Yumodel_33_layer3 = Matrix2List(feature_fe_RDM.model_33_fe_layer3);
    Yumodel_33_layer4 = Matrix2List(feature_fe_RDM.model_33_fe_layer4);
    Yumodel_33_last = Matrix2List(feature_fe_RDM.model_33_fe_last);
    Yumodel_33 = Matrix2List(YuRDM.model_33);

    Yumodel_34_ts1 = Matrix2List(YuRDM.model_34_ts_1);
    Yumodel_34_ts2 = Matrix2List(YuRDM.model_34_ts_2);
    Yumodel_34_ts3 = Matrix2List(YuRDM.model_34_ts_3);
    Yumodel_34_cdp1 = Matrix2List(YuRDM.model_34_cdp_1);
    Yumodel_34_cdp2 = Matrix2List(YuRDM.model_34_cdp_2);
    Yumodel_34_cdp3 = Matrix2List(YuRDM.model_34_cdp_3);
    Yumodel_34_conv1 = Matrix2List(feature_fe_RDM.model_34_fe_conv1);
    Yumodel_34_layer1 = Matrix2List(feature_fe_RDM.model_34_fe_layer1);
    Yumodel_34_layer2 = Matrix2List(feature_fe_RDM.model_34_fe_layer2);
    Yumodel_34_layer3 = Matrix2List(feature_fe_RDM.model_34_fe_layer3);
    Yumodel_34_layer4 = Matrix2List(feature_fe_RDM.model_34_fe_layer4);
    Yumodel_34_last = Matrix2List(feature_fe_RDM.model_34_fe_last);
    Yumodel_34 = Matrix2List(YuRDM.model_34);

    Yumodel_35_ts1 = Matrix2List(YuRDM.model_35_ts_1);
    Yumodel_35_ts2 = Matrix2List(YuRDM.model_35_ts_2);
    Yumodel_35_ts3 = Matrix2List(YuRDM.model_35_ts_3);
    Yumodel_35_cdp1 = Matrix2List(YuRDM.model_35_cdp_1);
    Yumodel_35_cdp2 = Matrix2List(YuRDM.model_35_cdp_2);
    Yumodel_35_cdp3 = Matrix2List(YuRDM.model_35_cdp_3);
    Yumodel_35_conv1 = Matrix2List(feature_fe_RDM.model_35_fe_conv1);
    Yumodel_35_layer1 = Matrix2List(feature_fe_RDM.model_35_fe_layer1);
    Yumodel_35_layer2 = Matrix2List(feature_fe_RDM.model_35_fe_layer2);
    Yumodel_35_layer3 = Matrix2List(feature_fe_RDM.model_35_fe_layer3);
    Yumodel_35_layer4 = Matrix2List(feature_fe_RDM.model_35_fe_layer4);
    Yumodel_35_last = Matrix2List(feature_fe_RDM.model_35_fe_last);
    Yumodel_35 = Matrix2List(YuRDM.model_35);

    Yumodel_36_ts1 = Matrix2List(YuRDM.model_36_ts_1);
    Yumodel_36_ts2 = Matrix2List(YuRDM.model_36_ts_2);
    Yumodel_36_ts3 = Matrix2List(YuRDM.model_36_ts_3);
    Yumodel_36_cdp1 = Matrix2List(YuRDM.model_36_cdp_1);
    Yumodel_36_cdp2 = Matrix2List(YuRDM.model_36_cdp_2);
    Yumodel_36_cdp3 = Matrix2List(YuRDM.model_36_cdp_3);
    Yumodel_36_conv1 = Matrix2List(feature_fe_RDM.model_36_fe_conv1);
    Yumodel_36_layer1 = Matrix2List(feature_fe_RDM.model_36_fe_layer1);
    Yumodel_36_layer2 = Matrix2List(feature_fe_RDM.model_36_fe_layer2);
    Yumodel_36_layer3 = Matrix2List(feature_fe_RDM.model_36_fe_layer3);
    Yumodel_36_layer4 = Matrix2List(feature_fe_RDM.model_36_fe_layer4);
    Yumodel_36_last = Matrix2List(feature_fe_RDM.model_36_fe_last);
    Yumodel_36 = Matrix2List(YuRDM.model_36);

    Yumodel_37_ts1 = Matrix2List(YuRDM.model_37_ts_1);
    Yumodel_37_ts2 = Matrix2List(YuRDM.model_37_ts_2);
    Yumodel_37_ts3 = Matrix2List(YuRDM.model_37_ts_3);
    Yumodel_37_cdp1 = Matrix2List(YuRDM.model_37_cdp_1);
    Yumodel_37_cdp2 = Matrix2List(YuRDM.model_37_cdp_2);
    Yumodel_37_cdp3 = Matrix2List(YuRDM.model_37_cdp_3);
    Yumodel_37_conv1 = Matrix2List(feature_fe_RDM.model_37_fe_conv1);
    Yumodel_37_layer1 = Matrix2List(feature_fe_RDM.model_37_fe_layer1);
    Yumodel_37_layer2 = Matrix2List(feature_fe_RDM.model_37_fe_layer2);
    Yumodel_37_layer3 = Matrix2List(feature_fe_RDM.model_37_fe_layer3);
    Yumodel_37_layer4 = Matrix2List(feature_fe_RDM.model_37_fe_layer4);
    Yumodel_37_last = Matrix2List(feature_fe_RDM.model_37_fe_last);
    Yumodel_37 = Matrix2List(YuRDM.model_37);

    Yumodel_38_ts1 = Matrix2List(YuRDM.model_38_ts_1);
    Yumodel_38_ts2 = Matrix2List(YuRDM.model_38_ts_2);
    Yumodel_38_ts3 = Matrix2List(YuRDM.model_38_ts_3);
    Yumodel_38_cdp1 = Matrix2List(YuRDM.model_38_cdp_1);
    Yumodel_38_cdp2 = Matrix2List(YuRDM.model_38_cdp_2);
    Yumodel_38_cdp3 = Matrix2List(YuRDM.model_38_cdp_3);
    Yumodel_38_conv1 = Matrix2List(feature_fe_RDM.model_38_fe_conv1);
    Yumodel_38_layer1 = Matrix2List(feature_fe_RDM.model_38_fe_layer1);
    Yumodel_38_layer2 = Matrix2List(feature_fe_RDM.model_38_fe_layer2);
    Yumodel_38_layer3 = Matrix2List(feature_fe_RDM.model_38_fe_layer3);
    Yumodel_38_layer4 = Matrix2List(feature_fe_RDM.model_38_fe_layer4);
    Yumodel_38_last = Matrix2List(feature_fe_RDM.model_38_fe_last);
    Yumodel_38 = Matrix2List(YuRDM.model_38);

    Yumodel_39_ts1 = Matrix2List(YuRDM.model_39_ts_1);
    Yumodel_39_ts2 = Matrix2List(YuRDM.model_39_ts_2);
    Yumodel_39_ts3 = Matrix2List(YuRDM.model_39_ts_3);
    Yumodel_39_cdp1 = Matrix2List(YuRDM.model_39_cdp_1);
    Yumodel_39_cdp2 = Matrix2List(YuRDM.model_39_cdp_2);
    Yumodel_39_cdp3 = Matrix2List(YuRDM.model_39_cdp_3);
    Yumodel_39_conv1 = Matrix2List(feature_fe_RDM.model_39_fe_conv1);
    Yumodel_39_layer1 = Matrix2List(feature_fe_RDM.model_39_fe_layer1);
    Yumodel_39_layer2 = Matrix2List(feature_fe_RDM.model_39_fe_layer2);
    Yumodel_39_layer3 = Matrix2List(feature_fe_RDM.model_39_fe_layer3);
    Yumodel_39_layer4 = Matrix2List(feature_fe_RDM.model_39_fe_layer4);
    Yumodel_39_last = Matrix2List(feature_fe_RDM.model_39_fe_last);
    Yumodel_39 = Matrix2List(YuRDM.model_39);

    Yumodel_40_ts1 = Matrix2List(YuRDM.model_40_ts_1);
    Yumodel_40_ts2 = Matrix2List(YuRDM.model_40_ts_2);
    Yumodel_40_ts3 = Matrix2List(YuRDM.model_40_ts_3);
    Yumodel_40_cdp1 = Matrix2List(YuRDM.model_40_cdp_1);
    Yumodel_40_cdp2 = Matrix2List(YuRDM.model_40_cdp_2);
    Yumodel_40_cdp3 = Matrix2List(YuRDM.model_40_cdp_3);
    Yumodel_40_conv1 = Matrix2List(feature_fe_RDM.model_40_fe_conv1);
    Yumodel_40_layer1 = Matrix2List(feature_fe_RDM.model_40_fe_layer1);
    Yumodel_40_layer2 = Matrix2List(feature_fe_RDM.model_40_fe_layer2);
    Yumodel_40_layer3 = Matrix2List(feature_fe_RDM.model_40_fe_layer3);
    Yumodel_40_layer4 = Matrix2List(feature_fe_RDM.model_40_fe_layer4);
    Yumodel_40_last = Matrix2List(feature_fe_RDM.model_40_fe_last);
    Yumodel_40 = Matrix2List(YuRDM.model_40);


    mkdir([OutputPath filesep Subject(subj).name]);
    
    % Load T Images of Each Conditon of Each Subject
    TImageFiles = dir([TImagePath filesep Subject(subj).name filesep 'spmT*.nii']); % make folder path
    nCond = length(TImageFiles);
    
    TotalTImage = {};
    for i = 1 : nCond
        [TImage,~,~] = rest_readfile([TImagePath filesep Subject(subj).name filesep TImageFiles(i).name]);
        TotalTImage{i} = TImage;
    end
    
    % defining matrix
    % ---------------------------------------------------------------
    corrT_Yumodel_31 = zeros(Header.dim);
    corrT_Yumodel_31_ts1 = zeros(Header.dim);
    corrT_Yumodel_31_ts2 = zeros(Header.dim);
    corrT_Yumodel_31_ts3 = zeros(Header.dim);
    corrT_Yumodel_31_cdp1 = zeros(Header.dim);
    corrT_Yumodel_31_cdp2 = zeros(Header.dim);
    corrT_Yumodel_31_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_31_gist = zeros(Header.dim);
    pcorrT_Yumodel_31_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_31_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_31_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_31_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_31_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_31_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_32 = zeros(Header.dim);
    corrT_Yumodel_32_ts1 = zeros(Header.dim);
    corrT_Yumodel_32_ts2 = zeros(Header.dim);
    corrT_Yumodel_32_ts3 = zeros(Header.dim);
    corrT_Yumodel_32_cdp1 = zeros(Header.dim);
    corrT_Yumodel_32_cdp2 = zeros(Header.dim);
    corrT_Yumodel_32_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_32_gist = zeros(Header.dim);
    pcorrT_Yumodel_32_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_32_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_32_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_32_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_32_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_32_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_33 = zeros(Header.dim);
    corrT_Yumodel_33_ts1 = zeros(Header.dim);
    corrT_Yumodel_33_ts2 = zeros(Header.dim);
    corrT_Yumodel_33_ts3 = zeros(Header.dim);
    corrT_Yumodel_33_cdp1 = zeros(Header.dim);
    corrT_Yumodel_33_cdp2 = zeros(Header.dim);
    corrT_Yumodel_33_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_33_gist = zeros(Header.dim);
    pcorrT_Yumodel_33_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_33_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_33_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_33_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_33_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_33_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_34 = zeros(Header.dim);
    corrT_Yumodel_34_ts1 = zeros(Header.dim);
    corrT_Yumodel_34_ts2 = zeros(Header.dim);
    corrT_Yumodel_34_ts3 = zeros(Header.dim);
    corrT_Yumodel_34_cdp1 = zeros(Header.dim);
    corrT_Yumodel_34_cdp2 = zeros(Header.dim);
    corrT_Yumodel_34_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_34_gist = zeros(Header.dim);
    pcorrT_Yumodel_34_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_34_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_34_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_34_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_34_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_34_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_35 = zeros(Header.dim);
    corrT_Yumodel_35_ts1 = zeros(Header.dim);
    corrT_Yumodel_35_ts2 = zeros(Header.dim);
    corrT_Yumodel_35_ts3 = zeros(Header.dim);
    corrT_Yumodel_35_cdp1 = zeros(Header.dim);
    corrT_Yumodel_35_cdp2 = zeros(Header.dim);
    corrT_Yumodel_35_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_35_gist = zeros(Header.dim);
    pcorrT_Yumodel_35_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_35_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_35_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_35_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_35_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_35_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_36 = zeros(Header.dim);
    corrT_Yumodel_36_ts1 = zeros(Header.dim);
    corrT_Yumodel_36_ts2 = zeros(Header.dim);
    corrT_Yumodel_36_ts3 = zeros(Header.dim);
    corrT_Yumodel_36_cdp1 = zeros(Header.dim);
    corrT_Yumodel_36_cdp2 = zeros(Header.dim);
    corrT_Yumodel_36_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_36_gist = zeros(Header.dim);
    pcorrT_Yumodel_36_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_36_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_36_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_36_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_36_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_36_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_37 = zeros(Header.dim);
    corrT_Yumodel_37_ts1 = zeros(Header.dim);
    corrT_Yumodel_37_ts2 = zeros(Header.dim);
    corrT_Yumodel_37_ts3 = zeros(Header.dim);
    corrT_Yumodel_37_cdp1 = zeros(Header.dim);
    corrT_Yumodel_37_cdp2 = zeros(Header.dim);
    corrT_Yumodel_37_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_37_gist = zeros(Header.dim);
    pcorrT_Yumodel_37_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_37_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_37_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_37_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_37_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_37_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_38 = zeros(Header.dim);
    corrT_Yumodel_38_ts1 = zeros(Header.dim);
    corrT_Yumodel_38_ts2 = zeros(Header.dim);
    corrT_Yumodel_38_ts3 = zeros(Header.dim);
    corrT_Yumodel_38_cdp1 = zeros(Header.dim);
    corrT_Yumodel_38_cdp2 = zeros(Header.dim);
    corrT_Yumodel_38_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_38_gist = zeros(Header.dim);
    pcorrT_Yumodel_38_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_38_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_38_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_38_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_38_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_38_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_39 = zeros(Header.dim);
    corrT_Yumodel_39_ts1 = zeros(Header.dim);
    corrT_Yumodel_39_ts2 = zeros(Header.dim);
    corrT_Yumodel_39_ts3 = zeros(Header.dim);
    corrT_Yumodel_39_cdp1 = zeros(Header.dim);
    corrT_Yumodel_39_cdp2 = zeros(Header.dim);
    corrT_Yumodel_39_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_39_gist = zeros(Header.dim);
    pcorrT_Yumodel_39_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_39_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_39_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_39_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_39_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_39_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_40 = zeros(Header.dim);
    corrT_Yumodel_40_ts1 = zeros(Header.dim);
    corrT_Yumodel_40_ts2 = zeros(Header.dim);
    corrT_Yumodel_40_ts3 = zeros(Header.dim);
    corrT_Yumodel_40_cdp1 = zeros(Header.dim);
    corrT_Yumodel_40_cdp2 = zeros(Header.dim);
    corrT_Yumodel_40_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_40_gist = zeros(Header.dim);
    pcorrT_Yumodel_40_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_40_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_40_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_40_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_40_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_40_cdp3_gist = zeros(Header.dim);
    
    PrintN = 0;
    
    for x = 1 : Header.dim(1)
        for y = 1 : Header.dim(2)
            for z = 1 : Header.dim(3)
                if Mask(x,y,z) > (1/3) % If mask is not bi, use it to threshold or probability
                    
                    % First-level Correlation: Create the Brain RDM of Each Voxel
                    Index = gen_ROI_fast(Vref,mmCoords,[x,y,z], Radium);
                    nVoxel = length(Index);
                    ROI_t = zeros(nVoxel,nCond);
                    
                    for Label = 1 : nCond
                        TImage_tmp = TotalTImage{1, Label};
                        ROI_t(:,Label) = TImage_tmp(Index);
                    end
                    BrainRDMt = ones(nCond,nCond) - corr(ROI_t,'type','Pearson','rows','all','tail','both');
                    BrainList = Matrix2List(BrainRDMt);                 
                    
                    
                    % Second-level Correlation: correlate between Brain RDM and
                    % Model RDM
                    corrT_Yumodel_31(x,y,z) = corr(BrainList,Yumodel_31,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_31_ts1(x,y,z) = corr(BrainList,Yumodel_31_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_31_ts2(x,y,z) = corr(BrainList,Yumodel_31_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_31_ts3(x,y,z) = corr(BrainList,Yumodel_31_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_31_cdp1(x,y,z) = corr(BrainList,Yumodel_31_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_31_cdp2(x,y,z) = corr(BrainList,Yumodel_31_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_31_cdp3(x,y,z) = corr(BrainList,Yumodel_31_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_31_gist(x,y,z) = partialcorr(BrainList,Yumodel_31,Yumodel_31_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_31_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_31_ts1,Yumodel_31_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_31_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_31_ts2,Yumodel_31_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_31_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_31_ts3,Yumodel_31_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_31_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_31_cdp1,Yumodel_31_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_31_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_31_cdp2,Yumodel_31_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_31_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_31_cdp3,Yumodel_31_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_32(x,y,z) = corr(BrainList,Yumodel_32,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_32_ts1(x,y,z) = corr(BrainList,Yumodel_32_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_32_ts2(x,y,z) = corr(BrainList,Yumodel_32_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_32_ts3(x,y,z) = corr(BrainList,Yumodel_32_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_32_cdp1(x,y,z) = corr(BrainList,Yumodel_32_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_32_cdp2(x,y,z) = corr(BrainList,Yumodel_32_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_32_cdp3(x,y,z) = corr(BrainList,Yumodel_32_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_32_gist(x,y,z) = partialcorr(BrainList,Yumodel_32,Yumodel_32_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_32_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_32_ts1,Yumodel_32_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_32_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_32_ts2,Yumodel_32_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_32_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_32_ts3,Yumodel_32_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_32_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_32_cdp1,Yumodel_32_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_32_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_32_cdp2,Yumodel_32_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_32_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_32_cdp3,Yumodel_32_last,'type','Spearman','rows','all','tail','both');
                    
                    corrT_Yumodel_33(x,y,z) = corr(BrainList,Yumodel_33,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_33_ts1(x,y,z) = corr(BrainList,Yumodel_33_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_33_ts2(x,y,z) = corr(BrainList,Yumodel_33_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_33_ts3(x,y,z) = corr(BrainList,Yumodel_33_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_33_cdp1(x,y,z) = corr(BrainList,Yumodel_33_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_33_cdp2(x,y,z) = corr(BrainList,Yumodel_33_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_33_cdp3(x,y,z) = corr(BrainList,Yumodel_33_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_33_gist(x,y,z) = partialcorr(BrainList,Yumodel_33,Yumodel_33_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_33_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_33_ts1,Yumodel_33_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_33_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_33_ts2,Yumodel_33_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_33_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_33_ts3,Yumodel_33_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_33_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_33_cdp1,Yumodel_33_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_33_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_33_cdp2,Yumodel_33_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_33_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_33_cdp3,Yumodel_33_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_34(x,y,z) = corr(BrainList,Yumodel_34,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_34_ts1(x,y,z) = corr(BrainList,Yumodel_34_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_34_ts2(x,y,z) = corr(BrainList,Yumodel_34_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_34_ts3(x,y,z) = corr(BrainList,Yumodel_34_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_34_cdp1(x,y,z) = corr(BrainList,Yumodel_34_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_34_cdp2(x,y,z) = corr(BrainList,Yumodel_34_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_34_cdp3(x,y,z) = corr(BrainList,Yumodel_34_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_34_gist(x,y,z) = partialcorr(BrainList,Yumodel_34,Yumodel_34_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_34_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_34_ts1,Yumodel_34_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_34_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_34_ts2,Yumodel_34_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_34_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_34_ts3,Yumodel_34_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_34_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_34_cdp1,Yumodel_34_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_34_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_34_cdp2,Yumodel_34_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_34_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_34_cdp3,Yumodel_34_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_35(x,y,z) = corr(BrainList,Yumodel_35,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_35_ts1(x,y,z) = corr(BrainList,Yumodel_35_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_35_ts2(x,y,z) = corr(BrainList,Yumodel_35_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_35_ts3(x,y,z) = corr(BrainList,Yumodel_35_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_35_cdp1(x,y,z) = corr(BrainList,Yumodel_35_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_35_cdp2(x,y,z) = corr(BrainList,Yumodel_35_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_35_cdp3(x,y,z) = corr(BrainList,Yumodel_35_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_35_gist(x,y,z) = partialcorr(BrainList,Yumodel_35,Yumodel_35_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_35_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_35_ts1,Yumodel_35_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_35_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_35_ts2,Yumodel_35_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_35_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_35_ts3,Yumodel_35_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_35_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_35_cdp1,Yumodel_35_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_35_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_35_cdp2,Yumodel_35_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_35_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_35_cdp3,Yumodel_35_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_36(x,y,z) = corr(BrainList,Yumodel_36,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_36_ts1(x,y,z) = corr(BrainList,Yumodel_36_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_36_ts2(x,y,z) = corr(BrainList,Yumodel_36_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_36_ts3(x,y,z) = corr(BrainList,Yumodel_36_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_36_cdp1(x,y,z) = corr(BrainList,Yumodel_36_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_36_cdp2(x,y,z) = corr(BrainList,Yumodel_36_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_36_cdp3(x,y,z) = corr(BrainList,Yumodel_36_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_36_gist(x,y,z) = partialcorr(BrainList,Yumodel_36,Yumodel_36_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_36_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_36_ts1,Yumodel_36_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_36_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_36_ts2,Yumodel_36_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_36_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_36_ts3,Yumodel_36_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_36_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_36_cdp1,Yumodel_36_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_36_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_36_cdp2,Yumodel_36_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_36_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_36_cdp3,Yumodel_36_last,'type','Spearman','rows','all','tail','both');
                    
                    corrT_Yumodel_37(x,y,z) = corr(BrainList,Yumodel_37,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_37_ts1(x,y,z) = corr(BrainList,Yumodel_37_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_37_ts2(x,y,z) = corr(BrainList,Yumodel_37_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_37_ts3(x,y,z) = corr(BrainList,Yumodel_37_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_37_cdp1(x,y,z) = corr(BrainList,Yumodel_37_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_37_cdp2(x,y,z) = corr(BrainList,Yumodel_37_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_37_cdp3(x,y,z) = corr(BrainList,Yumodel_37_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_37_gist(x,y,z) = partialcorr(BrainList,Yumodel_37,Yumodel_37_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_37_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_37_ts1,Yumodel_37_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_37_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_37_ts2,Yumodel_37_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_37_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_37_ts3,Yumodel_37_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_37_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_37_cdp1,Yumodel_37_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_37_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_37_cdp2,Yumodel_37_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_37_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_37_cdp3,Yumodel_37_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_38(x,y,z) = corr(BrainList,Yumodel_38,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_38_ts1(x,y,z) = corr(BrainList,Yumodel_38_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_38_ts2(x,y,z) = corr(BrainList,Yumodel_38_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_38_ts3(x,y,z) = corr(BrainList,Yumodel_38_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_38_cdp1(x,y,z) = corr(BrainList,Yumodel_38_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_38_cdp2(x,y,z) = corr(BrainList,Yumodel_38_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_38_cdp3(x,y,z) = corr(BrainList,Yumodel_38_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_38_gist(x,y,z) = partialcorr(BrainList,Yumodel_38,Yumodel_38_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_38_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_38_ts1,Yumodel_38_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_38_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_38_ts2,Yumodel_38_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_38_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_38_ts3,Yumodel_38_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_38_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_38_cdp1,Yumodel_38_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_38_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_38_cdp2,Yumodel_38_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_38_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_38_cdp3,Yumodel_38_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_39(x,y,z) = corr(BrainList,Yumodel_39,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_39_ts1(x,y,z) = corr(BrainList,Yumodel_39_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_39_ts2(x,y,z) = corr(BrainList,Yumodel_39_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_39_ts3(x,y,z) = corr(BrainList,Yumodel_39_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_39_cdp1(x,y,z) = corr(BrainList,Yumodel_39_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_39_cdp2(x,y,z) = corr(BrainList,Yumodel_39_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_39_cdp3(x,y,z) = corr(BrainList,Yumodel_39_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_39_gist(x,y,z) = partialcorr(BrainList,Yumodel_39,Yumodel_39_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_39_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_39_ts1,Yumodel_39_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_39_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_39_ts2,Yumodel_39_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_39_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_39_ts3,Yumodel_39_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_39_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_39_cdp1,Yumodel_39_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_39_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_39_cdp2,Yumodel_39_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_39_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_39_cdp3,Yumodel_39_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_40(x,y,z) = corr(BrainList,Yumodel_40,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_40_ts1(x,y,z) = corr(BrainList,Yumodel_40_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_40_ts2(x,y,z) = corr(BrainList,Yumodel_40_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_40_ts3(x,y,z) = corr(BrainList,Yumodel_40_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_40_cdp1(x,y,z) = corr(BrainList,Yumodel_40_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_40_cdp2(x,y,z) = corr(BrainList,Yumodel_40_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_40_cdp3(x,y,z) = corr(BrainList,Yumodel_40_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_40_gist(x,y,z) = partialcorr(BrainList,Yumodel_40,Yumodel_40_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_40_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_40_ts1,Yumodel_40_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_40_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_40_ts2,Yumodel_40_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_40_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_40_ts3,Yumodel_40_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_40_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_40_cdp1,Yumodel_40_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_40_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_40_cdp2,Yumodel_40_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_40_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_40_cdp3,Yumodel_40_last,'type','Spearman','rows','all','tail','both');

                    PrintN = PrintN + 1;
                    
                    display([Subject(subj).name,': ',num2str(PrintN),'/',num2str(PrintSum)]);
    
                end
            end
        end
    end
    
    % write_img_file
    rest_writefile(corrT_Yumodel_31,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_31.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_31_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_31_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_31_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_31_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_31_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_31_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_31_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_31_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_31_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_31_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_31_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_31_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_31_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_31_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_31_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_31_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_31_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_31_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_31_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_31_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_31_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_31_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_31_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_31_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_31_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_31_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_32,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_32.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_32_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_32_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_32_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_32_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_32_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_32_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_32_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_32_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_32_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_32_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_32_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_32_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_32_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_32_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_32_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_32_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_32_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_32_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_32_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_32_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_32_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_32_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_32_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_32_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_32_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_32_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');
    
    rest_writefile(corrT_Yumodel_33,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_33.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_33_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_33_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_33_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_33_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_33_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_33_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_33_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_33_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_33_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_33_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_33_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_33_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_33_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_33_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_33_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_33_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_33_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_33_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_33_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_33_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_33_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_33_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_33_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_33_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_33_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_33_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_34,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_34.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_34_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_34_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_34_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_34_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_34_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_34_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_34_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_34_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_34_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_34_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_34_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_34_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_34_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_34_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_34_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_34_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_34_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_34_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_34_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_34_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_34_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_34_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_34_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_34_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_34_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_34_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_35,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_35.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_35_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_35_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_35_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_35_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_35_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_35_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_35_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_35_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_35_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_35_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_35_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_35_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_35_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_35_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_35_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_35_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_35_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_35_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_35_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_35_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_35_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_35_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_35_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_35_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_35_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_35_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_36,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_36.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_36_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_36_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_36_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_36_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_36_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_36_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_36_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_36_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_36_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_36_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_36_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_36_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_36_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_36_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_36_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_36_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_36_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_36_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_36_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_36_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_36_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_36_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_36_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_36_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_36_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_36_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');
    
    rest_writefile(corrT_Yumodel_37,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_37.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_37_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_37_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_37_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_37_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_37_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_37_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_37_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_37_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_37_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_37_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_37_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_37_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_37_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_37_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_37_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_37_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_37_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_37_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_37_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_37_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_37_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_37_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_37_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_37_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_37_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_37_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_38,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_38.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_38_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_38_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_38_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_38_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_38_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_38_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_38_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_38_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_38_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_38_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_38_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_38_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_38_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_38_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_38_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_38_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_38_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_38_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_38_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_38_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_38_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_38_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_38_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_38_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_38_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_38_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_39,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_39.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_39_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_39_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_39_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_39_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_39_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_39_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_39_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_39_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_39_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_39_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_39_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_39_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_39_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_39_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_39_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_39_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_39_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_39_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_39_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_39_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_39_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_39_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_39_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_39_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_39_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_39_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_40,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_40.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_40_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_40_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_40_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_40_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_40_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_40_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_40_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_40_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_40_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_40_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_40_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_40_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_40_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_40_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_40_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_40_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_40_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_40_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_40_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_40_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_40_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_40_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_40_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_40_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_40_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_40_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

end
