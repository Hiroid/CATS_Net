function A0_RSA_one_subj_41to50(subj)

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
    load(['lib' filesep 'Yumodel_RDM_eval_TransE_41to50.mat']);
    load(['lib' filesep 'feature_fe_RDM_eval_TransE_41to50.mat']);

    % % Enter the File Path of the T Images here:
    TImagePath = '/data0/user/lxguo/Data/BNU/A0_WT95_picture_spmT_DIDed';
    % % Enter the Output File Path here:
    OutputPath = ['A1_Searchlight' filesep 'A0_all_results_eval_TransE_41to50'];
    Subject=dir([TImagePath filesep 'SUB*']);

    % Load Structure Image of Each Subject
    [Mask,VoxDim,Header] = rest_readfile(greymask_path);
    PrintSum = sum(sum(sum(int8(Mask > 1/3)))); % logic

    % load the ref MNI space and transform
    Vref = spm_vol(greymask_path);
    [~, mmCoords] = spm_read_vols(Vref); % return all the voxel's xyz

    % Load Each Model RDM
    % ---------------------------------------------------------------
    RN50_conv1 = Matrix2List(feature_fe_RDM.model_41_fe_conv1);
    RN50_layer1 = Matrix2List(feature_fe_RDM.model_41_fe_layer1);
    RN50_layer2 = Matrix2List(feature_fe_RDM.model_41_fe_layer2);
    RN50_layer3 = Matrix2List(feature_fe_RDM.model_41_fe_layer3);
    RN50_layer4 = Matrix2List(feature_fe_RDM.model_41_fe_layer4);
    RN50_last = Matrix2List(feature_fe_RDM.model_41_fe_last);
    
    Yumodel_41_ts1 = Matrix2List(YuRDM.model_41_ts_1);
    Yumodel_41_ts2 = Matrix2List(YuRDM.model_41_ts_2);
    Yumodel_41_ts3 = Matrix2List(YuRDM.model_41_ts_3);
    Yumodel_41_cdp1 = Matrix2List(YuRDM.model_41_cdp_1);
    Yumodel_41_cdp2 = Matrix2List(YuRDM.model_41_cdp_2);
    Yumodel_41_cdp3 = Matrix2List(YuRDM.model_41_cdp_3);
    Yumodel_41_conv1 = Matrix2List(feature_fe_RDM.model_41_fe_conv1);
    Yumodel_41_layer1 = Matrix2List(feature_fe_RDM.model_41_fe_layer1);
    Yumodel_41_layer2 = Matrix2List(feature_fe_RDM.model_41_fe_layer2);
    Yumodel_41_layer3 = Matrix2List(feature_fe_RDM.model_41_fe_layer3);
    Yumodel_41_layer4 = Matrix2List(feature_fe_RDM.model_41_fe_layer4);
    Yumodel_41_last = Matrix2List(feature_fe_RDM.model_41_fe_last);
    Yumodel_41 = Matrix2List(YuRDM.model_41);

    Yumodel_42_ts1 = Matrix2List(YuRDM.model_42_ts_1);
    Yumodel_42_ts2 = Matrix2List(YuRDM.model_42_ts_2);
    Yumodel_42_ts3 = Matrix2List(YuRDM.model_42_ts_3);
    Yumodel_42_cdp1 = Matrix2List(YuRDM.model_42_cdp_1);
    Yumodel_42_cdp2 = Matrix2List(YuRDM.model_42_cdp_2);
    Yumodel_42_cdp3 = Matrix2List(YuRDM.model_42_cdp_3);
    Yumodel_42_conv1 = Matrix2List(feature_fe_RDM.model_42_fe_conv1);
    Yumodel_42_layer1 = Matrix2List(feature_fe_RDM.model_42_fe_layer1);
    Yumodel_42_layer2 = Matrix2List(feature_fe_RDM.model_42_fe_layer2);
    Yumodel_42_layer3 = Matrix2List(feature_fe_RDM.model_42_fe_layer3);
    Yumodel_42_layer4 = Matrix2List(feature_fe_RDM.model_42_fe_layer4);
    Yumodel_42_last = Matrix2List(feature_fe_RDM.model_42_fe_last);
    Yumodel_42 = Matrix2List(YuRDM.model_42);

    Yumodel_43_ts1 = Matrix2List(YuRDM.model_43_ts_1);
    Yumodel_43_ts2 = Matrix2List(YuRDM.model_43_ts_2);
    Yumodel_43_ts3 = Matrix2List(YuRDM.model_43_ts_3);
    Yumodel_43_cdp1 = Matrix2List(YuRDM.model_43_cdp_1);
    Yumodel_43_cdp2 = Matrix2List(YuRDM.model_43_cdp_2);
    Yumodel_43_cdp3 = Matrix2List(YuRDM.model_43_cdp_3);
    Yumodel_43_conv1 = Matrix2List(feature_fe_RDM.model_43_fe_conv1);
    Yumodel_43_layer1 = Matrix2List(feature_fe_RDM.model_43_fe_layer1);
    Yumodel_43_layer2 = Matrix2List(feature_fe_RDM.model_43_fe_layer2);
    Yumodel_43_layer3 = Matrix2List(feature_fe_RDM.model_43_fe_layer3);
    Yumodel_43_layer4 = Matrix2List(feature_fe_RDM.model_43_fe_layer4);
    Yumodel_43_last = Matrix2List(feature_fe_RDM.model_43_fe_last);
    Yumodel_43 = Matrix2List(YuRDM.model_43);

    Yumodel_44_ts1 = Matrix2List(YuRDM.model_44_ts_1);
    Yumodel_44_ts2 = Matrix2List(YuRDM.model_44_ts_2);
    Yumodel_44_ts3 = Matrix2List(YuRDM.model_44_ts_3);
    Yumodel_44_cdp1 = Matrix2List(YuRDM.model_44_cdp_1);
    Yumodel_44_cdp2 = Matrix2List(YuRDM.model_44_cdp_2);
    Yumodel_44_cdp3 = Matrix2List(YuRDM.model_44_cdp_3);
    Yumodel_44_conv1 = Matrix2List(feature_fe_RDM.model_44_fe_conv1);
    Yumodel_44_layer1 = Matrix2List(feature_fe_RDM.model_44_fe_layer1);
    Yumodel_44_layer2 = Matrix2List(feature_fe_RDM.model_44_fe_layer2);
    Yumodel_44_layer3 = Matrix2List(feature_fe_RDM.model_44_fe_layer3);
    Yumodel_44_layer4 = Matrix2List(feature_fe_RDM.model_44_fe_layer4);
    Yumodel_44_last = Matrix2List(feature_fe_RDM.model_44_fe_last);
    Yumodel_44 = Matrix2List(YuRDM.model_44);

    Yumodel_45_ts1 = Matrix2List(YuRDM.model_45_ts_1);
    Yumodel_45_ts2 = Matrix2List(YuRDM.model_45_ts_2);
    Yumodel_45_ts3 = Matrix2List(YuRDM.model_45_ts_3);
    Yumodel_45_cdp1 = Matrix2List(YuRDM.model_45_cdp_1);
    Yumodel_45_cdp2 = Matrix2List(YuRDM.model_45_cdp_2);
    Yumodel_45_cdp3 = Matrix2List(YuRDM.model_45_cdp_3);
    Yumodel_45_conv1 = Matrix2List(feature_fe_RDM.model_45_fe_conv1);
    Yumodel_45_layer1 = Matrix2List(feature_fe_RDM.model_45_fe_layer1);
    Yumodel_45_layer2 = Matrix2List(feature_fe_RDM.model_45_fe_layer2);
    Yumodel_45_layer3 = Matrix2List(feature_fe_RDM.model_45_fe_layer3);
    Yumodel_45_layer4 = Matrix2List(feature_fe_RDM.model_45_fe_layer4);
    Yumodel_45_last = Matrix2List(feature_fe_RDM.model_45_fe_last);
    Yumodel_45 = Matrix2List(YuRDM.model_45);

    Yumodel_46_ts1 = Matrix2List(YuRDM.model_46_ts_1);
    Yumodel_46_ts2 = Matrix2List(YuRDM.model_46_ts_2);
    Yumodel_46_ts3 = Matrix2List(YuRDM.model_46_ts_3);
    Yumodel_46_cdp1 = Matrix2List(YuRDM.model_46_cdp_1);
    Yumodel_46_cdp2 = Matrix2List(YuRDM.model_46_cdp_2);
    Yumodel_46_cdp3 = Matrix2List(YuRDM.model_46_cdp_3);
    Yumodel_46_conv1 = Matrix2List(feature_fe_RDM.model_46_fe_conv1);
    Yumodel_46_layer1 = Matrix2List(feature_fe_RDM.model_46_fe_layer1);
    Yumodel_46_layer2 = Matrix2List(feature_fe_RDM.model_46_fe_layer2);
    Yumodel_46_layer3 = Matrix2List(feature_fe_RDM.model_46_fe_layer3);
    Yumodel_46_layer4 = Matrix2List(feature_fe_RDM.model_46_fe_layer4);
    Yumodel_46_last = Matrix2List(feature_fe_RDM.model_46_fe_last);
    Yumodel_46 = Matrix2List(YuRDM.model_46);

    Yumodel_47_ts1 = Matrix2List(YuRDM.model_47_ts_1);
    Yumodel_47_ts2 = Matrix2List(YuRDM.model_47_ts_2);
    Yumodel_47_ts3 = Matrix2List(YuRDM.model_47_ts_3);
    Yumodel_47_cdp1 = Matrix2List(YuRDM.model_47_cdp_1);
    Yumodel_47_cdp2 = Matrix2List(YuRDM.model_47_cdp_2);
    Yumodel_47_cdp3 = Matrix2List(YuRDM.model_47_cdp_3);
    Yumodel_47_conv1 = Matrix2List(feature_fe_RDM.model_47_fe_conv1);
    Yumodel_47_layer1 = Matrix2List(feature_fe_RDM.model_47_fe_layer1);
    Yumodel_47_layer2 = Matrix2List(feature_fe_RDM.model_47_fe_layer2);
    Yumodel_47_layer3 = Matrix2List(feature_fe_RDM.model_47_fe_layer3);
    Yumodel_47_layer4 = Matrix2List(feature_fe_RDM.model_47_fe_layer4);
    Yumodel_47_last = Matrix2List(feature_fe_RDM.model_47_fe_last);
    Yumodel_47 = Matrix2List(YuRDM.model_47);

    Yumodel_48_ts1 = Matrix2List(YuRDM.model_48_ts_1);
    Yumodel_48_ts2 = Matrix2List(YuRDM.model_48_ts_2);
    Yumodel_48_ts3 = Matrix2List(YuRDM.model_48_ts_3);
    Yumodel_48_cdp1 = Matrix2List(YuRDM.model_48_cdp_1);
    Yumodel_48_cdp2 = Matrix2List(YuRDM.model_48_cdp_2);
    Yumodel_48_cdp3 = Matrix2List(YuRDM.model_48_cdp_3);
    Yumodel_48_conv1 = Matrix2List(feature_fe_RDM.model_48_fe_conv1);
    Yumodel_48_layer1 = Matrix2List(feature_fe_RDM.model_48_fe_layer1);
    Yumodel_48_layer2 = Matrix2List(feature_fe_RDM.model_48_fe_layer2);
    Yumodel_48_layer3 = Matrix2List(feature_fe_RDM.model_48_fe_layer3);
    Yumodel_48_layer4 = Matrix2List(feature_fe_RDM.model_48_fe_layer4);
    Yumodel_48_last = Matrix2List(feature_fe_RDM.model_48_fe_last);
    Yumodel_48 = Matrix2List(YuRDM.model_48);

    Yumodel_49_ts1 = Matrix2List(YuRDM.model_49_ts_1);
    Yumodel_49_ts2 = Matrix2List(YuRDM.model_49_ts_2);
    Yumodel_49_ts3 = Matrix2List(YuRDM.model_49_ts_3);
    Yumodel_49_cdp1 = Matrix2List(YuRDM.model_49_cdp_1);
    Yumodel_49_cdp2 = Matrix2List(YuRDM.model_49_cdp_2);
    Yumodel_49_cdp3 = Matrix2List(YuRDM.model_49_cdp_3);
    Yumodel_49_conv1 = Matrix2List(feature_fe_RDM.model_49_fe_conv1);
    Yumodel_49_layer1 = Matrix2List(feature_fe_RDM.model_49_fe_layer1);
    Yumodel_49_layer2 = Matrix2List(feature_fe_RDM.model_49_fe_layer2);
    Yumodel_49_layer3 = Matrix2List(feature_fe_RDM.model_49_fe_layer3);
    Yumodel_49_layer4 = Matrix2List(feature_fe_RDM.model_49_fe_layer4);
    Yumodel_49_last = Matrix2List(feature_fe_RDM.model_49_fe_last);
    Yumodel_49 = Matrix2List(YuRDM.model_49);

    Yumodel_50_ts1 = Matrix2List(YuRDM.model_50_ts_1);
    Yumodel_50_ts2 = Matrix2List(YuRDM.model_50_ts_2);
    Yumodel_50_ts3 = Matrix2List(YuRDM.model_50_ts_3);
    Yumodel_50_cdp1 = Matrix2List(YuRDM.model_50_cdp_1);
    Yumodel_50_cdp2 = Matrix2List(YuRDM.model_50_cdp_2);
    Yumodel_50_cdp3 = Matrix2List(YuRDM.model_50_cdp_3);
    Yumodel_50_conv1 = Matrix2List(feature_fe_RDM.model_50_fe_conv1);
    Yumodel_50_layer1 = Matrix2List(feature_fe_RDM.model_50_fe_layer1);
    Yumodel_50_layer2 = Matrix2List(feature_fe_RDM.model_50_fe_layer2);
    Yumodel_50_layer3 = Matrix2List(feature_fe_RDM.model_50_fe_layer3);
    Yumodel_50_layer4 = Matrix2List(feature_fe_RDM.model_50_fe_layer4);
    Yumodel_50_last = Matrix2List(feature_fe_RDM.model_50_fe_last);
    Yumodel_50 = Matrix2List(YuRDM.model_50);


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
    corrT_Yumodel_41 = zeros(Header.dim);
    corrT_Yumodel_41_ts1 = zeros(Header.dim);
    corrT_Yumodel_41_ts2 = zeros(Header.dim);
    corrT_Yumodel_41_ts3 = zeros(Header.dim);
    corrT_Yumodel_41_cdp1 = zeros(Header.dim);
    corrT_Yumodel_41_cdp2 = zeros(Header.dim);
    corrT_Yumodel_41_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_41_gist = zeros(Header.dim);
    pcorrT_Yumodel_41_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_41_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_41_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_41_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_41_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_41_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_42 = zeros(Header.dim);
    corrT_Yumodel_42_ts1 = zeros(Header.dim);
    corrT_Yumodel_42_ts2 = zeros(Header.dim);
    corrT_Yumodel_42_ts3 = zeros(Header.dim);
    corrT_Yumodel_42_cdp1 = zeros(Header.dim);
    corrT_Yumodel_42_cdp2 = zeros(Header.dim);
    corrT_Yumodel_42_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_42_gist = zeros(Header.dim);
    pcorrT_Yumodel_42_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_42_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_42_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_42_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_42_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_42_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_43 = zeros(Header.dim);
    corrT_Yumodel_43_ts1 = zeros(Header.dim);
    corrT_Yumodel_43_ts2 = zeros(Header.dim);
    corrT_Yumodel_43_ts3 = zeros(Header.dim);
    corrT_Yumodel_43_cdp1 = zeros(Header.dim);
    corrT_Yumodel_43_cdp2 = zeros(Header.dim);
    corrT_Yumodel_43_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_43_gist = zeros(Header.dim);
    pcorrT_Yumodel_43_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_43_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_43_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_43_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_43_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_43_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_44 = zeros(Header.dim);
    corrT_Yumodel_44_ts1 = zeros(Header.dim);
    corrT_Yumodel_44_ts2 = zeros(Header.dim);
    corrT_Yumodel_44_ts3 = zeros(Header.dim);
    corrT_Yumodel_44_cdp1 = zeros(Header.dim);
    corrT_Yumodel_44_cdp2 = zeros(Header.dim);
    corrT_Yumodel_44_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_44_gist = zeros(Header.dim);
    pcorrT_Yumodel_44_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_44_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_44_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_44_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_44_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_44_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_45 = zeros(Header.dim);
    corrT_Yumodel_45_ts1 = zeros(Header.dim);
    corrT_Yumodel_45_ts2 = zeros(Header.dim);
    corrT_Yumodel_45_ts3 = zeros(Header.dim);
    corrT_Yumodel_45_cdp1 = zeros(Header.dim);
    corrT_Yumodel_45_cdp2 = zeros(Header.dim);
    corrT_Yumodel_45_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_45_gist = zeros(Header.dim);
    pcorrT_Yumodel_45_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_45_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_45_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_45_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_45_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_45_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_46 = zeros(Header.dim);
    corrT_Yumodel_46_ts1 = zeros(Header.dim);
    corrT_Yumodel_46_ts2 = zeros(Header.dim);
    corrT_Yumodel_46_ts3 = zeros(Header.dim);
    corrT_Yumodel_46_cdp1 = zeros(Header.dim);
    corrT_Yumodel_46_cdp2 = zeros(Header.dim);
    corrT_Yumodel_46_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_46_gist = zeros(Header.dim);
    pcorrT_Yumodel_46_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_46_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_46_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_46_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_46_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_46_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_47 = zeros(Header.dim);
    corrT_Yumodel_47_ts1 = zeros(Header.dim);
    corrT_Yumodel_47_ts2 = zeros(Header.dim);
    corrT_Yumodel_47_ts3 = zeros(Header.dim);
    corrT_Yumodel_47_cdp1 = zeros(Header.dim);
    corrT_Yumodel_47_cdp2 = zeros(Header.dim);
    corrT_Yumodel_47_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_47_gist = zeros(Header.dim);
    pcorrT_Yumodel_47_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_47_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_47_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_47_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_47_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_47_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_48 = zeros(Header.dim);
    corrT_Yumodel_48_ts1 = zeros(Header.dim);
    corrT_Yumodel_48_ts2 = zeros(Header.dim);
    corrT_Yumodel_48_ts3 = zeros(Header.dim);
    corrT_Yumodel_48_cdp1 = zeros(Header.dim);
    corrT_Yumodel_48_cdp2 = zeros(Header.dim);
    corrT_Yumodel_48_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_48_gist = zeros(Header.dim);
    pcorrT_Yumodel_48_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_48_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_48_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_48_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_48_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_48_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_49 = zeros(Header.dim);
    corrT_Yumodel_49_ts1 = zeros(Header.dim);
    corrT_Yumodel_49_ts2 = zeros(Header.dim);
    corrT_Yumodel_49_ts3 = zeros(Header.dim);
    corrT_Yumodel_49_cdp1 = zeros(Header.dim);
    corrT_Yumodel_49_cdp2 = zeros(Header.dim);
    corrT_Yumodel_49_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_49_gist = zeros(Header.dim);
    pcorrT_Yumodel_49_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_49_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_49_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_49_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_49_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_49_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_50 = zeros(Header.dim);
    corrT_Yumodel_50_ts1 = zeros(Header.dim);
    corrT_Yumodel_50_ts2 = zeros(Header.dim);
    corrT_Yumodel_50_ts3 = zeros(Header.dim);
    corrT_Yumodel_50_cdp1 = zeros(Header.dim);
    corrT_Yumodel_50_cdp2 = zeros(Header.dim);
    corrT_Yumodel_50_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_50_gist = zeros(Header.dim);
    pcorrT_Yumodel_50_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_50_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_50_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_50_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_50_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_50_cdp3_gist = zeros(Header.dim);
    
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
                    corrT_Yumodel_41(x,y,z) = corr(BrainList,Yumodel_41,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_41_ts1(x,y,z) = corr(BrainList,Yumodel_41_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_41_ts2(x,y,z) = corr(BrainList,Yumodel_41_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_41_ts3(x,y,z) = corr(BrainList,Yumodel_41_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_41_cdp1(x,y,z) = corr(BrainList,Yumodel_41_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_41_cdp2(x,y,z) = corr(BrainList,Yumodel_41_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_41_cdp3(x,y,z) = corr(BrainList,Yumodel_41_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_41_gist(x,y,z) = partialcorr(BrainList,Yumodel_41,Yumodel_41_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_41_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_41_ts1,Yumodel_41_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_41_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_41_ts2,Yumodel_41_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_41_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_41_ts3,Yumodel_41_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_41_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_41_cdp1,Yumodel_41_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_41_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_41_cdp2,Yumodel_41_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_41_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_41_cdp3,Yumodel_41_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_42(x,y,z) = corr(BrainList,Yumodel_42,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_42_ts1(x,y,z) = corr(BrainList,Yumodel_42_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_42_ts2(x,y,z) = corr(BrainList,Yumodel_42_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_42_ts3(x,y,z) = corr(BrainList,Yumodel_42_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_42_cdp1(x,y,z) = corr(BrainList,Yumodel_42_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_42_cdp2(x,y,z) = corr(BrainList,Yumodel_42_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_42_cdp3(x,y,z) = corr(BrainList,Yumodel_42_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_42_gist(x,y,z) = partialcorr(BrainList,Yumodel_42,Yumodel_42_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_42_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_42_ts1,Yumodel_42_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_42_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_42_ts2,Yumodel_42_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_42_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_42_ts3,Yumodel_42_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_42_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_42_cdp1,Yumodel_42_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_42_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_42_cdp2,Yumodel_42_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_42_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_42_cdp3,Yumodel_42_last,'type','Spearman','rows','all','tail','both');
                    
                    corrT_Yumodel_43(x,y,z) = corr(BrainList,Yumodel_43,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_43_ts1(x,y,z) = corr(BrainList,Yumodel_43_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_43_ts2(x,y,z) = corr(BrainList,Yumodel_43_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_43_ts3(x,y,z) = corr(BrainList,Yumodel_43_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_43_cdp1(x,y,z) = corr(BrainList,Yumodel_43_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_43_cdp2(x,y,z) = corr(BrainList,Yumodel_43_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_43_cdp3(x,y,z) = corr(BrainList,Yumodel_43_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_43_gist(x,y,z) = partialcorr(BrainList,Yumodel_43,Yumodel_43_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_43_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_43_ts1,Yumodel_43_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_43_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_43_ts2,Yumodel_43_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_43_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_43_ts3,Yumodel_43_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_43_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_43_cdp1,Yumodel_43_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_43_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_43_cdp2,Yumodel_43_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_43_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_43_cdp3,Yumodel_43_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_44(x,y,z) = corr(BrainList,Yumodel_44,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_44_ts1(x,y,z) = corr(BrainList,Yumodel_44_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_44_ts2(x,y,z) = corr(BrainList,Yumodel_44_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_44_ts3(x,y,z) = corr(BrainList,Yumodel_44_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_44_cdp1(x,y,z) = corr(BrainList,Yumodel_44_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_44_cdp2(x,y,z) = corr(BrainList,Yumodel_44_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_44_cdp3(x,y,z) = corr(BrainList,Yumodel_44_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_44_gist(x,y,z) = partialcorr(BrainList,Yumodel_44,Yumodel_44_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_44_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_44_ts1,Yumodel_44_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_44_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_44_ts2,Yumodel_44_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_44_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_44_ts3,Yumodel_44_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_44_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_44_cdp1,Yumodel_44_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_44_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_44_cdp2,Yumodel_44_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_44_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_44_cdp3,Yumodel_44_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_45(x,y,z) = corr(BrainList,Yumodel_45,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_45_ts1(x,y,z) = corr(BrainList,Yumodel_45_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_45_ts2(x,y,z) = corr(BrainList,Yumodel_45_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_45_ts3(x,y,z) = corr(BrainList,Yumodel_45_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_45_cdp1(x,y,z) = corr(BrainList,Yumodel_45_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_45_cdp2(x,y,z) = corr(BrainList,Yumodel_45_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_45_cdp3(x,y,z) = corr(BrainList,Yumodel_45_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_45_gist(x,y,z) = partialcorr(BrainList,Yumodel_45,Yumodel_45_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_45_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_45_ts1,Yumodel_45_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_45_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_45_ts2,Yumodel_45_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_45_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_45_ts3,Yumodel_45_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_45_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_45_cdp1,Yumodel_45_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_45_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_45_cdp2,Yumodel_45_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_45_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_45_cdp3,Yumodel_45_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_46(x,y,z) = corr(BrainList,Yumodel_46,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_46_ts1(x,y,z) = corr(BrainList,Yumodel_46_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_46_ts2(x,y,z) = corr(BrainList,Yumodel_46_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_46_ts3(x,y,z) = corr(BrainList,Yumodel_46_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_46_cdp1(x,y,z) = corr(BrainList,Yumodel_46_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_46_cdp2(x,y,z) = corr(BrainList,Yumodel_46_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_46_cdp3(x,y,z) = corr(BrainList,Yumodel_46_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_46_gist(x,y,z) = partialcorr(BrainList,Yumodel_46,Yumodel_46_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_46_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_46_ts1,Yumodel_46_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_46_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_46_ts2,Yumodel_46_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_46_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_46_ts3,Yumodel_46_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_46_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_46_cdp1,Yumodel_46_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_46_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_46_cdp2,Yumodel_46_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_46_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_46_cdp3,Yumodel_46_last,'type','Spearman','rows','all','tail','both');
                    
                    corrT_Yumodel_47(x,y,z) = corr(BrainList,Yumodel_47,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_47_ts1(x,y,z) = corr(BrainList,Yumodel_47_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_47_ts2(x,y,z) = corr(BrainList,Yumodel_47_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_47_ts3(x,y,z) = corr(BrainList,Yumodel_47_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_47_cdp1(x,y,z) = corr(BrainList,Yumodel_47_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_47_cdp2(x,y,z) = corr(BrainList,Yumodel_47_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_47_cdp3(x,y,z) = corr(BrainList,Yumodel_47_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_47_gist(x,y,z) = partialcorr(BrainList,Yumodel_47,Yumodel_47_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_47_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_47_ts1,Yumodel_47_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_47_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_47_ts2,Yumodel_47_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_47_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_47_ts3,Yumodel_47_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_47_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_47_cdp1,Yumodel_47_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_47_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_47_cdp2,Yumodel_47_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_47_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_47_cdp3,Yumodel_47_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_48(x,y,z) = corr(BrainList,Yumodel_48,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_48_ts1(x,y,z) = corr(BrainList,Yumodel_48_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_48_ts2(x,y,z) = corr(BrainList,Yumodel_48_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_48_ts3(x,y,z) = corr(BrainList,Yumodel_48_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_48_cdp1(x,y,z) = corr(BrainList,Yumodel_48_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_48_cdp2(x,y,z) = corr(BrainList,Yumodel_48_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_48_cdp3(x,y,z) = corr(BrainList,Yumodel_48_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_48_gist(x,y,z) = partialcorr(BrainList,Yumodel_48,Yumodel_48_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_48_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_48_ts1,Yumodel_48_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_48_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_48_ts2,Yumodel_48_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_48_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_48_ts3,Yumodel_48_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_48_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_48_cdp1,Yumodel_48_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_48_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_48_cdp2,Yumodel_48_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_48_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_48_cdp3,Yumodel_48_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_49(x,y,z) = corr(BrainList,Yumodel_49,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_49_ts1(x,y,z) = corr(BrainList,Yumodel_49_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_49_ts2(x,y,z) = corr(BrainList,Yumodel_49_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_49_ts3(x,y,z) = corr(BrainList,Yumodel_49_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_49_cdp1(x,y,z) = corr(BrainList,Yumodel_49_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_49_cdp2(x,y,z) = corr(BrainList,Yumodel_49_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_49_cdp3(x,y,z) = corr(BrainList,Yumodel_49_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_49_gist(x,y,z) = partialcorr(BrainList,Yumodel_49,Yumodel_49_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_49_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_49_ts1,Yumodel_49_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_49_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_49_ts2,Yumodel_49_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_49_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_49_ts3,Yumodel_49_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_49_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_49_cdp1,Yumodel_49_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_49_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_49_cdp2,Yumodel_49_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_49_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_49_cdp3,Yumodel_49_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_50(x,y,z) = corr(BrainList,Yumodel_50,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_50_ts1(x,y,z) = corr(BrainList,Yumodel_50_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_50_ts2(x,y,z) = corr(BrainList,Yumodel_50_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_50_ts3(x,y,z) = corr(BrainList,Yumodel_50_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_50_cdp1(x,y,z) = corr(BrainList,Yumodel_50_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_50_cdp2(x,y,z) = corr(BrainList,Yumodel_50_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_50_cdp3(x,y,z) = corr(BrainList,Yumodel_50_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_50_gist(x,y,z) = partialcorr(BrainList,Yumodel_50,Yumodel_50_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_50_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_50_ts1,Yumodel_50_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_50_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_50_ts2,Yumodel_50_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_50_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_50_ts3,Yumodel_50_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_50_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_50_cdp1,Yumodel_50_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_50_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_50_cdp2,Yumodel_50_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_50_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_50_cdp3,Yumodel_50_last,'type','Spearman','rows','all','tail','both');

                    PrintN = PrintN + 1;
                    
                    display([Subject(subj).name,': ',num2str(PrintN),'/',num2str(PrintSum)]);
    
                end
            end
        end
    end
    
    % write_img_file
    rest_writefile(corrT_Yumodel_41,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_41.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_41_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_41_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_41_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_41_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_41_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_41_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_41_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_41_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_41_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_41_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_41_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_41_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_41_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_41_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_41_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_41_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_41_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_41_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_41_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_41_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_41_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_41_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_41_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_41_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_41_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_41_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_42,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_42.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_42_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_42_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_42_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_42_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_42_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_42_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_42_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_42_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_42_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_42_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_42_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_42_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_42_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_42_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_42_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_42_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_42_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_42_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_42_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_42_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_42_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_42_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_42_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_42_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_42_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_42_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');
    
    rest_writefile(corrT_Yumodel_43,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_43.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_43_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_43_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_43_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_43_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_43_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_43_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_43_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_43_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_43_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_43_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_43_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_43_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_43_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_43_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_43_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_43_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_43_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_43_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_43_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_43_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_43_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_43_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_43_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_43_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_43_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_43_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_44,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_44.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_44_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_44_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_44_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_44_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_44_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_44_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_44_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_44_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_44_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_44_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_44_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_44_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_44_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_44_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_44_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_44_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_44_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_44_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_44_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_44_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_44_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_44_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_44_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_44_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_44_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_44_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_45,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_45.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_45_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_45_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_45_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_45_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_45_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_45_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_45_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_45_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_45_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_45_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_45_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_45_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_45_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_45_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_45_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_45_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_45_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_45_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_45_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_45_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_45_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_45_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_45_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_45_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_45_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_45_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_46,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_46.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_46_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_46_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_46_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_46_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_46_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_46_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_46_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_46_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_46_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_46_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_46_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_46_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_46_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_46_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_46_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_46_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_46_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_46_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_46_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_46_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_46_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_46_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_46_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_46_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_46_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_46_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');
    
    rest_writefile(corrT_Yumodel_47,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_47.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_47_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_47_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_47_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_47_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_47_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_47_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_47_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_47_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_47_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_47_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_47_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_47_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_47_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_47_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_47_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_47_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_47_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_47_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_47_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_47_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_47_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_47_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_47_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_47_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_47_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_47_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_48,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_48.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_48_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_48_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_48_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_48_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_48_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_48_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_48_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_48_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_48_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_48_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_48_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_48_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_48_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_48_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_48_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_48_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_48_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_48_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_48_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_48_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_48_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_48_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_48_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_48_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_48_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_48_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_49,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_49.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_49_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_49_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_49_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_49_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_49_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_49_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_49_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_49_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_49_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_49_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_49_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_49_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_49_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_49_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_49_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_49_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_49_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_49_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_49_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_49_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_49_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_49_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_49_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_49_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_49_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_49_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_50,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_50.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_50_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_50_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_50_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_50_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_50_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_50_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_50_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_50_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_50_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_50_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_50_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_50_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_50_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_50_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_50_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_50_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_50_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_50_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_50_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_50_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_50_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_50_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_50_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_50_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_50_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_50_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

end
