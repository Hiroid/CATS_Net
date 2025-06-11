function A0_RSA_one_subj_21to30(subj)

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
    load(['lib' filesep 'Yumodel_RDM_eval_TransE_21to30.mat']);
    load(['lib' filesep 'feature_fe_RDM_eval_TransE_21to30.mat']);

    % % Enter the File Path of the T Images here:
    TImagePath = '/data0/user/lxguo/Data/BNU/A0_WT95_picture_spmT_DIDed';
    % % Enter the Output File Path here:
    OutputPath = ['A1_Searchlight' filesep 'A0_all_results_eval_TransE_21to30'];
    Subject=dir([TImagePath filesep 'SUB*']);

    % Load Structure Image of Each Subject
    [Mask,VoxDim,Header] = rest_readfile(greymask_path);
    PrintSum = sum(sum(sum(int8(Mask > 1/3)))); % logic

    % load the ref MNI space and transform
    Vref = spm_vol(greymask_path);
    [~, mmCoords] = spm_read_vols(Vref); % return all the voxel's xyz

    % Load Each Model RDM
    % ---------------------------------------------------------------
    RN50_conv1 = Matrix2List(feature_fe_RDM.model_21_fe_conv1);
    RN50_layer1 = Matrix2List(feature_fe_RDM.model_21_fe_layer1);
    RN50_layer2 = Matrix2List(feature_fe_RDM.model_21_fe_layer2);
    RN50_layer3 = Matrix2List(feature_fe_RDM.model_21_fe_layer3);
    RN50_layer4 = Matrix2List(feature_fe_RDM.model_21_fe_layer4);
    RN50_last = Matrix2List(feature_fe_RDM.model_21_fe_last);
    
    Yumodel_21_ts1 = Matrix2List(YuRDM.model_21_ts_1);
    Yumodel_21_ts2 = Matrix2List(YuRDM.model_21_ts_2);
    Yumodel_21_ts3 = Matrix2List(YuRDM.model_21_ts_3);
    Yumodel_21_cdp1 = Matrix2List(YuRDM.model_21_cdp_1);
    Yumodel_21_cdp2 = Matrix2List(YuRDM.model_21_cdp_2);
    Yumodel_21_cdp3 = Matrix2List(YuRDM.model_21_cdp_3);
    Yumodel_21_conv1 = Matrix2List(feature_fe_RDM.model_21_fe_conv1);
    Yumodel_21_layer1 = Matrix2List(feature_fe_RDM.model_21_fe_layer1);
    Yumodel_21_layer2 = Matrix2List(feature_fe_RDM.model_21_fe_layer2);
    Yumodel_21_layer3 = Matrix2List(feature_fe_RDM.model_21_fe_layer3);
    Yumodel_21_layer4 = Matrix2List(feature_fe_RDM.model_21_fe_layer4);
    Yumodel_21_last = Matrix2List(feature_fe_RDM.model_21_fe_last);
    Yumodel_21 = Matrix2List(YuRDM.model_21);

    Yumodel_22_ts1 = Matrix2List(YuRDM.model_22_ts_1);
    Yumodel_22_ts2 = Matrix2List(YuRDM.model_22_ts_2);
    Yumodel_22_ts3 = Matrix2List(YuRDM.model_22_ts_3);
    Yumodel_22_cdp1 = Matrix2List(YuRDM.model_22_cdp_1);
    Yumodel_22_cdp2 = Matrix2List(YuRDM.model_22_cdp_2);
    Yumodel_22_cdp3 = Matrix2List(YuRDM.model_22_cdp_3);
    Yumodel_22_conv1 = Matrix2List(feature_fe_RDM.model_22_fe_conv1);
    Yumodel_22_layer1 = Matrix2List(feature_fe_RDM.model_22_fe_layer1);
    Yumodel_22_layer2 = Matrix2List(feature_fe_RDM.model_22_fe_layer2);
    Yumodel_22_layer3 = Matrix2List(feature_fe_RDM.model_22_fe_layer3);
    Yumodel_22_layer4 = Matrix2List(feature_fe_RDM.model_22_fe_layer4);
    Yumodel_22_last = Matrix2List(feature_fe_RDM.model_22_fe_last);
    Yumodel_22 = Matrix2List(YuRDM.model_22);

    Yumodel_23_ts1 = Matrix2List(YuRDM.model_23_ts_1);
    Yumodel_23_ts2 = Matrix2List(YuRDM.model_23_ts_2);
    Yumodel_23_ts3 = Matrix2List(YuRDM.model_23_ts_3);
    Yumodel_23_cdp1 = Matrix2List(YuRDM.model_23_cdp_1);
    Yumodel_23_cdp2 = Matrix2List(YuRDM.model_23_cdp_2);
    Yumodel_23_cdp3 = Matrix2List(YuRDM.model_23_cdp_3);
    Yumodel_23_conv1 = Matrix2List(feature_fe_RDM.model_23_fe_conv1);
    Yumodel_23_layer1 = Matrix2List(feature_fe_RDM.model_23_fe_layer1);
    Yumodel_23_layer2 = Matrix2List(feature_fe_RDM.model_23_fe_layer2);
    Yumodel_23_layer3 = Matrix2List(feature_fe_RDM.model_23_fe_layer3);
    Yumodel_23_layer4 = Matrix2List(feature_fe_RDM.model_23_fe_layer4);
    Yumodel_23_last = Matrix2List(feature_fe_RDM.model_23_fe_last);
    Yumodel_23 = Matrix2List(YuRDM.model_23);

    Yumodel_24_ts1 = Matrix2List(YuRDM.model_24_ts_1);
    Yumodel_24_ts2 = Matrix2List(YuRDM.model_24_ts_2);
    Yumodel_24_ts3 = Matrix2List(YuRDM.model_24_ts_3);
    Yumodel_24_cdp1 = Matrix2List(YuRDM.model_24_cdp_1);
    Yumodel_24_cdp2 = Matrix2List(YuRDM.model_24_cdp_2);
    Yumodel_24_cdp3 = Matrix2List(YuRDM.model_24_cdp_3);
    Yumodel_24_conv1 = Matrix2List(feature_fe_RDM.model_24_fe_conv1);
    Yumodel_24_layer1 = Matrix2List(feature_fe_RDM.model_24_fe_layer1);
    Yumodel_24_layer2 = Matrix2List(feature_fe_RDM.model_24_fe_layer2);
    Yumodel_24_layer3 = Matrix2List(feature_fe_RDM.model_24_fe_layer3);
    Yumodel_24_layer4 = Matrix2List(feature_fe_RDM.model_24_fe_layer4);
    Yumodel_24_last = Matrix2List(feature_fe_RDM.model_24_fe_last);
    Yumodel_24 = Matrix2List(YuRDM.model_24);

    Yumodel_25_ts1 = Matrix2List(YuRDM.model_25_ts_1);
    Yumodel_25_ts2 = Matrix2List(YuRDM.model_25_ts_2);
    Yumodel_25_ts3 = Matrix2List(YuRDM.model_25_ts_3);
    Yumodel_25_cdp1 = Matrix2List(YuRDM.model_25_cdp_1);
    Yumodel_25_cdp2 = Matrix2List(YuRDM.model_25_cdp_2);
    Yumodel_25_cdp3 = Matrix2List(YuRDM.model_25_cdp_3);
    Yumodel_25_conv1 = Matrix2List(feature_fe_RDM.model_25_fe_conv1);
    Yumodel_25_layer1 = Matrix2List(feature_fe_RDM.model_25_fe_layer1);
    Yumodel_25_layer2 = Matrix2List(feature_fe_RDM.model_25_fe_layer2);
    Yumodel_25_layer3 = Matrix2List(feature_fe_RDM.model_25_fe_layer3);
    Yumodel_25_layer4 = Matrix2List(feature_fe_RDM.model_25_fe_layer4);
    Yumodel_25_last = Matrix2List(feature_fe_RDM.model_25_fe_last);
    Yumodel_25 = Matrix2List(YuRDM.model_25);

    Yumodel_26_ts1 = Matrix2List(YuRDM.model_26_ts_1);
    Yumodel_26_ts2 = Matrix2List(YuRDM.model_26_ts_2);
    Yumodel_26_ts3 = Matrix2List(YuRDM.model_26_ts_3);
    Yumodel_26_cdp1 = Matrix2List(YuRDM.model_26_cdp_1);
    Yumodel_26_cdp2 = Matrix2List(YuRDM.model_26_cdp_2);
    Yumodel_26_cdp3 = Matrix2List(YuRDM.model_26_cdp_3);
    Yumodel_26_conv1 = Matrix2List(feature_fe_RDM.model_26_fe_conv1);
    Yumodel_26_layer1 = Matrix2List(feature_fe_RDM.model_26_fe_layer1);
    Yumodel_26_layer2 = Matrix2List(feature_fe_RDM.model_26_fe_layer2);
    Yumodel_26_layer3 = Matrix2List(feature_fe_RDM.model_26_fe_layer3);
    Yumodel_26_layer4 = Matrix2List(feature_fe_RDM.model_26_fe_layer4);
    Yumodel_26_last = Matrix2List(feature_fe_RDM.model_26_fe_last);
    Yumodel_26 = Matrix2List(YuRDM.model_26);

    Yumodel_27_ts1 = Matrix2List(YuRDM.model_27_ts_1);
    Yumodel_27_ts2 = Matrix2List(YuRDM.model_27_ts_2);
    Yumodel_27_ts3 = Matrix2List(YuRDM.model_27_ts_3);
    Yumodel_27_cdp1 = Matrix2List(YuRDM.model_27_cdp_1);
    Yumodel_27_cdp2 = Matrix2List(YuRDM.model_27_cdp_2);
    Yumodel_27_cdp3 = Matrix2List(YuRDM.model_27_cdp_3);
    Yumodel_27_conv1 = Matrix2List(feature_fe_RDM.model_27_fe_conv1);
    Yumodel_27_layer1 = Matrix2List(feature_fe_RDM.model_27_fe_layer1);
    Yumodel_27_layer2 = Matrix2List(feature_fe_RDM.model_27_fe_layer2);
    Yumodel_27_layer3 = Matrix2List(feature_fe_RDM.model_27_fe_layer3);
    Yumodel_27_layer4 = Matrix2List(feature_fe_RDM.model_27_fe_layer4);
    Yumodel_27_last = Matrix2List(feature_fe_RDM.model_27_fe_last);
    Yumodel_27 = Matrix2List(YuRDM.model_27);

    Yumodel_28_ts1 = Matrix2List(YuRDM.model_28_ts_1);
    Yumodel_28_ts2 = Matrix2List(YuRDM.model_28_ts_2);
    Yumodel_28_ts3 = Matrix2List(YuRDM.model_28_ts_3);
    Yumodel_28_cdp1 = Matrix2List(YuRDM.model_28_cdp_1);
    Yumodel_28_cdp2 = Matrix2List(YuRDM.model_28_cdp_2);
    Yumodel_28_cdp3 = Matrix2List(YuRDM.model_28_cdp_3);
    Yumodel_28_conv1 = Matrix2List(feature_fe_RDM.model_28_fe_conv1);
    Yumodel_28_layer1 = Matrix2List(feature_fe_RDM.model_28_fe_layer1);
    Yumodel_28_layer2 = Matrix2List(feature_fe_RDM.model_28_fe_layer2);
    Yumodel_28_layer3 = Matrix2List(feature_fe_RDM.model_28_fe_layer3);
    Yumodel_28_layer4 = Matrix2List(feature_fe_RDM.model_28_fe_layer4);
    Yumodel_28_last = Matrix2List(feature_fe_RDM.model_28_fe_last);
    Yumodel_28 = Matrix2List(YuRDM.model_28);

    Yumodel_29_ts1 = Matrix2List(YuRDM.model_29_ts_1);
    Yumodel_29_ts2 = Matrix2List(YuRDM.model_29_ts_2);
    Yumodel_29_ts3 = Matrix2List(YuRDM.model_29_ts_3);
    Yumodel_29_cdp1 = Matrix2List(YuRDM.model_29_cdp_1);
    Yumodel_29_cdp2 = Matrix2List(YuRDM.model_29_cdp_2);
    Yumodel_29_cdp3 = Matrix2List(YuRDM.model_29_cdp_3);
    Yumodel_29_conv1 = Matrix2List(feature_fe_RDM.model_29_fe_conv1);
    Yumodel_29_layer1 = Matrix2List(feature_fe_RDM.model_29_fe_layer1);
    Yumodel_29_layer2 = Matrix2List(feature_fe_RDM.model_29_fe_layer2);
    Yumodel_29_layer3 = Matrix2List(feature_fe_RDM.model_29_fe_layer3);
    Yumodel_29_layer4 = Matrix2List(feature_fe_RDM.model_29_fe_layer4);
    Yumodel_29_last = Matrix2List(feature_fe_RDM.model_29_fe_last);
    Yumodel_29 = Matrix2List(YuRDM.model_29);

    Yumodel_30_ts1 = Matrix2List(YuRDM.model_30_ts_1);
    Yumodel_30_ts2 = Matrix2List(YuRDM.model_30_ts_2);
    Yumodel_30_ts3 = Matrix2List(YuRDM.model_30_ts_3);
    Yumodel_30_cdp1 = Matrix2List(YuRDM.model_30_cdp_1);
    Yumodel_30_cdp2 = Matrix2List(YuRDM.model_30_cdp_2);
    Yumodel_30_cdp3 = Matrix2List(YuRDM.model_30_cdp_3);
    Yumodel_30_conv1 = Matrix2List(feature_fe_RDM.model_30_fe_conv1);
    Yumodel_30_layer1 = Matrix2List(feature_fe_RDM.model_30_fe_layer1);
    Yumodel_30_layer2 = Matrix2List(feature_fe_RDM.model_30_fe_layer2);
    Yumodel_30_layer3 = Matrix2List(feature_fe_RDM.model_30_fe_layer3);
    Yumodel_30_layer4 = Matrix2List(feature_fe_RDM.model_30_fe_layer4);
    Yumodel_30_last = Matrix2List(feature_fe_RDM.model_30_fe_last);
    Yumodel_30 = Matrix2List(YuRDM.model_30);


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
    corrT_Yumodel_21 = zeros(Header.dim);
    corrT_Yumodel_21_ts1 = zeros(Header.dim);
    corrT_Yumodel_21_ts2 = zeros(Header.dim);
    corrT_Yumodel_21_ts3 = zeros(Header.dim);
    corrT_Yumodel_21_cdp1 = zeros(Header.dim);
    corrT_Yumodel_21_cdp2 = zeros(Header.dim);
    corrT_Yumodel_21_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_21_gist = zeros(Header.dim);
    pcorrT_Yumodel_21_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_21_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_21_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_21_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_21_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_21_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_22 = zeros(Header.dim);
    corrT_Yumodel_22_ts1 = zeros(Header.dim);
    corrT_Yumodel_22_ts2 = zeros(Header.dim);
    corrT_Yumodel_22_ts3 = zeros(Header.dim);
    corrT_Yumodel_22_cdp1 = zeros(Header.dim);
    corrT_Yumodel_22_cdp2 = zeros(Header.dim);
    corrT_Yumodel_22_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_22_gist = zeros(Header.dim);
    pcorrT_Yumodel_22_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_22_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_22_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_22_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_22_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_22_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_23 = zeros(Header.dim);
    corrT_Yumodel_23_ts1 = zeros(Header.dim);
    corrT_Yumodel_23_ts2 = zeros(Header.dim);
    corrT_Yumodel_23_ts3 = zeros(Header.dim);
    corrT_Yumodel_23_cdp1 = zeros(Header.dim);
    corrT_Yumodel_23_cdp2 = zeros(Header.dim);
    corrT_Yumodel_23_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_23_gist = zeros(Header.dim);
    pcorrT_Yumodel_23_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_23_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_23_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_23_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_23_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_23_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_24 = zeros(Header.dim);
    corrT_Yumodel_24_ts1 = zeros(Header.dim);
    corrT_Yumodel_24_ts2 = zeros(Header.dim);
    corrT_Yumodel_24_ts3 = zeros(Header.dim);
    corrT_Yumodel_24_cdp1 = zeros(Header.dim);
    corrT_Yumodel_24_cdp2 = zeros(Header.dim);
    corrT_Yumodel_24_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_24_gist = zeros(Header.dim);
    pcorrT_Yumodel_24_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_24_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_24_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_24_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_24_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_24_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_25 = zeros(Header.dim);
    corrT_Yumodel_25_ts1 = zeros(Header.dim);
    corrT_Yumodel_25_ts2 = zeros(Header.dim);
    corrT_Yumodel_25_ts3 = zeros(Header.dim);
    corrT_Yumodel_25_cdp1 = zeros(Header.dim);
    corrT_Yumodel_25_cdp2 = zeros(Header.dim);
    corrT_Yumodel_25_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_25_gist = zeros(Header.dim);
    pcorrT_Yumodel_25_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_25_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_25_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_25_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_25_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_25_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_26 = zeros(Header.dim);
    corrT_Yumodel_26_ts1 = zeros(Header.dim);
    corrT_Yumodel_26_ts2 = zeros(Header.dim);
    corrT_Yumodel_26_ts3 = zeros(Header.dim);
    corrT_Yumodel_26_cdp1 = zeros(Header.dim);
    corrT_Yumodel_26_cdp2 = zeros(Header.dim);
    corrT_Yumodel_26_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_26_gist = zeros(Header.dim);
    pcorrT_Yumodel_26_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_26_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_26_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_26_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_26_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_26_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_27 = zeros(Header.dim);
    corrT_Yumodel_27_ts1 = zeros(Header.dim);
    corrT_Yumodel_27_ts2 = zeros(Header.dim);
    corrT_Yumodel_27_ts3 = zeros(Header.dim);
    corrT_Yumodel_27_cdp1 = zeros(Header.dim);
    corrT_Yumodel_27_cdp2 = zeros(Header.dim);
    corrT_Yumodel_27_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_27_gist = zeros(Header.dim);
    pcorrT_Yumodel_27_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_27_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_27_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_27_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_27_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_27_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_28 = zeros(Header.dim);
    corrT_Yumodel_28_ts1 = zeros(Header.dim);
    corrT_Yumodel_28_ts2 = zeros(Header.dim);
    corrT_Yumodel_28_ts3 = zeros(Header.dim);
    corrT_Yumodel_28_cdp1 = zeros(Header.dim);
    corrT_Yumodel_28_cdp2 = zeros(Header.dim);
    corrT_Yumodel_28_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_28_gist = zeros(Header.dim);
    pcorrT_Yumodel_28_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_28_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_28_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_28_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_28_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_28_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_29 = zeros(Header.dim);
    corrT_Yumodel_29_ts1 = zeros(Header.dim);
    corrT_Yumodel_29_ts2 = zeros(Header.dim);
    corrT_Yumodel_29_ts3 = zeros(Header.dim);
    corrT_Yumodel_29_cdp1 = zeros(Header.dim);
    corrT_Yumodel_29_cdp2 = zeros(Header.dim);
    corrT_Yumodel_29_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_29_gist = zeros(Header.dim);
    pcorrT_Yumodel_29_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_29_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_29_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_29_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_29_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_29_cdp3_gist = zeros(Header.dim);

    corrT_Yumodel_30 = zeros(Header.dim);
    corrT_Yumodel_30_ts1 = zeros(Header.dim);
    corrT_Yumodel_30_ts2 = zeros(Header.dim);
    corrT_Yumodel_30_ts3 = zeros(Header.dim);
    corrT_Yumodel_30_cdp1 = zeros(Header.dim);
    corrT_Yumodel_30_cdp2 = zeros(Header.dim);
    corrT_Yumodel_30_cdp3 = zeros(Header.dim);
    pcorrT_Yumodel_30_gist = zeros(Header.dim);
    pcorrT_Yumodel_30_ts1_gist = zeros(Header.dim);
    pcorrT_Yumodel_30_ts2_gist = zeros(Header.dim);
    pcorrT_Yumodel_30_ts3_gist = zeros(Header.dim);
    pcorrT_Yumodel_30_cdp1_gist = zeros(Header.dim);
    pcorrT_Yumodel_30_cdp2_gist = zeros(Header.dim);
    pcorrT_Yumodel_30_cdp3_gist = zeros(Header.dim);
    
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
                    corrT_Yumodel_21(x,y,z) = corr(BrainList,Yumodel_21,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_21_ts1(x,y,z) = corr(BrainList,Yumodel_21_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_21_ts2(x,y,z) = corr(BrainList,Yumodel_21_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_21_ts3(x,y,z) = corr(BrainList,Yumodel_21_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_21_cdp1(x,y,z) = corr(BrainList,Yumodel_21_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_21_cdp2(x,y,z) = corr(BrainList,Yumodel_21_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_21_cdp3(x,y,z) = corr(BrainList,Yumodel_21_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_21_gist(x,y,z) = partialcorr(BrainList,Yumodel_21,Yumodel_21_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_21_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_21_ts1,Yumodel_21_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_21_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_21_ts2,Yumodel_21_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_21_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_21_ts3,Yumodel_21_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_21_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_21_cdp1,Yumodel_21_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_21_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_21_cdp2,Yumodel_21_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_21_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_21_cdp3,Yumodel_21_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_22(x,y,z) = corr(BrainList,Yumodel_22,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_22_ts1(x,y,z) = corr(BrainList,Yumodel_22_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_22_ts2(x,y,z) = corr(BrainList,Yumodel_22_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_22_ts3(x,y,z) = corr(BrainList,Yumodel_22_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_22_cdp1(x,y,z) = corr(BrainList,Yumodel_22_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_22_cdp2(x,y,z) = corr(BrainList,Yumodel_22_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_22_cdp3(x,y,z) = corr(BrainList,Yumodel_22_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_22_gist(x,y,z) = partialcorr(BrainList,Yumodel_22,Yumodel_22_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_22_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_22_ts1,Yumodel_22_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_22_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_22_ts2,Yumodel_22_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_22_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_22_ts3,Yumodel_22_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_22_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_22_cdp1,Yumodel_22_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_22_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_22_cdp2,Yumodel_22_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_22_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_22_cdp3,Yumodel_22_last,'type','Spearman','rows','all','tail','both');
                    
                    corrT_Yumodel_23(x,y,z) = corr(BrainList,Yumodel_23,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_23_ts1(x,y,z) = corr(BrainList,Yumodel_23_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_23_ts2(x,y,z) = corr(BrainList,Yumodel_23_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_23_ts3(x,y,z) = corr(BrainList,Yumodel_23_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_23_cdp1(x,y,z) = corr(BrainList,Yumodel_23_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_23_cdp2(x,y,z) = corr(BrainList,Yumodel_23_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_23_cdp3(x,y,z) = corr(BrainList,Yumodel_23_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_23_gist(x,y,z) = partialcorr(BrainList,Yumodel_23,Yumodel_23_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_23_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_23_ts1,Yumodel_23_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_23_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_23_ts2,Yumodel_23_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_23_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_23_ts3,Yumodel_23_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_23_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_23_cdp1,Yumodel_23_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_23_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_23_cdp2,Yumodel_23_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_23_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_23_cdp3,Yumodel_23_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_24(x,y,z) = corr(BrainList,Yumodel_24,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_24_ts1(x,y,z) = corr(BrainList,Yumodel_24_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_24_ts2(x,y,z) = corr(BrainList,Yumodel_24_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_24_ts3(x,y,z) = corr(BrainList,Yumodel_24_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_24_cdp1(x,y,z) = corr(BrainList,Yumodel_24_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_24_cdp2(x,y,z) = corr(BrainList,Yumodel_24_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_24_cdp3(x,y,z) = corr(BrainList,Yumodel_24_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_24_gist(x,y,z) = partialcorr(BrainList,Yumodel_24,Yumodel_24_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_24_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_24_ts1,Yumodel_24_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_24_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_24_ts2,Yumodel_24_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_24_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_24_ts3,Yumodel_24_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_24_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_24_cdp1,Yumodel_24_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_24_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_24_cdp2,Yumodel_24_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_24_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_24_cdp3,Yumodel_24_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_25(x,y,z) = corr(BrainList,Yumodel_25,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_25_ts1(x,y,z) = corr(BrainList,Yumodel_25_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_25_ts2(x,y,z) = corr(BrainList,Yumodel_25_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_25_ts3(x,y,z) = corr(BrainList,Yumodel_25_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_25_cdp1(x,y,z) = corr(BrainList,Yumodel_25_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_25_cdp2(x,y,z) = corr(BrainList,Yumodel_25_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_25_cdp3(x,y,z) = corr(BrainList,Yumodel_25_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_25_gist(x,y,z) = partialcorr(BrainList,Yumodel_25,Yumodel_25_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_25_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_25_ts1,Yumodel_25_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_25_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_25_ts2,Yumodel_25_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_25_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_25_ts3,Yumodel_25_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_25_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_25_cdp1,Yumodel_25_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_25_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_25_cdp2,Yumodel_25_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_25_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_25_cdp3,Yumodel_25_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_26(x,y,z) = corr(BrainList,Yumodel_26,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_26_ts1(x,y,z) = corr(BrainList,Yumodel_26_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_26_ts2(x,y,z) = corr(BrainList,Yumodel_26_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_26_ts3(x,y,z) = corr(BrainList,Yumodel_26_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_26_cdp1(x,y,z) = corr(BrainList,Yumodel_26_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_26_cdp2(x,y,z) = corr(BrainList,Yumodel_26_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_26_cdp3(x,y,z) = corr(BrainList,Yumodel_26_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_26_gist(x,y,z) = partialcorr(BrainList,Yumodel_26,Yumodel_26_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_26_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_26_ts1,Yumodel_26_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_26_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_26_ts2,Yumodel_26_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_26_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_26_ts3,Yumodel_26_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_26_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_26_cdp1,Yumodel_26_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_26_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_26_cdp2,Yumodel_26_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_26_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_26_cdp3,Yumodel_26_last,'type','Spearman','rows','all','tail','both');
                    
                    corrT_Yumodel_27(x,y,z) = corr(BrainList,Yumodel_27,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_27_ts1(x,y,z) = corr(BrainList,Yumodel_27_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_27_ts2(x,y,z) = corr(BrainList,Yumodel_27_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_27_ts3(x,y,z) = corr(BrainList,Yumodel_27_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_27_cdp1(x,y,z) = corr(BrainList,Yumodel_27_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_27_cdp2(x,y,z) = corr(BrainList,Yumodel_27_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_27_cdp3(x,y,z) = corr(BrainList,Yumodel_27_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_27_gist(x,y,z) = partialcorr(BrainList,Yumodel_27,Yumodel_27_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_27_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_27_ts1,Yumodel_27_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_27_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_27_ts2,Yumodel_27_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_27_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_27_ts3,Yumodel_27_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_27_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_27_cdp1,Yumodel_27_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_27_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_27_cdp2,Yumodel_27_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_27_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_27_cdp3,Yumodel_27_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_28(x,y,z) = corr(BrainList,Yumodel_28,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_28_ts1(x,y,z) = corr(BrainList,Yumodel_28_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_28_ts2(x,y,z) = corr(BrainList,Yumodel_28_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_28_ts3(x,y,z) = corr(BrainList,Yumodel_28_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_28_cdp1(x,y,z) = corr(BrainList,Yumodel_28_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_28_cdp2(x,y,z) = corr(BrainList,Yumodel_28_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_28_cdp3(x,y,z) = corr(BrainList,Yumodel_28_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_28_gist(x,y,z) = partialcorr(BrainList,Yumodel_28,Yumodel_28_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_28_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_28_ts1,Yumodel_28_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_28_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_28_ts2,Yumodel_28_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_28_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_28_ts3,Yumodel_28_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_28_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_28_cdp1,Yumodel_28_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_28_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_28_cdp2,Yumodel_28_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_28_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_28_cdp3,Yumodel_28_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_29(x,y,z) = corr(BrainList,Yumodel_29,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_29_ts1(x,y,z) = corr(BrainList,Yumodel_29_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_29_ts2(x,y,z) = corr(BrainList,Yumodel_29_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_29_ts3(x,y,z) = corr(BrainList,Yumodel_29_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_29_cdp1(x,y,z) = corr(BrainList,Yumodel_29_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_29_cdp2(x,y,z) = corr(BrainList,Yumodel_29_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_29_cdp3(x,y,z) = corr(BrainList,Yumodel_29_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_29_gist(x,y,z) = partialcorr(BrainList,Yumodel_29,Yumodel_29_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_29_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_29_ts1,Yumodel_29_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_29_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_29_ts2,Yumodel_29_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_29_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_29_ts3,Yumodel_29_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_29_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_29_cdp1,Yumodel_29_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_29_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_29_cdp2,Yumodel_29_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_29_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_29_cdp3,Yumodel_29_last,'type','Spearman','rows','all','tail','both');

                    corrT_Yumodel_30(x,y,z) = corr(BrainList,Yumodel_30,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_30_ts1(x,y,z) = corr(BrainList,Yumodel_30_ts1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_30_ts2(x,y,z) = corr(BrainList,Yumodel_30_ts2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_30_ts3(x,y,z) = corr(BrainList,Yumodel_30_ts3,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_30_cdp1(x,y,z) = corr(BrainList,Yumodel_30_cdp1,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_30_cdp2(x,y,z) = corr(BrainList,Yumodel_30_cdp2,'type','Spearman','rows','all','tail','both');
                    corrT_Yumodel_30_cdp3(x,y,z) = corr(BrainList,Yumodel_30_cdp3,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_30_gist(x,y,z) = partialcorr(BrainList,Yumodel_30,Yumodel_30_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_30_ts1_gist(x,y,z) = partialcorr(BrainList,Yumodel_30_ts1,Yumodel_30_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_30_ts2_gist(x,y,z) = partialcorr(BrainList,Yumodel_30_ts2,Yumodel_30_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_30_ts3_gist(x,y,z) = partialcorr(BrainList,Yumodel_30_ts3,Yumodel_30_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_30_cdp1_gist(x,y,z) = partialcorr(BrainList,Yumodel_30_cdp1,Yumodel_30_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_30_cdp2_gist(x,y,z) = partialcorr(BrainList,Yumodel_30_cdp2,Yumodel_30_last,'type','Spearman','rows','all','tail','both');
                    pcorrT_Yumodel_30_cdp3_gist(x,y,z) = partialcorr(BrainList,Yumodel_30_cdp3,Yumodel_30_last,'type','Spearman','rows','all','tail','both');

                    PrintN = PrintN + 1;
                    
                    display([Subject(subj).name,': ',num2str(PrintN),'/',num2str(PrintSum)]);
    
                end
            end
        end
    end
    
    % write_img_file
    rest_writefile(corrT_Yumodel_21,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_21.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_21_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_21_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_21_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_21_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_21_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_21_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_21_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_21_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_21_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_21_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_21_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_21_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_21_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_21_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_21_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_21_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_21_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_21_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_21_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_21_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_21_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_21_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_21_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_21_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_21_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_21_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_22,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_22.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_22_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_22_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_22_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_22_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_22_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_22_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_22_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_22_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_22_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_22_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_22_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_22_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_22_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_22_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_22_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_22_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_22_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_22_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_22_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_22_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_22_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_22_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_22_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_22_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_22_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_22_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');
    
    rest_writefile(corrT_Yumodel_23,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_23.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_23_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_23_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_23_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_23_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_23_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_23_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_23_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_23_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_23_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_23_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_23_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_23_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_23_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_23_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_23_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_23_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_23_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_23_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_23_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_23_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_23_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_23_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_23_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_23_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_23_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_23_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_24,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_24.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_24_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_24_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_24_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_24_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_24_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_24_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_24_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_24_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_24_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_24_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_24_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_24_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_24_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_24_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_24_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_24_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_24_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_24_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_24_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_24_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_24_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_24_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_24_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_24_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_24_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_24_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_25,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_25.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_25_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_25_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_25_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_25_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_25_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_25_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_25_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_25_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_25_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_25_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_25_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_25_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_25_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_25_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_25_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_25_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_25_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_25_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_25_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_25_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_25_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_25_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_25_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_25_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_25_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_25_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_26,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_26.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_26_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_26_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_26_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_26_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_26_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_26_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_26_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_26_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_26_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_26_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_26_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_26_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_26_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_26_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_26_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_26_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_26_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_26_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_26_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_26_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_26_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_26_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_26_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_26_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_26_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_26_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');
    
    rest_writefile(corrT_Yumodel_27,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_27.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_27_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_27_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_27_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_27_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_27_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_27_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_27_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_27_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_27_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_27_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_27_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_27_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_27_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_27_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_27_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_27_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_27_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_27_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_27_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_27_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_27_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_27_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_27_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_27_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_27_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_27_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_28,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_28.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_28_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_28_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_28_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_28_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_28_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_28_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_28_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_28_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_28_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_28_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_28_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_28_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_28_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_28_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_28_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_28_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_28_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_28_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_28_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_28_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_28_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_28_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_28_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_28_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_28_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_28_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_29,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_29.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_29_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_29_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_29_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_29_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_29_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_29_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_29_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_29_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_29_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_29_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_29_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_29_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_29_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_29_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_29_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_29_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_29_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_29_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_29_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_29_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_29_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_29_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_29_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_29_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_29_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_29_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

    rest_writefile(corrT_Yumodel_30,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_30.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_30_ts1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_30_ts1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_30_ts2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_30_ts2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_30_ts3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_30_ts3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_30_cdp1,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_30_cdp1.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_30_cdp2,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_30_cdp2.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(corrT_Yumodel_30_cdp3,[OutputPath filesep Subject(subj).name filesep 'corrT_Yumodel_30_cdp3.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_30_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_30_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_30_ts1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_30_ts1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_30_ts2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_30_ts2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_30_ts3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_30_ts3_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_30_cdp1_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_30_cdp1_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_30_cdp2_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_30_cdp2_gist.nii'],Header.dim,VoxDim,Header,'double');
    rest_writefile(pcorrT_Yumodel_30_cdp3_gist,[OutputPath filesep Subject(subj).name filesep 'pcorrT_Yumodel_30_cdp3_gist.nii'],Header.dim,VoxDim,Header,'double');

end
