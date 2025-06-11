clear all;
clc;

% % Add software, add DPABI before spm
if ismac
    addpath(genpath(fullfile('..', 'DPABI_V8.1_240101')));
    addpath(genpath(fullfile('..', 'REST_V1.8_130615')));
    addpath(fullfile('..', 'spm'));
elseif ispc
    addpath(genpath(fullfile('..', 'DPABI_V8.1_240101')));
    addpath(genpath(fullfile('..', 'REST_V1.8_130615')));
    addpath(fullfile('..', 'spm'));
elseif isunix
    addpath(genpath('/data0/user/lxguo/Downloads/DPABI_V8.1_240101'));
    addpath(genpath('/data0/user/lxguo/Downloads/REST_V1.8_130615'));
    addpath('/data0/user/lxguo/Downloads/spm');
end

%% Smooth 
RootFolder='A1_Searchlight';
Subject = dir([RootFolder filesep 'A1_fishZ_all_eval_TransE_21to30' filesep 'SUB*']);
nSub = length(Subject);

BatchFolder = 'lib';
for s=1:nSub
    load([BatchFolder filesep 'Smooth_Batch.mat']);
    RSAfiles = dir([Subject(s).folder filesep  Subject(s).name filesep 'fishZ_*.nii']);
    filecell=cell(length(RSAfiles),1);
    for i = 1 : length(RSAfiles)
        filecell{i}=[Subject(s).folder filesep  Subject(s).name filesep RSAfiles(i).name];
    end
    matlabbatch{1}.spm.spatial.smooth.data=filecell;
    mkdir([RootFolder filesep 'matlabbatch_eval_TransE_21to30']);
    save([RootFolder filesep 'matlabbatch_eval_TransE_21to30' filesep 's_' Subject(s).name '.mat'],'matlabbatch');
    spm_jobman('run', matlabbatch);
    clear matlabbatch
end


%% move the Smooth files
clear all;

RootFolder='A1_Searchlight';
Source=[RootFolder filesep 'A1_fishZ_all_eval_TransE_21to30'];
Subject=dir([Source filesep 'SUB*']);
nSub=length(Subject);

Imgs=dir([Source filesep Subject(1).name filesep 'sfishZ_*.nii']);

for i=1:length(Imgs)
    mkdir([RootFolder filesep 'A2_Smooth_fishZ_all_eval_TransE_21to30' filesep Imgs(i).name(1:end-4)]);
    for subj=1:nSub
        if exist([Source filesep Subject(subj).name filesep Imgs(i).name],'file')
            copyfile([Source filesep Subject(subj).name filesep Imgs(i).name], [RootFolder filesep 'A2_Smooth_fishZ_all_eval_TransE_21to30' filesep Imgs(i).name(1:end-4) filesep Imgs(i).name(1:end-4) '_' Subject(subj).name '.nii']);
        end
    end
end

