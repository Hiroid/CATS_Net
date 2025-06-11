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

RootFolder='A1_Searchlight';
RawResultFolder = [RootFolder filesep 'A0_all_results_eval_TransE_41to50'];
SubFolder = dir([RawResultFolder filesep 'SUB*']);
nSub = length(SubFolder);

for subj=1:nSub
    rfile=dir([SubFolder(subj).folder filesep SubFolder(subj).name filesep '*.nii']);
    nfile=length(rfile);
    targetFolder = [RootFolder filesep 'A1_fishZ_all_eval_TransE_41to50' filesep  SubFolder(subj).name];
    mkdir(targetFolder);
    for i=1:nfile
        [data,vox,head]=rest_readfile([rfile(i).folder filesep rfile(i).name]);
        data_new=0.5*log((1+data)./(1-data));
        rest_writefile(data_new,[targetFolder filesep 'fishZ_' rfile(i).name],size(data),vox,head,'double');
    end
end