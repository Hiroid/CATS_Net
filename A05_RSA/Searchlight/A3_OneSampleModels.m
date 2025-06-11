%%
clear;clc;
[mask,vox,header] = rest_readfile('E:\matlabtool\DPABI_V8.2_240510\Templates\GreyMask_02_61x73x61.img');

%% load input and average
patterns = {'Yumodel_*_concept','*Yumodel_*_cdp1', '*Yumodel_*_cdp2', '*Yumodel_*_cdp3'};
FolderName= {'CA1','CA2','CA3'};
RootPath = 'E:\YuLab\fMRI_results\Searchlight\A2_Smooth_fishZ_all_eval_TransE_21to50';
OutputBase = 'E:\YuLab\fMRI_results\Searchlight\A3_Smooth_fishZ_meanSubj';

for p=1:length(patterns)
    pattern = patterns{p};
    ConceptFolder = dir([RootPath filesep pattern]);
    ConceptFolder = ConceptFolder([ConceptFolder.isdir]);
    mkdir(OutputBase);

    for i=1:30
        rho_images = zeros([header.dim,26]);
        subjFiles = dir([RootPath filesep ConceptFolder(i).name filesep '*.nii']);
        for subj=1:26
            temp_img = rest_readfile([subjFiles(subj).folder filesep subjFiles(subj).name]);
            rho_images(:,:,:,subj) = temp_img;
        end
        mean_images = mean(rho_images,4);
        outFileName = [OutputBase filesep FolderName{p} filesep ConceptFolder(i).name '.nii'];
        mkdir([OutputBase filesep FolderName{p}]);
        rest_writefile(mean_images, outFileName, header.dim, vox, header, 'double');
    end
end



%% One sample ttests

for p=1:length(patterns)
    pattern = patterns{p};
    patternShort = FolderName{p}; 

    clear matlabbatch
    load('E:\YuLab\fMRI_results\lib/OneSampleTtest.mat');

    OutPath = ['E:\YuLab\fMRI_results\Searchlight\A5_OneSampleTtest\' patternShort];
    if ~exist(OutPath, 'dir')
        mkdir(OutPath);
    end

    ImageFolders = dir([OutputBase filesep patternShort filesep '*.nii']);
    ImageCells = cell(length(ImageFolders), 1);
    for i = 1:length(ImageFolders)
        ImageCells{i} = [ImageFolders(i).folder filesep ImageFolders(i).name];
    end

    matlabbatch{1}.spm.stats.factorial_design.dir = {OutPath};
    matlabbatch{1}.spm.stats.factorial_design.des.t1.scans = ImageCells;
    matlabbatch{1}.spm.stats.factorial_design.masking.em = {'E:\matlabtool\DPABI_V8.2_240510\Templates\GreyMask_02_61x73x61.img,1'};
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.name = 'ttest';
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.weights = 1;
    matlabbatch{3}.spm.stats.con.consess{1}.tcon.sessrep = 'none';
    matlabbatch{3}.spm.stats.con.delete = 0;

    save([OutPath filesep  'StatBatch.mat'],'matlabbatch');
    spm_jobman('run',matlabbatch);
end


