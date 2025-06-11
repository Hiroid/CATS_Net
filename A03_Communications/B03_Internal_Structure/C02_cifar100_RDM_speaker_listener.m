clear all;
clc;

load('vik.mat');

rdm_type = 'cosine'; % 'cosine' 'pearson'

num_listeners = 100;
all_correlations = zeros(num_listeners, 1);

% Load speaker data once outside the loop
ct_speaker_full = load('cifar100_ss20_ni1e-1_ychen_trail1.mat');
ct_speaker_data_full = struct2array(ct_speaker_full);

for listener_idx = 1:num_listeners
    fprintf('Processing listener %d...\n', listener_idx);

    % Prepare speaker data excluding the current listener index
    ct_speaker_data = ct_speaker_data_full([1:listener_idx-1, listener_idx+1:end], :);

    if strcmp(rdm_type, 'pearson')
        ct_speaker_RDM = ones(99,99) - corr(ct_speaker_data','type','Pearson','rows','all','tail','both');
    else
        ct_speaker_dist_orig = pdist(ct_speaker_data, 'cosine');
        ct_speaker_RDM = squareform(ct_speaker_dist_orig);
    end

    % Load listener data
    listener_name = fullfile('..', 'B02_Communication_Game', 'Symbol_and_Model_of_Listener', 'contexts', sprintf('context_id_%d_e_1999.mat', listener_idx-1));
    ct_listener = load(listener_name);
    ct_listener_data_full = struct2array(ct_listener);
    % Prepare listener data excluding the current listener index (assuming it was trained with all 100 speakers)
    ct_listener_data = ct_listener_data_full([1:listener_idx-1, listener_idx+1:end], :);

    if strcmp(rdm_type, 'pearson')
        ct_listener_RDM = ones(99,99) - corr(ct_listener_data','type','Pearson','rows','all','tail','both');
    else
        ct_listener_dist_orig = pdist(ct_listener_data, 'cosine');
        ct_listener_RDM = squareform(ct_listener_dist_orig);
    end

    % Calculate correlation
    [corr_speaker_listener, ~] = corr(Matrix2List(ct_speaker_RDM), Matrix2List(ct_listener_RDM),'type','Spearman','rows','all','tail','both');

    all_correlations(listener_idx) = corr_speaker_listener;

    %fprintf('corr_speaker_listener %0.2f\n', corr_speaker_listener);
end

% figure;
% min_val = min([min(ct_speaker_RDM(:)), min(ct_listener_RDM(:))]);
% max_val = max([max(ct_speaker_RDM(:)), max(ct_listener_RDM(:))]);

% subplot(1, 2, 1);
% imagesc(ct_speaker_RDM);
% colorbar;
% axis square;
% axis off;
% caxis([min_val max_val]); 

% subplot(1, 2, 2);
% imagesc(ct_listener_RDM);
% colorbar;
% axis square;
% axis off;
% caxis([min_val max_val]); 

% colormap(vik);

% % Save results to CSV
% output_filename = 'speaker_listener_correlations.csv';
% writematrix(all_correlations, output_filename);
% fprintf('Saved %d correlation values to %s\n', num_listeners, output_filename);


% --- One-sample t-test ---
sim_threshold = 0.35;
[h, p_ttest, ci, stats] = ttest(all_correlations, sim_threshold, 'Tail', 'right');

fprintf('\n--- One-Sample t-test for RDM Correlations ---\n');
fprintf('Hypothesis: Mean RDM Correlation > %0.2f\n', sim_threshold);
fprintf('t-statistic: %.4f\n', stats.tstat);
fprintf('Degrees of Freedom: %d\n', stats.df);
fprintf('P-value: %.4f\n', p_ttest);
fprintf('95%% Confidence Interval (lower bound): %.4f\n', ci(1));

% Calculate Cohen's d
cohens_d = (mean(all_correlations) - sim_threshold) / stats.sd;
fprintf('Cohen''s d: %.4f\n', cohens_d);

if h == 1
    fprintf('Conclusion: The mean RDM correlation is significantly greater than %0.2f (p = %.4f).\n', sim_threshold, p_ttest);
else
    fprintf('Conclusion: There is not enough evidence to conclude that the mean RDM correlation is significantly greater than %0.2f (p = %.4f).\n', sim_threshold, p_ttest);
end