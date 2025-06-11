% Read the data from the CSV file
data = readtable('../../Results/single_ct_acc.csv');

% Extract the 'best_acc_2' column
acc_data = data.best_acc_2;

% Define the value to test against
mu0 = 0.5;

% Perform a one-sample t-test (right-tailed)
% H0: mean(acc_data) = mu0
% H1: mean(acc_data) > mu0
[h, p, ci, stats] = ttest(acc_data, mu0, 'Tail', 'right');

% Display the results
fprintf('One-sample t-test results for best_acc_2 > %.2f:\n', mu0);
fprintf('--------------------------------------------------\n');
fprintf('H (Hypothesis rejected? 1=yes, 0=no): %d\n', h);
fprintf('P-value: %.4f\n', p);
fprintf('Confidence Interval (CI): [%.4f, %.4f]\n', ci(1), ci(2));
fprintf('T-statistic: %.4f\n', stats.tstat);
fprintf('Degrees of Freedom (df): %d\n', stats.df);

% Calculate Cohen's d
cohens_d = (mean(acc_data) - mu0) / stats.sd;
fprintf('Cohen''s d: %.4f\n', cohens_d);

% Interpret the results
if h == 1
    fprintf('\nThe null hypothesis (mean <= %.2f) is rejected at the default 5%% significance level.\n', mu0);
    fprintf('There is significant evidence to suggest that the mean accuracy is greater than %.2f.\n', mu0);
else
    fprintf('\nThe null hypothesis (mean <= %.2f) is not rejected at the default 5%% significance level.\n', mu0);
    fprintf('There is not significant evidence to suggest that the mean accuracy is greater than %.2f.\n', mu0);
end
