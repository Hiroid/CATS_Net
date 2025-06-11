clear all;
clc;

method_linkage = 'average';
link_crit = 'min';
listener_idx = 0; % from 1 to 100, 0 indicate the speaker

class_label = load("meta.mat");


if listener_idx == 0
    fine_label_names = class_label.fine_label_names;
    ct = load('cifar100_ss20_ni1e-1_ychen_trail1.mat');
    ct_ychen_data = struct2array(ct);
else
    fine_label_names = class_label.fine_label_names([1:listener_idx-1, listener_idx+1:end]);
    listener_name = fullfile('..', 'B02_Communication_Game', 'Symbol_and_Model_of_Listener', 'contexts', sprintf('context_id_%d_e_1999.mat', listener_idx-1));
    ct = load(listener_name);
    ct_ychen_data = struct2array(ct);
    ct_ychen_data = ct_ychen_data([1:listener_idx-1, listener_idx+1:end], :);
end

ct_link = linkage(ct_ychen_data, method_linkage, 'cosine'); 
% return a matrix of (n-1) x 3, each line is a new cluster, numbered from n+1
ct_dist_orig = pdist(ct_ychen_data, 'cosine');
ct_dist_squre = squareform(ct_dist_orig);

amat = zeros(size(ct_ychen_data, 1));
ct_link_size = size(ct_link);

for i = 1 : ct_link_size(1)
    % travel throughout all the cluster link
    [cluster1_id, cluster2_id] = cluster_search(ct_link(i, :), ct_link);
    % find the minimum
    if strcmp(link_crit, 'min')
        cluster_link = ct_dist_squre(cluster1_id, cluster2_id); % selected cluster
        [link_value, link_id_lin] = min(cluster_link, [], 'all','linear'); % minimum value and index in the selected cluster, linear index
        [link_id1, link_id2] = ind2sub(size(cluster_link), link_id_lin); % return the min index in the selected cluster
        amat(cluster1_id(link_id1), cluster2_id(link_id2)) = 1 - ct_dist_squre(cluster1_id(link_id1), cluster2_id(link_id2));

    end
end

if listener_idx == 0
    cluster_info = cluster(ct_link,'cutoff', 0.3785,'Criterion','distance'); % get the label info of cluster
else
    cluster_info = cluster(ct_link,'cutoff', 0.31,'Criterion','distance'); % get the label info of cluster
end

fileID_n = fopen('Nods_information.csv','w');
fprintf(fileID_n, 'Id, Label, Modularity Class\n');
for i=1:size(amat, 1)
    fprintf(fileID_n,['%d, ', fine_label_names{i}, ', ', num2str(cluster_info(i)),'\n'], i);
end
fclose(fileID_n);

fileID_e = fopen('Edge_information.csv','w');
fprintf(fileID_e,'Source, Target, Type, Id, Label, Weight\n');
edge_id = 1;
for i=1:size(amat,1)
    for j = 1:size(amat,2)
        if amat(i,j) ~= 0
            fprintf(fileID_e, '%d, %d, Undirected, %d, ,%2.3f\n',i,j, edge_id, amat(i,j));
            edge_id = edge_id + 1;
        end
    end
end
fclose(fileID_e);

dendrogram(ct_link,0)
