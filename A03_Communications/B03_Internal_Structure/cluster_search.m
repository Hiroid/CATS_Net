function [cluster1_id,cluster2_id] = cluster_search(bifur_node, linkage_mat)
    cluster1_id = [];
    cluster2_id = [];
    cluster1_node = bifur_node(1);
    cluster2_node = bifur_node(2);
    num_nodes = size(linkage_mat, 1) + 1;
    if  cluster1_node <= num_nodes
        cluster1_id = [cluster1_id, cluster1_node];
    else
        cluster1_id = [cluster1_id, leaf_search(cluster1_node - num_nodes, linkage_mat)];
    end
    
    if  cluster2_node <= num_nodes
        cluster2_id = [cluster2_id, cluster2_node];
    else
        cluster2_id = [cluster2_id, leaf_search(cluster2_node - num_nodes, linkage_mat)];
    end

end


function leaf_id = leaf_search(node_id, linkage_mat)
    num_nodes = size(linkage_mat, 1) + 1;
    % recursive search
    if linkage_mat(node_id, 1) <= num_nodes && linkage_mat(node_id,2) <= num_nodes
        leaf_id = linkage_mat(node_id, [1, 2]);

    elseif linkage_mat(node_id, 1) > num_nodes && linkage_mat(node_id,2) <= num_nodes
        sub_leaf1_id = leaf_search(linkage_mat(node_id, 1) - num_nodes, linkage_mat);
        leaf_id = [sub_leaf1_id, linkage_mat(node_id,2)];

    elseif linkage_mat(node_id, 1) <= num_nodes && linkage_mat(node_id,2) > num_nodes
        sub_leaf2_id = leaf_search(linkage_mat(node_id, 2) - num_nodes, linkage_mat);
        leaf_id = [sub_leaf2_id, linkage_mat(node_id,1)];

    else
        sub_leaf1_id = leaf_search(linkage_mat(node_id, 1) - num_nodes, linkage_mat);
        sub_leaf2_id = leaf_search(linkage_mat(node_id, 2) - num_nodes, linkage_mat);
        leaf_id = [sub_leaf1_id, sub_leaf2_id];

    end

end

