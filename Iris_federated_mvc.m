close all; clear; clc;

%% SECTION 1: Data Loading
fprintf('=== HK-MVFC DEMONSTRATION ===\n');

rng(18);
if exist('fisheriris.mat', 'file')
    fprintf('Loading existing Fisher Iris dataset...\n');
    load('fisheriris.mat');
    % fprintf('✓ Loaded federated dataset with %d clients\n', P);
else
    fprintf('Multi-view dataset not found...\n');
end

X=meas;
y=categorical(species);
label=grp2idx(y);
clear meas y species 

%% SECTION 2: 2-view Iris data scene
% Via the pairwise combinations of attributes in Iris:
% View 1 : attributes 1&3
% View 2 : attributes 2&4

points{1} = X(:, [1, 3]);
points{2} = X(:, [2, 4]);

clear X 

% Dataset verification
total_samples = length(label);
num_views = length(points);
num_clusters = max(label);

fprintf('Dataset Summary:\n');
fprintf('- Total samples: %d\n', total_samples);
fprintf('- Number of views: %d\n', num_views);
fprintf('- Number of clusters: %d\n', num_clusters);
fprintf('- View 1 dimensions: %d\n', size(points{1}, 2));
fprintf('- View 2 dimensions: %d\n', size(points{2}, 2));

%% Verify cluster distribution in original dataset
fprintf('\nOriginal Cluster Distribution:\n');
for k = 1:num_clusters
    cluster_count = sum(label == k);
    fprintf('- Cluster %d: %d samples (%.1f%%)\n', k, cluster_count, 100*cluster_count/total_samples);
end

%% Federated Partitioning Strategy (Stratified Sampling)
% Based on LaTeX guidelines: Client 1 gets 90 samples, Client 2 gets 60 samples
% Ensure both clients have access to all four cluster types

client1_size = 80;  % 85% of data
client2_size = 70;  % 15% of data

fprintf('\n=== STRATIFIED SAMPLING APPROACH ===\n');
fprintf('Target distribution:\n');
fprintf('- Client 1: %d samples (%.1f%%)\n', client1_size, 100*client1_size/total_samples);
fprintf('- Client 2: %d samples (%.1f%%)\n', client2_size, 100*client2_size/total_samples);

% Calculate samples per cluster for each client (proportional allocation)
samples_per_cluster = total_samples / num_clusters;  % 2500 samples per cluster
client1_per_cluster = round(client1_size / num_clusters);  % ~2125 per cluster for Client 1
client2_per_cluster = round(client2_size / num_clusters);  % ~375 per cluster for Client 2

% Adjust to ensure exact totals
adjustment_needed = client1_size - (client1_per_cluster * num_clusters);
if adjustment_needed ~= 0
    client1_per_cluster = client1_per_cluster + adjustment_needed / num_clusters;
end

fprintf('\nStratified allocation per cluster:\n');
fprintf('- Client 1: ~%d samples per cluster\n', round(client1_per_cluster));
fprintf('- Client 2: ~%d samples per cluster\n', round(client2_per_cluster));

%% Perform stratified sampling
rng(42); % Set seed for reproducibility

client1_indices = [];
client2_indices = [];

% For each cluster, randomly sample the required number of points
for k = 1:num_clusters
    % Get indices of all samples belonging to cluster k
    cluster_indices = find(label == k);
    cluster_size = length(cluster_indices);
    
    % Calculate how many samples each client gets from this cluster
    client1_cluster_samples = round(client1_size * cluster_size / total_samples);
    client2_cluster_samples = cluster_size - client1_cluster_samples;
    
    % Randomly permute cluster indices
    perm_indices = cluster_indices(randperm(cluster_size));
    
    % Assign to clients
    client1_cluster_indices = perm_indices(1:client1_cluster_samples);
    client2_cluster_indices = perm_indices(client1_cluster_samples+1:end);
    
    % Accumulate indices
    client1_indices = [client1_indices; client1_cluster_indices];
    client2_indices = [client2_indices; client2_cluster_indices];
    
    fprintf('Cluster %d: Client 1 gets %d samples, Client 2 gets %d samples\n', ...
            k, length(client1_cluster_indices), length(client2_cluster_indices));
end

%% Final adjustment to ensure exact sample counts
current_client1_size = length(client1_indices);
current_client2_size = length(client2_indices);

fprintf('\nBefore adjustment:\n');
fprintf('- Client 1: %d samples\n', current_client1_size);
fprintf('- Client 2: %d samples\n', current_client2_size);

% Adjust if necessary (move samples between clients)
if current_client1_size > client1_size
    % Move excess samples from client 1 to client 2
    excess = current_client1_size - client1_size;
    move_indices = client1_indices(end-excess+1:end);
    client1_indices = client1_indices(1:end-excess);
    client2_indices = [client2_indices; move_indices];
elseif current_client1_size < client1_size
    % Move samples from client 2 to client 1
    deficit = client1_size - current_client1_size;
    move_indices = client2_indices(1:deficit);
    client2_indices = client2_indices(deficit+1:end);
    client1_indices = [client1_indices; move_indices];
end

%% Create federated datasets
fprintf('\n=== CREATING FEDERATED DATASETS ===\n');

% Client 1 data
X_client1 = cell(num_views, 1);
for v = 1:num_views
    X_client1{v} = points{v}(client1_indices, :);
end
label_client1 = label(client1_indices);

% Client 2 data  
X_client2 = cell(num_views, 1);
for v = 1:num_views
    X_client2{v} = points{v}(client2_indices, :);
end
label_client2 = label(client2_indices);

%% Verification of federated partition
fprintf('Final partition verification:\n');
fprintf('- Client 1: %d samples\n', length(client1_indices));
fprintf('- Client 2: %d samples\n', length(client2_indices));
fprintf('- Total: %d samples\n', length(client1_indices) + length(client2_indices));

% Verify cluster distribution for each client
fprintf('\nClient 1 cluster distribution:\n');
for k = 1:num_clusters
    count = sum(label_client1 == k);
    fprintf('- Cluster %d: %d samples (%.1f%%)\n', k, count, 100*count/length(label_client1));
end

fprintf('\nClient 2 cluster distribution:\n');
for k = 1:num_clusters
    count = sum(label_client2 == k);
    fprintf('- Cluster %d: %d samples (%.1f%%)\n', k, count, 100*count/length(label_client2));
end

% Check that both clients have all cluster types
client1_clusters = unique(label_client1);
client2_clusters = unique(label_client2);

fprintf('\nCluster type verification:\n');
fprintf('- Client 1 has clusters: [%s]\n', num2str(client1_clusters'));
fprintf('- Client 2 has clusters: [%s]\n', num2str(client2_clusters'));

if length(client1_clusters) == num_clusters && length(client2_clusters) == num_clusters
    fprintf(' SUCCESS: Both clients have access to all four cluster types\n');
else
    fprintf(' ERROR: Not all clients have access to all cluster types\n');
end

%% Save federated datasets
fprintf('\n=== SAVING FEDERATED DATASETS ===\n');

% Prepare data structures for federated learning
P = 2;  % Number of clients
X_clients = cell(P, 1);
X_clients{1} = X_client1;
X_clients{2} = X_client2;

% Save individual client data
save('client1_iris_data.mat', 'X_client1', 'label_client1', 'client1_indices');
save('client2_iris_data.mat', 'X_client2', 'label_client2', 'client2_indices');

% Save combined federated structure
save('federated_iris_data.mat', 'X_clients', 'P', 'points', 'label', ...
     'client1_indices', 'client2_indices', 'X_client1', 'X_client2', ...
     'label_client1', 'label_client2');

fprintf(' Saved federated datasets:\n');
fprintf('  - client1_iris_data.mat (Client 1 data)\n');
fprintf('  - client2_iris_data.mat (Client 2 data)\n');
fprintf('  - federated_iris_data.mat (Combined federated structure)\n');

%% Summary Statistics
fprintf('\n=== SUMMARY STATISTICS ===\n');
fprintf('Federated partitioning completed successfully!\n');
fprintf('- Total samples: %d\n', total_samples);
fprintf('- Client 1: %d samples (%.1f%%)\n', length(client1_indices), 100*length(client1_indices)/total_samples);
fprintf('- Client 2: %d samples (%.1f%%)\n', length(client2_indices), 100*length(client2_indices)/total_samples);
fprintf('- Both clients have access to all %d cluster types\n', num_clusters);
fprintf('- Stratified sampling maintains cluster representation\n');
fprintf('- Ready for FedHK-MVFC algorithm testing\n');
fprintf('========================================\n');

%% Clean up workspace (keep essential variables)
clear k cluster_indices cluster_size client1_cluster_samples client2_cluster_samples;
clear perm_indices client1_cluster_indices client2_cluster_indices;
clear current_client1_size current_client2_size excess deficit move_indices;
clear cluster_mask colors markers v count;
clear adjustment_needed samples_per_cluster client1_per_cluster client2_per_cluster;
clear cluster_count num_views num_clusters client1_clusters client2_clusters;




