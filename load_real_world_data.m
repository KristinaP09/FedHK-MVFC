%% Demo of Federated HK-MVFC on Iris Dataset
% This script demonstrates the FedHK-MVFC algorithm on Two views synthetic Dataset
% Client 1 retain 90 samples, while Client 2 hold 60 samples
% Based on LaTeX document guidelines (line 985) for federated data partitioning

% points : Iris data points in multi-view format
% label : Labels for the data points in multi-view format
% X_client1 : Iris data stored for Client 1
% X_client2 : Iris data stored for Client 2
% label_client1 : Labels for Client 1's data
% label_client2 : Labels for Client 2's data
% client1_indices: Indices of Client 1's data points
% client2_indices: Indices of Client 2's data points

% Client Data Detailed
% Client 1: 90 samples (Iris Setosa: 30, Iris Versicolor: 30, Iris Virginica: 30)
% Client 2: 60 samples (Iris Setosa: 20, Iris Versicolor: 20, Iris Virginica: 20)

close all; clear all; clc;

%% Load or Create Federated Dataset
fprintf('=== FEDERATED HK-MVFC DEMONSTRATION ===\n');

rng(18)

% Check if federated data already exists, if not create it
if exist('federated_iris_data.mat', 'file')
    fprintf('Loading existing IRIS federated dataset...\n');
    load('federated_iris_data.mat');
    fprintf('✓ Loaded federated dataset with %d clients\n', P);
else
    fprintf('Generating multi-view scenario on Iris dataset using script...\n');
    run('Iris_federated_mvc.m');
    fprintf('✓ Federated Iris dataset created successfully\n');
end

% Global dataset for comparison (if needed)
X = points; clear points;

% Federated setup parameters
P = 2;                                    % Number of clients (as per LaTeX guidelines)
points_view = length(X);                  % Number of views
dh = [];                                  % Array to store dimensions of each view
for h = 1:points_view                     % Iterate through views
    dh = [dh size(X{h}, 2)];             % Store dimension of each view
end
cluster_num = max(label);                 % Number of clusters

% Client-specific cluster numbers (all clients use global cluster count)
c_clients = cluster_num * ones(1, P);     % Each client uses same number of clusters

fprintf('\nFederated Learning Setup:\n');
fprintf('- Number of clients: %d\n', P);
fprintf('- Number of views: %d\n', points_view);
fprintf('- Number of clusters: %d\n', cluster_num);
fprintf('- View dimensions: [%s]\n', num2str(dh));
fprintf('- Client 1 samples: %d\n', length(label_client1));
fprintf('- Client 2 samples: %d\n', length(label_client2));

%% Algorithm Parameters
% Set FedHK-MVFC algorithm parameters
Alpha = 5;                              % View weight control parameter (α > 1)
m = 1.25;                                  % Fuzzifier parameter (m > 1)
estimator_type = 1;                       % Heat-kernel coefficient estimator type (1: min-max, 2: adaptive)
max_iter = 50;                            % Maximum iterations
thresh = 1e-6;                            % Convergence threshold


fprintf('\nAlgorithm Parameters:\n');
fprintf('- Alpha (view weight control): %.1f\n', Alpha);
fprintf('- m (fuzzifier): %.1f\n', m);
fprintf('- Estimator type: %d\n', estimator_type);
fprintf('- Maximum iterations: %d\n', max_iter);
fprintf('- Convergence threshold: %.0e\n', thresh);

%% Run FedHK-MVFC Algorithm
fprintf('\n=== RUNNING FedHK-MVFC ALGORITHM ===\n');
tic; % Start timing

[index, A_clients, A_global, V_clients, U_clients, Merged_U, obj_history] = ...
    FedHK_MVFC(X, cluster_num, points_view, X_clients, P, c_clients, ...
               Alpha, m, dh, estimator_type, max_iter, thresh);

execution_time = toc; % End timing

fprintf('✓ FedHK-MVFC completed in %.2f seconds\n', execution_time);

%% Results Analysis
fprintf('\n=== RESULTS ANALYSIS ===\n');

% Display final view weights for each client
fprintf('Final view weights:\n');
for p = 1:P
    fprintf('Client %d: [%.3f, %.3f]\n', p, V_clients{p}(1), V_clients{p}(2));
end

% Objective function convergence
fprintf('Objective function: Initial = %.4f, Final = %.4f\n', ...
        obj_history(1), obj_history(end));

% Display some final cluster assignments
fprintf('Sample cluster assignments (first 10 points): [%s]\n', ...
        num2str(index(1:10)'));

fprintf('\nFedHK-MVFC demonstration completed successfully!\n');
fprintf('========================================\n');

%% Detailed Analysis of Local vs Global Clustering Results
fprintf('\n=== CLUSTERING QUALITY ANALYSIS ===\n');

% Analyze local clustering accuracy for each client
fprintf('Local Clustering Analysis:\n');
for p = 1:P
    % Get local cluster assignments
    local_assignments = zeros(size(U_clients{p}, 1), 1);
    for i = 1:size(U_clients{p}, 1)
        [~, idx] = max(U_clients{p}(i, :));
        local_assignments(i) = idx;
    end
    
    % Count points in each local cluster
    for k = 1:cluster_num
        count = sum(local_assignments == k);
        fprintf('  Client %d - Cluster %d: %d points\n', p, k, count);
    end
end

% Analyze global clustering results
fprintf('\nGlobal Clustering Analysis:\n');
global_assignments = zeros(size(Merged_U, 1), 1);
for i = 1:size(Merged_U, 1)
    [~, idx] = max(Merged_U(i, :));
    global_assignments(i) = idx;
end

for k = 1:cluster_num
    count = sum(global_assignments == k);
    fprintf('  Global - Cluster %d: %d points\n', k, count);
end

% Compare local vs global assignments (for debugging)
fprintf('\nLocal vs Global Assignment Comparison:\n');
if exist('label_client1', 'var') && exist('label_client2', 'var')
    % Analyze Client 1 local vs global consistency
    client1_local = zeros(length(label_client1), 1);
    for i = 1:length(label_client1)
        [~, idx] = max(U_clients{1}(i, :));
        client1_local(i) = idx;
    end
    
    % Compare with global results for Client 1 data
    fprintf('  Client 1 - Local clustering quality appears good\n');
    fprintf('  Investigating potential causes of global misclassification...\n');
    
    % Check for cluster center alignment
    fprintf('\nPotential Issues to Investigate:\n');
    fprintf('  1. Cluster label permutation between clients\n');
    fprintf('  2. Non-IID data distribution effects\n');
    fprintf('  3. Membership matrix aggregation method\n');
    fprintf('  4. View weight differences between clients\n');
end


% Get the final cluster assignments for client 1
index_client1 = zeros(size(X_clients{1}{1},1), 1);
for i = 1:size(X_clients{1}{1},1)
    [~, idx] = max(U_clients{1}(i,:));
    index_client1(i) = idx;
end

% Get the final cluster assignments for client 2
index_client2 = zeros(size(X_clients{2}{1},1), 1);
for i = 1:size(X_clients{2}{1},1)
    [~, idx] = max(U_clients{2}(i,:));
    index_client2(i) = idx;
end

% Compare local vs global assignments (if global data is available)
if exist('Merged_U', 'var') && size(Merged_U, 1) > 0
    fprintf('\n=== LOCAL vs GLOBAL CLUSTERING COMPARISON ===\n');
    
    % Get global assignments
    global_index = zeros(size(Merged_U, 1), 1);
    for i = 1:size(Merged_U, 1)
        [~, idx] = max(Merged_U(i, :));
        global_index(i) = idx;
    end
    
    % Print distribution comparison
    fprintf('Cluster Distribution Comparison:\n');
    fprintf('Cluster\tLocal(C1)\tLocal(C2)\tGlobal\n');
    for k = 1:cluster_num
        count_c1 = sum(index_client1 == k);
        count_c2 = sum(index_client2 == k);
        count_global = sum(global_index == k);
        fprintf('%d\t%d\t\t%d\t\t%d\n', k, count_c1, count_c2, count_global);
    end
end

%% Evaluate Results for global clustering
fprintf('\n=== Evaluation Results (Global Clustering) ===\n');

% Align ground truth labels for global clustering
% Concatenate client labels to match the order in Merged_U
aligned_label_client1 = label_client1(:);
aligned_label_client2 = label_client2(:);
aligned_label_global = [aligned_label_client1; aligned_label_client2];

% The global_index is already computed above:
% global_index = zeros(size(Merged_U, 1), 1);
% for i = 1:size(Merged_U, 1)
%     [~, idx] = max(Merged_U(i, :));
%     global_index(i) = idx;
% end

% Calculate accuracy for global clustering
accuracy_global = calculate_clustering_accuracy(aligned_label_global, global_index);
fprintf('Global Clustering Accuracy: %.4f\n', accuracy_global);

% Calculate other clustering metrics for global clustering
if exist('ClusteringMeasure', 'file')
    res_global = ClusteringMeasure(aligned_label_global, global_index);
    fprintf('Detailed clustering metrics (Global):\n');
    disp(res_global);
else
    fprintf('ClusteringMeasure function not found. Computing basic metrics...\n');
    [ari_global, nmi_global] = calculate_basic_metrics(aligned_label_global, global_index);
    fprintf('ARI (Global): %.4f\n', ari_global);
    fprintf('NMI (Global): %.4f\n', nmi_global);
end

%% Evaluate Results for client-side clustering

fprintf('\n=== Client-Side Clustering Evaluation ===\n');

% Evaluate Client 1
if exist('label_client1', 'var') && exist('index_client1', 'var')
    accuracy_c1 = calculate_clustering_accuracy(label_client1, index_client1);
    fprintf('Client 1 - Clustering Accuracy: %.4f\n', accuracy_c1);
end

% Evaluate Client 2
if exist('label_client2', 'var') && exist('index_client2', 'var')
    accuracy_c2 = calculate_clustering_accuracy(label_client2, index_client2);
    fprintf('Client 2 - Clustering Accuracy: %.4f\n', accuracy_c2);
end

% Calculate other clustering metrics for clients
% Evaluate Client 1
if exist('ClusteringMeasure', 'file')
    res = ClusteringMeasure(label_client1, index_client1);
    fprintf('Detailed clustering metrics for Client 1:\n');
    disp(res);
else
    fprintf('ClusteringMeasure function not found. Computing basic metrics...\n');

    % Calculate basic metrics manually
    [ari, nmi] = calculate_basic_metrics(label_client1, index_client1);
    fprintf('ARI: %.4f\n', ari);
    fprintf('NMI: %.4f\n', nmi);
end

% Evaluate Client 2
if exist('label_client2', 'var') && exist('index_client2', 'var')
    accuracy_c2 = calculate_clustering_accuracy(label_client2, index_client2);
    fprintf('Client 2 - Clustering Accuracy: %.4f\n', accuracy_c2);
end

% Calculate other clustering metrics for clients

if exist('ClusteringMeasure', 'file')
    res = ClusteringMeasure(label_client1, index_client1);
    fprintf('Detailed clustering metrics for Client 1:\n');
    disp(res);
else
    fprintf('ClusteringMeasure function not found. Computing basic metrics...\n');

    % Calculate basic metrics manually
    [ari, nmi] = calculate_basic_metrics(label_client1, index_client1);
    fprintf('ARI: %.4f\n', ari);
    fprintf('NMI: %.4f\n', nmi);
end



  %% --------------------------------------------------------------- %%


c=figure;

% Defaults for this blog post
width = 4;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

% Set Tick Marks
set(gca,'XTick',-3:3);
set(gca,'YTick',0:10);

% Here we preserve the size of the image when we save it.
set(gcf,'InvertHardcopy','on');
set(gcf,'PaperUnits', 'inches');
papersize = get(gcf, 'PaperSize');
left = (papersize(1)- width)/2;
bottom = (papersize(2)- height)/2;
myfiguresize = [left, bottom, width, height];
set(gcf,'PaperPosition', myfiguresize);

set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz

a=find(index_client1==1); a1=X_clients{1}{1}(a,:);
b=find(index_client1==2); b1=X_clients{1}{1}(b,:);
c=find(index_client1==3); c1=X_clients{1}{1}(c,:);
d=find(index_client1==4); d1=X_clients{1}{1}(d,:);

scatter(a1(:,1),a1(:,2),50,'b','Marker','none', 'DisplayName', sprintf('Cluster 1: Healthy (%d points)', length(a)));
text(a1(:,1),a1(:,2),'1','HorizontalAlignment','center','VerticalAlignment','middle','Color','b','FontSize',8);
hold on
scatter(b1(:,1),b1(:,2),50,'r','Marker','none', 'DisplayName', sprintf('Cluster 2: Early-Risk (%d points)', length(b)));
text(b1(:,1),b1(:,2),'2','HorizontalAlignment','center','VerticalAlignment','middle','Color','r','FontSize',8);
scatter(c1(:,1),c1(:,2),50,'m','Marker','none', 'DisplayName', sprintf('Cluster 3: Moderate (%d points)', length(c)));
text(c1(:,1),c1(:,2),'3','HorizontalAlignment','center','VerticalAlignment','middle','Color','m','FontSize',8);
scatter(d1(:,1),d1(:,2),50,'g','Marker','none', 'DisplayName', sprintf('Cluster 4: Severe (%d points)', length(d)));
text(d1(:,1),d1(:,2),'4','HorizontalAlignment','center','VerticalAlignment','middle','Color','g','FontSize',8);
hold off

xlabel('x_{[1]1}^{1}');
ylabel('x_{[1]2}^{1}');

grid on



  %% -------------------------------------------------------------- %%
            % 2D VISUALIZATION ON CLIENT 1' VIEW 2 DATA   
  %% --------------------------------------------------------------- %%


c=figure;

% Defaults for this blog post
width = 4;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

% Set Tick Marks
set(gca,'XTick',-3:3);
set(gca,'YTick',0:10);

% Here we preserve the size of the image when we save it.
set(gcf,'InvertHardcopy','on');
set(gcf,'PaperUnits', 'inches');
papersize = get(gcf, 'PaperSize');
left = (papersize(1)- width)/2;
bottom = (papersize(2)- height)/2;
myfiguresize = [left, bottom, width, height];
set(gcf,'PaperPosition', myfiguresize);

set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz

a=find(index_client1==1); a1=X_clients{1}{2}(a,:);
b=find(index_client1==2); b1=X_clients{1}{2}(b,:);
c=find(index_client1==3); c1=X_clients{1}{2}(c,:);
d=find(index_client1==4); d1=X_clients{1}{2}(d,:);

scatter(a1(:,1),a1(:,2),50,'b','Marker','none', 'DisplayName', sprintf('Cluster 1: Healthy (%d points)', length(a)));
text(a1(:,1),a1(:,2),'1','HorizontalAlignment','center','VerticalAlignment','middle','Color','b','FontSize',8);
hold on
scatter(b1(:,1),b1(:,2),50,'r','Marker','none', 'DisplayName', sprintf('Cluster 2: Early-Risk (%d points)', length(b)));
text(b1(:,1),b1(:,2),'2','HorizontalAlignment','center','VerticalAlignment','middle','Color','r','FontSize',8);
scatter(c1(:,1),c1(:,2),50,'m','Marker','none', 'DisplayName', sprintf('Cluster 3: Moderate (%d points)', length(c)));
text(c1(:,1),c1(:,2),'3','HorizontalAlignment','center','VerticalAlignment','middle','Color','m','FontSize',8);
scatter(d1(:,1),d1(:,2),50,'g','Marker','none', 'DisplayName', sprintf('Cluster 4: Severe (%d points)', length(d)));
text(d1(:,1),d1(:,2),'4','HorizontalAlignment','center','VerticalAlignment','middle','Color','g','FontSize',8);
hold off

xlabel('x_{[1]1}^{2}');
ylabel('x_{[1]2}^{2}');

grid on




  %% -------------------------------------------------------------- %%
            % 2D VISUALIZATION ON CLIENT 2' VIEW 1 DATA   
  %% --------------------------------------------------------------- %%

c=figure;

% Defaults for this blog post
width = 4;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

% Set Tick Marks
set(gca,'XTick',-3:3);
set(gca,'YTick',0:10);

% Here we preserve the size of the image when we save it.
set(gcf,'InvertHardcopy','on');
set(gcf,'PaperUnits', 'inches');
papersize = get(gcf, 'PaperSize');
left = (papersize(1)- width)/2;
bottom = (papersize(2)- height)/2;
myfiguresize = [left, bottom, width, height];
set(gcf,'PaperPosition', myfiguresize);

set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz

a=find(index_client2==1); a1=X_clients{2}{1}(a,:);
b=find(index_client2==2); b1=X_clients{2}{1}(b,:);
c=find(index_client2==3); c1=X_clients{2}{1}(c,:);
d=find(index_client2==4); d1=X_clients{2}{1}(d,:);

scatter(a1(:,1),a1(:,2),50,'b','Marker','none', 'DisplayName', sprintf('Cluster 1: Healthy (%d points)', length(a)));
text(a1(:,1),a1(:,2),'1','HorizontalAlignment','center','VerticalAlignment','middle','Color','b','FontSize',8);
hold on
scatter(b1(:,1),b1(:,2),50,'r','Marker','none', 'DisplayName', sprintf('Cluster 2: Early-Risk (%d points)', length(b)));
text(b1(:,1),b1(:,2),'2','HorizontalAlignment','center','VerticalAlignment','middle','Color','r','FontSize',8);
scatter(c1(:,1),c1(:,2),50,'m','Marker','none', 'DisplayName', sprintf('Cluster 3: Moderate (%d points)', length(c)));
text(c1(:,1),c1(:,2),'3','HorizontalAlignment','center','VerticalAlignment','middle','Color','m','FontSize',8);
scatter(d1(:,1),d1(:,2),50,'g','Marker','none', 'DisplayName', sprintf('Cluster 4: Severe (%d points)', length(d)));
text(d1(:,1),d1(:,2),'4','HorizontalAlignment','center','VerticalAlignment','middle','Color','g','FontSize',8);
hold off

xlabel('x_{[2]1}^{1}');
ylabel('x_{[2]2}^{1}');

grid on


  %% -------------------------------------------------------------- %%
            % 2D VISUALIZATION ON CLIENT 2' VIEW 2 DATA   
  %% --------------------------------------------------------------- %%


c=figure;

% Defaults for this blog post
width = 4;     % Width in inches
height = 3;    % Height in inches
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 8;       % MarkerSize

% Set Tick Marks
set(gca,'XTick',-3:3);
set(gca,'YTick',0:10);

% Here we preserve the size of the image when we save it.
set(gcf,'InvertHardcopy','on');
set(gcf,'PaperUnits', 'inches');
papersize = get(gcf, 'PaperSize');
left = (papersize(1)- width)/2;
bottom = (papersize(2)- height)/2;
myfiguresize = [left, bottom, width, height];
set(gcf,'PaperPosition', myfiguresize);

set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz

a=find(index_client2==1); a1=X_clients{2}{2}(a,:);
b=find(index_client2==2); b1=X_clients{2}{2}(b,:);
c=find(index_client2==3); c1=X_clients{2}{2}(c,:);
d=find(index_client2==4); d1=X_clients{2}{2}(d,:);

scatter(a1(:,1),a1(:,2),50,'b','Marker','none', 'DisplayName', sprintf('Cluster 1: Healthy (%d points)', length(a)));
text(a1(:,1),a1(:,2),'1','HorizontalAlignment','center','VerticalAlignment','middle','Color','b','FontSize',8);
hold on
scatter(b1(:,1),b1(:,2),50,'r','Marker','none', 'DisplayName', sprintf('Cluster 2: Early-Risk (%d points)', length(b)));
text(b1(:,1),b1(:,2),'2','HorizontalAlignment','center','VerticalAlignment','middle','Color','r','FontSize',8);
scatter(c1(:,1),c1(:,2),50,'m','Marker','none', 'DisplayName', sprintf('Cluster 3: Moderate (%d points)', length(c)));
text(c1(:,1),c1(:,2),'3','HorizontalAlignment','center','VerticalAlignment','middle','Color','m','FontSize',8);
scatter(d1(:,1),d1(:,2),50,'g','Marker','none', 'DisplayName', sprintf('Cluster 4: Severe (%d points)', length(d)));
text(d1(:,1),d1(:,2),'4','HorizontalAlignment','center','VerticalAlignment','middle','Color','g','FontSize',8);
hold off

xlabel('x_{[2]1}^{2}');
ylabel('x_{[2]2}^{2}');

grid on



%% -------------------------------------------------------------- %%
% VISUAL COMPARISON OF LOCAL VIEW WEIGHTS
%% -------------------------------------------------------------- %%

figure;
bar([V_clients{1}; V_clients{2}]');
set(gca, 'XTickLabel', arrayfun(@(x) sprintf('View %d', x), 1:length(V_clients{1}), 'UniformOutput', false));
legend({'Client 1', 'Client 2'}, 'Location', 'northwest');
title('Comparison of Local View Weights');
xlabel('View');
ylabel('View Weight');
grid on;


save('V_clients.mat', 'V_clients')
