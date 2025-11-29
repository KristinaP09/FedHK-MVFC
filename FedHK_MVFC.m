%% Manuscript title: Heat Kernel-Enhanced Multi-View Clustering: Centralized and Federated Approaches
%% FedHK-MVFC: Federated Heat Kernel-Enhanced Multi-View Fuzzy Clustering Algorithm Implementation
%% FedHK-MVFC bridging quantum field theory concepts with practical decentralized learning algorithm
%% Clustering that leverages heat-kernel coefficients (HK-C) 
% This implementation presents a novel centralized HK-MVFC algorithm for multi-view
%% Objective Function 
%% J_{FedHK - MVFC}^\ell (V,{U^*},A) = \sum\limits_{\ell  = 1}^M {\sum\limits_{h = 1}^{s\left( \ell  \right)} {v_{\left[ \ell  \right]h}^\alpha \sum\limits_{i = 1}^{n\left( \ell  \right)} {\sum\limits_{k = 1}^{c\left( \ell  \right)} {{{\left( {\mu _{\left[ \ell  \right]ik}^*} \right)}^m}{\rm{FKED}}\left( {x_{\left[ \ell  \right]ij}^h,a_{\left[ \ell  \right]kj}^h} \right)} } } } 
%% \text{FKED}\left( {x_{\left[ \ell  \right]ij}^h,a_{\left[ \ell  \right]kj}^h} \right) = 1 - \exp \left( { - \sum\limits_{h = 1}^{s\left( \ell  \right)} {\sum\limits_{j = 1}^{d_{\left[ \ell  \right]}^h} {\delta _{\left[ \ell  \right]ij}^h{{\left( {x_{\left[ \ell  \right]ij}^h - a_{\left[ \ell  \right]kj}^h} \right)}^2}} } } \right)

%--------------------------------------------------------------------------

% Authors: Kristina P. Sinaga
% Date: Aug 13th, 2025
% Version: 1.0
% Tested on: MATLAB R2024b

%--------------------------------------------------------------------------

% Copyright (c) 2025 Kristina P. Sinaga
% Contact: ndkristinaps25@gmail.com

%--------------------------------------------------------------------------

%% Function Arguments:
%   Inputs:
%     X           - (cell array) Global multi-view dataset matrices
%     cluster_num - (integer) Global collective number of clusters
%     points_view - (integer) Number of data views
%     X_clients   - (cell array) Client-specific multi-view dataset matrices
%     P           - (integer) Number of clients
%     c_clients   - (vector) Number of clusters for each client (not used, all clients use global c)
%     Alpha       - (scalar) Exponent parameter to control view weights (α > 1)
%     m           - (scalar) The fuzzifier parameter (m > 1)
%     dh          - (vector) View-specific dimension parameters
%     estimator_type - (integer) Heat-kernel coefficient estimator type (1 or 2)
%     max_iter    - (integer) Maximum number of iterations (optional, default 50)
%     thresh      - (scalar) Convergence threshold (optional, default 1e-6)

%   Outputs:
%     index       - (vector) Final cluster assignments for all data points
%     A_clients   - (cell array) Client-specific view cluster centers
%     A_global    - (cell array) Global view cluster centers
%     V_clients   - (cell array) Client-specific view weights
%     U_clients   - (cell array) Client-specific membership matrices
%     Merged_U    - (matrix) Merged membership matrix across all clients
%     obj_history - (vector) Objective function values over iterations

% Internal Variables:
%   s            - Shorthand for points_view
%   c            - shorthand for global cluster_num
%   c_clients    - Shorthand for c_clients
%   data_n       - Number of collective data points from participating clients
%   n_clients    - Number of samples held by each client
%   Param_Alpha  - Copy of Alpha parameter
%   Param_m      - Copy of m parameter
%   Param_Delta  - Copy of Delta parameter
%   Delta_clients- (cell array) Client-specific heat-kernel coefficients
%   time         - Current iteration counter
%   max_time     - Maximum number of iterations
%   obj_FedHK_MVFC  - Objective function values over iterations

% Algorithm Stages:
%   1. Initialization: Global cluster centers and client-specific parameters
%   2. Iterative Federated Learning Loop:
%        a. Compute Federated Heat-Kernel Coefficients (FedH-KC) using Eq. fed_delta
%        b. Compute FKED (Federated Kernel Exponential Distance)
%        c. Update membership matrices using Eq. diff_U_FedHKMVFC
%        d. Update cluster centers using Eq. diff_A_FedHKMVFC
%        e. Update view weights using Eq. diff_V_FedHKMVFC
%        f. Federated aggregation of global model
%   3. Convergence check and final cluster assignment generation

% Mathematical Equations Implemented:
%   - FedH-KC: δ[ℓ]ij^h = (x[ℓ]ij^h - min(x[ℓ]j^h)) / (max(x[ℓ]j^h) - min(x[ℓ]j^h))
%   - FKED: 1 - exp(-Σ_j δ[ℓ]ij^h * (x[ℓ]ij^h - a[ℓ]kj^h)²)
%   - Membership update: Eq. diff_U_FedHKMVFC
%   - Cluster center update: Eq. diff_A_FedHKMVFC
%   - View weight update: Eq. diff_V_FedHKMVFC

% Notes:
%   - Follows the exact mathematical formulations from the LaTeX document
%   - Implements proper federated learning with weighted aggregation
%   - Includes convergence checking and objective function monitoring
%--------------------------------------------------------------------------

function [index, A_clients, A_global, V_clients, U_clients, Merged_U, obj_history] = FedHK_MVFC(X, cluster_num, points_view, X_clients, P, c_clients, Alpha, m, dh, estimator_type, max_iter, thresh)
% Variable initialization with proper documentation
% Basic parameters
s = points_view;            % Number of views (shorthand)
c = cluster_num;            % Number of clusters (shorthand)
data_n = size(X{1},1);     % Total number of data points

% Parameter validation
if nargin < 10
    estimator_type = 1;     % Default to min-max normalization
end
if nargin < 11
    max_iter = 50;          % Default maximum iterations
end
if nargin < 12
    thresh = 1e-6;          % Default convergence threshold
end

% Parameter copies for internal use
Param_Alpha = Alpha;        % View weight control parameter
Param_m = m;               % Fuzzifier parameter

%--------------------------------------------------------------------------
%% INITIALIZATION STAGE: Seamless Central Server
% Initialize global model cluster centers and distribute to clients
%--------------------------------------------------------------------------
disp('- The seamless central server initialize the global cluster centers A...');
for h = 1:s
    % Initialize cluster centers using k-means with multiple restarts
    try
        [~, A_global{h}] = kmeans(X{h}, c, 'Replicates', 10, 'MaxIter', 100);
    catch
        % Fallback initialization if kmeans fails
        A_global{h} = X{h}(randperm(size(X{h},1), c), :);
    end
end
disp('- The seamless central server send the initial global model of A to all clients...');

%--------------------------------------------------------------------------
% Initialize client-specific parameters
n_clients = zeros(1,P);           % Array to store number of instances per client

% Calculate number of instances per client
for p = 1:P
    n_clients(p) = size(X_clients{p}{1}, 1);  % Assuming all views have same number of samples per client
end

%--------------------------------------------------------------------------
%% INITIALIZATION STAGE 1: CLIENTS
% Initialize client-side parameters and variables
%--------------------------------------------------------------------------
% Initialize view weights for each client
disp('- The clients initialize their weighted view V...');
for p = 1:P
    V_clients{p} = ones(1,s)./s;  % Equal weights initially
end

% Initialize local cluster centers for each client (copy from global)
for p = 1:P
    for h = 1:s
        A_clients{p}{h} = A_global{h};  % Initialize with global centers
    end
end

% Initialize iteration variables
time = 1;                             % Current iteration
obj_history = zeros(1,max_iter);      % Store objective values

%--------------------------------------------------------------------------
% Start the main federated learning iteration
%--------------------------------------------------------------------------
for time = 1:max_iter
    fprintf('--------------  Iteration %d Starts ---------------\n', time); 
    
    %% --------------------------------------------------------------------- %%
    %                           STAGE 1: CLIENTS                               %
    %% --------------------------------------------------------------------- %%  
    
    % Step 1: Compute Federated Heat-Kernel Coefficients (FedH-KC)
    % Using Eq. fed_delta from the LaTeX document
    for p = 1:P
        for h = 1:s
            for j = 1:dh(h)
                for i = 1:n_clients(p)
                    if estimator_type == 1
                        % Min-max normalization estimator (Eq. fed_delta)
                        min_val = min(X_clients{p}{h}(:, j));
                        max_val = max(X_clients{p}{h}(:, j));
                        if max_val > min_val
                            delta_clients{p}{h}(i, j) = (X_clients{p}{h}(i, j) - min_val) / (max_val - min_val);
                        else
                            delta_clients{p}{h}(i, j) = 0.5; % Default for constant features
                        end
                    else
                        % Mean deviation estimator
                        mean_val = mean(X_clients{p}{h}(:, j));
                        delta_clients{p}{h}(i, j) = abs(X_clients{p}{h}(i, j) - mean_val);
                    end
                end
            end
        end
    end

    %--------------------------------------------------------------------------
    % Step 2: Compute FKED (Federated Kernel Exponential Distance)
    % Using the equation: FKED = 1 - exp(-sum(delta * (x-a)^2))
    for p = 1:P
        for h = 1:s
            FKED_clients{p}{h} = zeros(n_clients(p), c);
            for i = 1:n_clients(p)
                for k = 1:c
                    % Compute the exponential term for current view h
                    weighted_sum = 0;
                    for j = 1:dh(h)
                        weighted_sum = weighted_sum + delta_clients{p}{h}(i, j) * ...
                            (X_clients{p}{h}(i, j) - A_clients{p}{h}(k, j))^2;
                    end
                    % FKED = 1 - exp(-weighted_sum)
                    FKED_clients{p}{h}(i, k) = 1 - exp(-weighted_sum);
                end
            end
        end
    end
    
    %--------------------------------------------------------------------------
    % Step 3: Update membership matrix U using Eq. diff_U_FedHKMVFC
    for p = 1:P
        U_clients{p} = zeros(n_clients(p), c);
        
        for i = 1:n_clients(p)
            % Compute denominator for normalization
            denominator_sum = 0;
            distance_ik = zeros(1, c);
            
            for k = 1:c
                % Compute weighted FKED across all views for point i to cluster k
                weighted_distance = 0;
                for h = 1:s
                    weighted_distance = weighted_distance + ...
                        (V_clients{p}(h)^Param_Alpha) * FKED_clients{p}{h}(i, k);
                end
                distance_ik(k) = weighted_distance;
                
                % Add to denominator (avoid division by zero)
                if weighted_distance > eps
                    denominator_sum = denominator_sum + (weighted_distance^(-1/(Param_m-1)));
                else
                    denominator_sum = denominator_sum + 1e10; % Very large value for zero distance
                end
            end
            
            % Compute fuzzy membership for each cluster
            for k = 1:c
                if distance_ik(k) > eps && denominator_sum > eps
                    U_clients{p}(i, k) = (distance_ik(k)^(-1/(Param_m-1))) / denominator_sum;
                else
                    % Handle edge case: if distance is zero, assign full membership
                    if distance_ik(k) <= eps
                        U_clients{p}(i, k) = 1.0;
                        % Set other memberships to zero
                        for k_other = 1:c
                            if k_other ~= k
                                U_clients{p}(i, k_other) = 0.0;
                            end
                        end
                        break;
                    else
                        U_clients{p}(i, k) = 1/c; % Equal membership as fallback
                    end
                end
            end
            
            % Ensure membership normalization
            membership_sum = sum(U_clients{p}(i, :));
            if membership_sum > eps
                U_clients{p}(i, :) = U_clients{p}(i, :) / membership_sum;
            else
                U_clients{p}(i, :) = ones(1, c) / c; % Equal membership fallback
            end
        end
    end  
    %--------------------------------------------------------------------------
    % Step 4: Update cluster centers A using Eq. diff_A_FedHKMVFC
    for p = 1:P
        for h = 1:s
            for k = 1:c
                for j = 1:dh(h)
                    numerator = 0;
                    denominator = 0;
                    
                    for i = 1:n_clients(p)
                        % Compute exponential weight for current feature j
                        % exp(-delta_{ij}^h * ||x_i^h - a_k^h||^2)
                        squared_distance = 0;
                        for jj = 1:dh(h)
                            squared_distance = squared_distance + ...
                                (X_clients{p}{h}(i, jj) - A_clients{p}{h}(k, jj))^2;
                        end
                        exp_weight = exp(-delta_clients{p}{h}(i, j) * squared_distance);
                        
                        % Combined weight: (μ_{ik}^*)^m * v_h^α * exp(...)
                        combined_weight = (U_clients{p}(i, k)^Param_m) * ...
                            (V_clients{p}(h)^Param_Alpha) * exp_weight;
                        
                        numerator = numerator + combined_weight * X_clients{p}{h}(i, j);
                        denominator = denominator + combined_weight;
                    end
                    
                    % Update cluster center
                    if denominator > eps
                        A_clients{p}{h}(k, j) = numerator / denominator;
                    end
                end
            end
        end
    end
    
    %--------------------------------------------------------------------------
    % Step 5: Update view weights V using Eq. diff_V_FedHKMVFC
    for p = 1:P
        V_numerator = zeros(1, s);
        
        % Compute the weighted sum for each view
        for h = 1:s
            view_sum = 0;
            for i = 1:n_clients(p)
                for k = 1:c
                    view_sum = view_sum + (U_clients{p}(i, k)^Param_m) * FKED_clients{p}{h}(i, k);
                end
            end
            
            % Compute numerator for view h
            if view_sum > eps
                V_numerator(h) = (view_sum)^(-1/(Param_Alpha-1));
            else
                V_numerator(h) = 1e10; % Very large value for zero cost
            end
        end
        
        % Normalize to get view weights
        V_denominator = sum(V_numerator);
        if V_denominator > eps
            V_clients{p} = V_numerator / V_denominator;
        else
            V_clients{p} = ones(1, s) / s; % Fallback to uniform weights
        end
    end   
    %% --------------------------------------------------------------------- %%
    %        STAGE 2: Federated Learning Server Aggregation                    %
    %% --------------------------------------------------------------------- %% 
    fprintf('- The FL server aggregating models from %d clients at iteration %d \n', P, time); 
    
    % Aggregate cluster centers using weighted averaging based on data sizes
    for h = 1:s
        % Weighted average of client cluster centers
        weighted_sum = zeros(c, dh(h));
        total_weight = 0;
        
        for p = 1:P
            weight = n_clients(p) / sum(n_clients); % Weight by data size
            weighted_sum = weighted_sum + weight * A_clients{p}{h};
            total_weight = total_weight + weight;
        end
        
        % Update global cluster centers
        if total_weight > eps
            A_global{h} = weighted_sum / total_weight;
        end
    end
    
    % Update local client models with weighted combination of global and local models
    gamma = 0.7; % Personalization parameter (can be made adaptive)
    for p = 1:P
        for h = 1:s
            A_clients{p}{h} = gamma * A_global{h} + (1 - gamma) * A_clients{p}{h};
        end
    end
    
    %--------------------------------------------------------------------------
    % Step 6: Compute objective function value
    disp('- Computing objective function values...');
    obj_total = 0;
    
    for p = 1:P
        obj_client = 0;
        for h = 1:s
            for i = 1:n_clients(p)
                for k = 1:c
                    obj_client = obj_client + (V_clients{p}(h)^Param_Alpha) * ...
                        (U_clients{p}(i, k)^Param_m) * FKED_clients{p}{h}(i, k);
                end
            end
        end
        obj_total = obj_total + obj_client;
    end
    
    obj_history(time) = obj_total;
    
    %--------------------------------------------------------------------------
    % Check convergence criteria 
    fprintf('FedHK-MVFC: Iteration = %d, Objective = %f\n', time, obj_history(time));
    
    if time > 1 && abs(obj_history(time) - obj_history(time-1)) <= thresh
        fprintf('------------ FedHK-MVFC has converged -----------\n');
        obj_history = obj_history(1:time);  % Trim unused entries
        break;
    end
    
    if time == max_iter
        fprintf('------------ FedHK-MVFC reached maximum iterations -----------\n');
        break;
    end
end

%--------------------------------------------------------------------------
% Final step: Generate cluster assignments
%--------------------------------------------------------------------------
% Merge membership matrices from all clients
Merged_U = [];
for p = 1:P
    Merged_U = [Merged_U; U_clients{p}];
end

% Generate final hard cluster assignments
index = zeros(data_n, 1);
for i = 1:data_n
    [~, idx] = max(Merged_U(i, :));
    index(i) = idx;
end

disp('------------ FedHK-MVFC Algorithm Completed -----------');
end
