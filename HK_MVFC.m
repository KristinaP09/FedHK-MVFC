%--------------------------------------------------------------------------

% Authors: Kristina P. Sinaga
% Date: Aug 12th, 2025
% Version: 1.0
% Tested on: MATLAB R2024b

%--------------------------------------------------------------------------

% Copyright (c) 2025 Kristina P. Sinaga
% Contact: ndkristinaps25@gmail.com

%--------------------------------------------------------------------------

%% Function Arguments:
%   Inputs:
%     X           - (cell array) Multi-view dataset matrices
%     cluster_num - (integer) Number of clusters to form
%     points_view - (integer) Number of data views
%     Alpha       - (integer/vector) Exponent parameter to control view weights
%     m           - (integer) The fuzzifier parameter (m > 1)
%     Delta       - (cell array) heat-kernel coefficients in Euclidean space
%     dh          - (vector) View-specific dimension parameters
%     thresh      - (integer) convergence threshold (optional, default 1e-6)

%   Outputs:
%     A           - (cell array) view-specific cluster centers
%     V           - (vector) View weights for all data views
%     U           - (matrix) Membership matrices for all data views
%     index       - (vector) Cluster assignments for each data point
%     obj_history - objective function values over iterations

% Internal Variables:
%   s            - Shorthand for points_view
%   c            - Shorthand for cluster_num
%   data_n       - Number of data points
%   Param_Alpha  - Copy of Alpha parameter
%   Param_m      - Copy of m parameter
%   Param_Delta  - Copy of Delta parameter
%   time         - Current iteration counter
%   max_time     - Maximum number of iterations
%   obj_HK_MVFC  - Objective function values over iterations

% Algorithm Stages:
%   1. Initialization Stage - Setup of cluster centers and weight factors
%   2. Compute the H-KC Delta
%   3. Compute memberships matrix U
%   4. Update cluster centers A
%   5. Update view weights V
%   6. Convergence Step  - Final model evaluation

% Notes:
%   - Default maximum iterations is 50 (adjustable)
%--------------------------------------------------------------------------

%-------------------------------------------------------------------------------------------------------------------------------------
% Contributed by Kristina P. Sinaga 
%--------------------------------------------------------------------------------------------------------------------------------------
% Implements HK-MVFC (Exponential Multi-View Fuzzy C-Means) Algorithm
% Based on mathematical formulation in EB-MVFCM.tex Section 3.2.1 (lines 406-451)
% Input   : X is a multi-view dataset (sample-view space)
%           cluster_num is the number of clusters (c)
%           points_view is the number of data views (s)
%           m is the fuzzifier parameter (m > 1)
%           alpha is the view weight exponent parameter (alpha > 1)
%           estimator_type: 1 for min-max normalization, 2 for mean deviation
%           dh is the hth view number of dimensions
%           max_iter: maximum iterations (optional, default 100)
%           thresh: convergence threshold (optional, default 1e-6)
%-------------------------------------------------------------------------------------------------------------------
% Output : A  : the cluster centers for h-th view data
%          V  : the view weights
%          U  : the fuzzy membership matrix (soft clustering)
%          index : the hard cluster assignment for multi-view data (derived from U)
%          obj_history : objective function values over iterations
%-------------------------------------------------------------------------------------------------------------------
function [A, V, U, index, obj_history] = HK_MVFC(X, cluster_num, points_view, m, alpha, estimator_type, dh, max_iter, thresh)

% Parameter setup and validation
if nargin < 6
    error('HK-MVFCM requires at least 6 input arguments');
end
if nargin < 7 || isempty(max_iter)
    max_iter = 100;
end
if nargin < 8 || isempty(thresh)
    thresh = 1e-6;
end

s = points_view;
c = cluster_num;
data_n = size(X{1},1);

% Validate inputs
if m <= 1
    warning('Fuzzifier m should be > 1 for fuzzy clustering. Setting m = 2.0');
    m = 2.0;
end
if alpha <= 1
    warning('Alpha should be > 1 for proper view weighting. Setting alpha = 2.0');
    alpha = 2.0;
end

%--------------------------------------------------------------------------
% Initialize the cluster centers A using k-means++ for better starting centers
for h = 1:s
    try
        % Use k-means++ initialization for better starting centers
        [~, A{h}] = kmeans(X{h}, c, 'Replicates', 5, 'MaxIter', 50, 'Start', 'plus');
    catch
        % Fallback to FCM initialization if kmeans fails
        fprintf('Warning: K-means initialization failed for view %d, using FCM centers\n', h);
        [A{h}, ~, ~] = fcm(X{h}, c);
    end
end

%--------------------------------------------------------------------------
% Initialize the weighted view V (uniform initialization)
V = ones(1,s) / s;

%--------------------------------------------------------------------------
% Initialize other variables
time = 1;
obj_history = zeros(1, max_iter);
epsilon = thresh;

%--------------------------------------------------------------------------
% Compute heat-kernel coefficients delta_{ij}^h
delta = cell(s, 1);
for h = 1:s
    delta{h} = zeros(data_n, dh(h));
    for j = 1:dh(h)
        if estimator_type == 1
            % Min-max normalization estimator (Eq. deltaest)
            min_val = min(X{h}(:, j));
            max_val = max(X{h}(:, j));
            if max_val > min_val
                delta{h}(:, j) = (X{h}(:, j) - min_val) / (max_val - min_val);
            else
                delta{h}(:, j) = ones(data_n, 1);  % Handle constant features
            end
        else
            % Mean deviation estimator (Eq. deltaest2)
            mean_val = mean(X{h}(:, j));
            delta{h}(:, j) = abs(X{h}(:, j) - mean_val);
        end
    end
end

%--------------------------------------------------------------------------
% Start the main E-MVFCM iteration
%--------------------------------------------------------------------------
while time <= max_iter
    fprintf('--------------  HK-MVFCM Iteration %d ----------------------\n', time);
    
    %% Step 1: Compute heat-kernel coefficients δ_{ij}^h using Eq. deltaest
    for h = 1:s
        for j = 1:dh(h)
            for i = 1:data_n
                if estimator_type == 1
                    % Min-max normalization estimator (Eq. deltaest)
                    min_val = min(X{h}(:, j));
                    max_val = max(X{h}(:, j));
                    if max_val > min_val
                        delta{h}(i, j) = (X{h}(i, j) - min_val) / (max_val - min_val);
                    else
                        delta{h}(i, j) = 0.5; % Default value for constant features
                    end
                else
                    % Mean deviation estimator (Eq. deltaest2)
                    mean_val = mean(X{h}(:, j));
                    delta{h}(i, j) = abs(X{h}(i, j) - mean_val);
                end
            end
        end
    end
    
    %% Step 2: Compute exponential distances d_exp(ik,j)^h = KED_2
    % KED_2(x_i^h, a_k^h) = 1 - exp(-∑_{j=1}^{d_h} δ_{ij}^h (x_{ij}^h - a_{kj}^h)^2)
    d_exp = cell(s, 1);
    for h = 1:s
        d_exp{h} = zeros(data_n, c);
        for i = 1:data_n
            for k = 1:c
                % Compute sum of weighted squared differences
                weighted_sum = 0;
                for j = 1:dh(h)
                    weighted_sum = weighted_sum + delta{h}(i, j) * (X{h}(i, j) - A{h}(k, j))^2;
                end
                % Exponential distance: d_exp = 1 - exp(-weighted_sum)
                d_exp{h}(i, k) = 1 - exp(-weighted_sum);
            end
        end
    end
    
    %% Step 3: Update fuzzy membership matrix U using Eq. UpdateU_E-MVFCM
    % μ_{ik}^* = (∑_{h=1}^s v_h^α d_exp(ik,j)^h)^{-1/(m-1)} / ∑_{k'=1}^c (∑_{h=1}^s v_h^α d_exp(ik',j)^h)^{-1/(m-1)}
    U = zeros(data_n, c);
    for i = 1:data_n
        % Compute denominator for normalization
        denominator_sum = 0;
        distance_ik = zeros(1, c);
        
        for k = 1:c
            % Compute weighted distance across all views for point i to cluster k
            weighted_distance = 0;
            for h = 1:s
                weighted_distance = weighted_distance + (V(h)^alpha) * d_exp{h}(i, k);
            end
            distance_ik(k) = weighted_distance;
            
            % Add to denominator (avoid division by zero)
            if weighted_distance > eps
                denominator_sum = denominator_sum + (weighted_distance^(-1/(m-1)));
            else
                denominator_sum = denominator_sum + 1e10; % Very large value for zero distance
            end
        end
        
        % Compute fuzzy membership for each cluster
        for k = 1:c
            if distance_ik(k) > eps && denominator_sum > eps
                U(i, k) = (distance_ik(k)^(-1/(m-1))) / denominator_sum;
            else
                % Handle edge case: if distance is zero, assign full membership
                if distance_ik(k) <= eps
                    U(i, k) = 1.0;
                    % Set other memberships to zero
                    for k_other = 1:c
                        if k_other ~= k
                            U(i, k_other) = 0.0;
                        end
                    end
                    break;
                else
                    U(i, k) = 1/c; % Equal membership as fallback
                end
            end
        end
        
        % Ensure membership normalization (∑_k μ_{ik} = 1)
        membership_sum = sum(U(i, :));
        if membership_sum > eps
            U(i, :) = U(i, :) / membership_sum;
        else
            U(i, :) = ones(1, c) / c; % Equal membership fallback
        end
    end
    
    %% Step 4: Update cluster centers A using Eq. UpdateA_E-MVFCM
    % a_{kj}^h = ∑_{i=1}^n (μ_{ik}^*)^m v_h^α exp(-δ_{ij}^h ||x_i^h - a_k^h||^2) x_{ij}^h / ∑_{i=1}^n (μ_{ik}^*)^m v_h^α exp(-δ_{ij}^h ||x_i^h - a_k^h||^2)
    for h = 1:s
        for k = 1:c
            for j = 1:dh(h)
                numerator = 0;
                denominator = 0;
                
                for i = 1:data_n
                    % Compute exponential weight for this specific feature j
                    % Note: Using only delta_{ij}^h for feature j, not the full vector
                    exp_weight = exp(-delta{h}(i, j) * (X{h}(i, j) - A{h}(k, j))^2);
                    
                    % Combined weight: (μ_{ik}^*)^m * v_h^α * exp(...)
                    combined_weight = (U(i, k)^m) * (V(h)^alpha) * exp_weight;
                    
                    numerator = numerator + combined_weight * X{h}(i, j);
                    denominator = denominator + combined_weight;
                end
                
                % Update cluster center
                if denominator > eps
                    A{h}(k, j) = numerator / denominator;
                end
            end
        end
    end
    
    %% Step 5: Update view weights V using Eq. UpdateV_E-MVFCM
    % v_h = (∑_{i=1}^n ∑_{k=1}^c (μ_{ik}^*)^m d_exp(ik,j)^h)^{-1/(α-1)} / ∑_{h'=1}^s (∑_{i=1}^n ∑_{k=1}^c (μ_{ik}^*)^m d_exp(ik',j)^{h'})^{-1/(α-1)}
    V_numerator = zeros(1, s);
    
    % Compute the weighted sum for each view
    for h = 1:s
        view_sum = 0;
        for i = 1:data_n
            for k = 1:c
                view_sum = view_sum + (U(i, k)^m) * d_exp{h}(i, k);
            end
        end
        
        % Compute numerator for view h
        if view_sum > eps
            V_numerator(h) = (view_sum)^(-1/(alpha-1));
        else
            V_numerator(h) = 1e10; % Very large value for zero cost
        end
    end
    
    % Normalize to get view weights
    V_denominator = sum(V_numerator);
    if V_denominator > eps
        V = V_numerator / V_denominator;
    else
        V = ones(1, s) / s; % Fallback to uniform weights
    end
    
    %% Step 6: Compute objective function J_E-MVFCM using Eq. E-MVFCM
    % J_E-MVFCM = ∑_{h=1}^s v_h^α ∑_{i=1}^n ∑_{k=1}^c (μ_{ik}^*)^m d_exp(ik,j)^h
    obj = 0;
    for h = 1:s
        view_obj = 0;
        for i = 1:data_n
            for k = 1:c
                view_obj = view_obj + (U(i, k)^m) * d_exp{h}(i, k);
            end
        end
        obj = obj + (V(h)^alpha) * view_obj;
    end
    obj_history(time) = obj;
    
    % Print the result
    fprintf('HK-MVFCM: Iteration count = %d, Objective = %f\n', time, obj_history(time));
    fprintf('View weights: [');
    for h = 1:s
        fprintf('%.3f ', V(h));
    end
    fprintf(']\n');
    
    %% Step 7: Check convergence
    if time > 1 && abs(obj_history(time) - obj_history(time-1)) <= epsilon
        fprintf('------------ HK-MVFCM has converged -----------\n\n');
        obj_history = obj_history(1:time);  % Trim unused entries
        break;
    end
    
    time = time + 1;
end

% Trim objective history if max iterations reached without convergence
if time > max_iter
    obj_history = obj_history(1:max_iter);
    fprintf('------------ HK-MVFCM reached maximum iterations -----------\n\n');
end

%% Generate final hard cluster assignment from fuzzy memberships
index = zeros(data_n, 1);
for i = 1:data_n
    [~, idx] = max(U(i, :));
    index(i) = idx;
end

end
