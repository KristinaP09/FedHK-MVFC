%%   This code is provided (written and created) by Kristina P. Sinaga 
%%   Copyright (c) 2025 Kristina P. Sinaga 
%%   All rights reserved
%%   Contact :  krist.p.sinaga@gmail.com
%%   ------------------------------------------------------------------- %%
%%   Modified: Aug 13, 2025 - Enhanced to generate distinct cluster shapes
%%   Two-view, four-cluster multi-view data with different geometric shapes

clear all;close all;clc

%% Generate View 1 - Four clusters with distinct shapes (Expanded Format)
rng(45); 
points_per_cluster = 2500;  % 10000 total points / 4 clusters

% Cluster 1: Circular shape (Bottom-left quadrant)
center1 = [-8, -8];
theta1 = 2*pi*rand(points_per_cluster, 1);
r1 = 1.5 * sqrt(rand(points_per_cluster, 1));  % Increased radius
cluster1 = [center1(1) + r1.*cos(theta1), center1(2) + r1.*sin(theta1)];

% Cluster 2: Elongated horizontal ellipse (Bottom-right quadrant)
center2 = [12, -8];
theta2 = 2*pi*rand(points_per_cluster, 1);
r2 = sqrt(rand(points_per_cluster, 1));
a2 = 3.5; b2 = 1.2;  % Increased semi-axes for better visibility
cluster2 = [center2(1) + a2*r2.*cos(theta2), center2(2) + b2*r2.*sin(theta2)];

% Cluster 3: Crescent/banana shape (Top-left quadrant)
center3 = [-8, 12];
% Generate crescent by combining outer arc with selective inner exclusion
n_samples = points_per_cluster;
crescent_points = [];
attempts = 0;
max_attempts = 10;

while size(crescent_points, 1) < n_samples && attempts < max_attempts
    attempts = attempts + 1;
    % Generate points in a crescent-shaped region
    t3 = linspace(-pi/3, pi/3, n_samples * 2)' + 0.15*randn(n_samples * 2, 1);
    r3_outer = 2.8 + 0.2*randn(n_samples * 2, 1);
    
    % Outer arc points
    x3_temp = center3(1) + r3_outer .* cos(t3);
    y3_temp = center3(2) + r3_outer .* sin(t3);
    
    % Inner circle to exclude (offset to create crescent)
    x3_offset = center3(1) + 1.0;
    y3_offset = center3(2);
    inner_radius = 1.4;
    
    % Keep points outside inner circle
    dist_to_inner = sqrt((x3_temp - x3_offset).^2 + (y3_temp - y3_offset).^2);
    valid_mask = dist_to_inner > inner_radius;
    
    % Collect valid crescent points
    valid_points = [x3_temp(valid_mask), y3_temp(valid_mask)];
    crescent_points = [crescent_points; valid_points];
end

% Ensure exactly points_per_cluster samples
if size(crescent_points, 1) >= n_samples
    cluster3 = crescent_points(1:n_samples, :);
else
    % If still not enough, pad with additional arc points
    remaining = n_samples - size(crescent_points, 1);
    t3_extra = linspace(-pi/4, pi/4, remaining)' + 0.1*randn(remaining, 1);
    r3_extra = 2.5 + 0.3*randn(remaining, 1);
    extra_points = [center3(1) + r3_extra.*cos(t3_extra), center3(2) + r3_extra.*sin(t3_extra)];
    cluster3 = [crescent_points; extra_points];
end

% Cluster 4: S-curve/spiral shape (Top-right quadrant)
center4 = [12, 12];
t4 = linspace(0, 2*pi, points_per_cluster)' + 0.08*randn(points_per_cluster, 1);
r4 = 0.8 + 0.6*sin(3*t4) + 0.15*randn(points_per_cluster, 1);  % Increased spiral amplitude
cluster4 = [center4(1) + r4.*cos(t4), center4(2) + r4.*sin(t4)];

% Combine all clusters for View 1
points{1} = [cluster1; cluster2; cluster3; cluster4];
label = [ones(size(cluster1,1),1); 2*ones(size(cluster2,1),1); 
         3*ones(size(cluster3,1),1); 4*ones(size(cluster4,1),1)];



%% Generate View 2 - Four clusters with different distinct shapes (Expanded Format)

% Cluster 1: Diamond/rhombus shape (Bottom-left quadrant)
center1_v2 = [-8, -8];
t1_v2 = 2*pi*rand(points_per_cluster, 1);
r1_v2 = 1.2 + 0.7*abs(cos(4*t1_v2)) + 0.2*randn(points_per_cluster, 1);  % Increased diamond size
cluster1_v2 = [center1_v2(1) + r1_v2.*cos(t1_v2), center1_v2(2) + r1_v2.*sin(t1_v2)];

% Cluster 2: Ring/donut shape (Top-right quadrant)
center2_v2 = [12, 12];
theta2_v2 = 2*pi*rand(points_per_cluster, 1);
r2_inner = 1.8; r2_outer = 3.2;  % Increased ring dimensions
r2_v2 = r2_inner + (r2_outer - r2_inner)*rand(points_per_cluster, 1);
cluster2_v2 = [center2_v2(1) + r2_v2.*cos(theta2_v2), center2_v2(2) + r2_v2.*sin(theta2_v2)];

% Cluster 3: Cross/plus shape (Top-left quadrant)
center3_v2 = [-8, 12];
% Horizontal bar (expanded)
x3_h = center3_v2(1) + 4.5*(rand(points_per_cluster/2, 1) - 0.5);  % Increased width
y3_h = center3_v2(2) + 0.6*randn(points_per_cluster/2, 1);  % Increased thickness
% Vertical bar (expanded)
x3_v = center3_v2(1) + 0.6*randn(points_per_cluster/2, 1);  % Increased thickness
y3_v = center3_v2(2) + 4.5*(rand(points_per_cluster/2, 1) - 0.5);  % Increased height
cluster3_v2 = [x3_h, y3_h; x3_v, y3_v];

% Cluster 4: Heart shape (Bottom-right quadrant)
center4_v2 = [12, -8];
t4_v2 = linspace(0, 2*pi, points_per_cluster)' + 0.15*randn(points_per_cluster, 1);
% Parametric heart equation (scaled up)
scale = 0.7;  % Increased scale factor
x4_heart = center4_v2(1) + scale * (16*sin(t4_v2).^3);
y4_heart = center4_v2(2) + scale * (13*cos(t4_v2) - 5*cos(2*t4_v2) - 2*cos(3*t4_v2) - cos(4*t4_v2));
cluster4_v2 = [x4_heart + 0.2*randn(points_per_cluster, 1), y4_heart + 0.2*randn(points_per_cluster, 1)];  % Increased noise

% Combine all clusters for View 2
points{2} = [cluster1_v2; cluster2_v2; cluster3_v2; cluster4_v2];

%% Sample Count Verification
fprintf('=== SAMPLE COUNT VERIFICATION ===\n');
fprintf('Expected points per cluster: %d\n', points_per_cluster);
fprintf('Expected total points per view: %d\n\n', 4*points_per_cluster);

% View 1 Analysis
fprintf('VIEW 1 ANALYSIS:\n');
fprintf('Cluster 1 samples: %d\n', size(cluster1,1));
fprintf('Cluster 2 samples: %d\n', size(cluster2,1));
fprintf('Cluster 3 samples: %d\n', size(cluster3,1));
fprintf('Cluster 4 samples: %d\n', size(cluster4,1));
fprintf('Total View 1 samples: %d\n\n', size(points{1},1));

% View 2 Analysis
fprintf('VIEW 2 ANALYSIS:\n');
fprintf('Cluster 1 samples: %d\n', size(cluster1_v2,1));
fprintf('Cluster 2 samples: %d\n', size(cluster2_v2,1));
fprintf('Cluster 3 samples: %d\n', size(cluster3_v2,1));
fprintf('Cluster 4 samples: %d\n', size(cluster4_v2,1));
fprintf('Total View 2 samples: %d\n\n', size(points{2},1));

% Check for discrepancies
view1_total = size(points{1},1);
view2_total = size(points{2},1);
if view1_total == view2_total && view1_total == 4*points_per_cluster
    fprintf('✓ SUCCESS: Both views have equal sample counts (%d)\n', view1_total);
else
    fprintf('✗ ERROR: Sample count mismatch! View 1: %d, View 2: %d, Expected: %d\n', view1_total, view2_total, 4*points_per_cluster);
end

% Label Analysis
fprintf('\nLABEL ANALYSIS:\n');
for i = 1:4
    cluster_count = sum(label == i);
    fprintf('Cluster %d label count: %d\n', i, cluster_count);
end
fprintf('Total labels: %d\n', length(label));
fprintf('========================================\n\n');

clear cluster1 cluster2 cluster3 cluster4 cluster1_v2 cluster2_v2 cluster3_v2 cluster4_v2
clear center1 center2 center3 center4 center1_v2 center2_v2 center3_v2 center4_v2
clear theta1 theta2 t3 t4 theta2_v2 t1_v2 t4_v2 r1 r2 r3_outer r4 r1_v2 r2_v2
clear a2 b2 x3_h y3_h x3_v y3_v x4_heart y4_heart r2_inner r2_outer scale points_per_cluster 
clear n_samples crescent_points attempts max_attempts x3_temp y3_temp x3_offset y3_offset
clear inner_radius dist_to_inner valid_mask valid_points remaining t3_extra r3_extra extra_points 

clear label_ori mix mu obj points_n points_ori sigma temp_points
save data_distinct_shapes_expanded.mat


c=figure;

% Defaults for this blog post
width = 6;     % Width in inches (increased for expanded data)
height = 5;    % Height in inches (increased for expanded data)
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 6;       % MarkerSize (reduced for better visibility with more data points)

% Set Tick Marks for expanded range
set(gca,'XTick',-12:4:16);
set(gca,'YTick',-12:4:16);

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


% Extract clusters for visualization
a=find(label==1); a1=points{1}(a,:);
b=find(label==2); b1=points{1}(b,:);
c=find(label==3); c1=points{1}(c,:);
d=find(label==4); d1=points{1}(d,:);

% Plot with different markers to highlight distinct shapes
scatter(a1(:,1),a1(:,2),6,'b','o','filled');  % Cluster 1: Blue circles
hold on
scatter(b1(:,1),b1(:,2),6,'r','s','filled');  % Cluster 2: Red squares  
scatter(c1(:,1),c1(:,2),6,'m','^','filled');  % Cluster 3: Magenta triangles
scatter(d1(:,1),d1(:,2),6,'g','d','filled');  % Cluster 4: Green diamonds
hold off

% Add legend and title
legend('Cluster 1 (Circle)', 'Cluster 2 (H-Ellipse)', 'Cluster 3 (Crescent)', 'Cluster 4 (Spiral)', 'Location', 'best');
title('View 1: Multi-View Data with Distinct Cluster Shapes');
xlabel('x_{1}^{1}');
ylabel('x_{2}^{1}');
grid on
axis equal  % Maintain aspect ratio for shape clarity


c=figure;

% Defaults for this blog post
width = 6;     % Width in inches (increased for expanded data)
height = 5;    % Height in inches (increased for expanded data)
alw = 0.75;    % AxesLineWidth
fsz = 11;      % Fontsize
lw = 1.5;      % LineWidth
msz = 6;       % MarkerSize (reduced for better visibility)

% Set Tick Marks for expanded range
set(gca,'XTick',-12:4:16);
set(gca,'YTick',-12:4:16);

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


% Extract clusters for View 2 visualization
a=find(label==1); a2=points{2}(a,:);
b=find(label==2); b2=points{2}(b,:);
c=find(label==3); c2=points{2}(c,:);
d=find(label==4); d2=points{2}(d,:);

% Plot with different markers and colors for View 2
scatter(a2(:,1),a2(:,2),6,'b','o','filled');    % Cluster 1: Blue circles
hold on
scatter(b2(:,1),b2(:,2),6,'r','s','filled');    % Cluster 2: Red squares
scatter(c2(:,1),c2(:,2),6,'m','^','filled');    % Cluster 3: Magenta triangles  
scatter(d2(:,1),d2(:,2),6,'g','d','filled');    % Cluster 4: Green diamonds
hold off

% Add legend and title
legend('Cluster 1 (Diamond)', 'Cluster 2 (Ring)', 'Cluster 3 (Cross)', 'Cluster 4 (Heart)', 'Location', 'best');
title('View 2: Multi-View Data with Different Distinct Shapes');
xlabel('x_{1}^{2}');
ylabel('x_{2}^{2}');
grid on
axis equal  % Maintain aspect ratio for shape clarity

%% Partition
load data_distinct_shapes_expanded.mat
P1_view1= points{1}([1:8500],:);
P1_view2= points{2}([1:8500],:);
P1_label= label([1:8500]);
P2_view1= points{1}([8501:10000],:);
P2_view2= points{2}([8501:10000],:);
P2_label= label([8501:10000]);

% %% Another Partitions
% 
% load data_distinct_shapes_expanded.mat
% n=size(points{1},1);
% c=max(label);
% s = length(points);
% points = {points{1} points{2}};
% nclients = 2;
% rng(2);
% partition_num1 = random_partition(n,nclients);
% c_lients = c*ones(1,nclients);
% % Data on clients
% for i = 1:nclients
%     for h = 1:s
%         rng(2);
%         partition_num{h} = random_partition(n,nclients);
%     end
% end
% 
% dataview_set = cell(1, nclients);
% labelview_set = cell(1,nclients);
% for h = 1:s
%     stat_new{h} = 1;
%     for i = 1:nclients
%         temp_new{h} = points{h}(fix(stat_new{h}:stat_new{h}+partition_num{h}(i)-1),:);
%         stat_new{h} = stat_new{h} + partition_num{h}(i);
%         dataview_set{i}{h} = temp_new{h};
%     end
% end 
% 
% stat = 1;
% labelset = cell(1,nclients);
% for i = 1:nclients
%     temp_label = label(fix(stat:stat+partition_num1(i)-1),:);
%     stat = stat + partition_num1(i);
%     labelset{i} = temp_label;
% end
% P = length(dataview_set); % The number of clients
% 
% % The number of instances of MV data stored on clients devices
% for p = 1:P
%     for h = 1:s
%         n_clients(p) = size(dataview_set{p}{h}, 1);
%     end
% end
% 
% 
% % The ground truth of clusters stored on individual clients devices
% for p = 1:P
%     nClust_clients(p) = length(unique(labelset{p}))
% end
% 
% 
% c=figure;
% 
% % Defaults for this blog post
% width = 4;     % Width in inches
% height = 3;    % Height in inches
% alw = 0.75;    % AxesLineWidth
% fsz = 11;      % Fontsize
% lw = 1.5;      % LineWidth
% msz = 8;       % MarkerSize
% 
% % Set Tick Marks
% set(gca,'XTick',-3:3);
% set(gca,'YTick',0:10);
% 
% % Here we preserve the size of the image when we save it.
% set(gcf,'InvertHardcopy','on');
% set(gcf,'PaperUnits', 'inches');
% papersize = get(gcf, 'PaperSize');
% left = (papersize(1)- width)/2;
% bottom = (papersize(2)- height)/2;
% myfiguresize = [left, bottom, width, height];
% set(gcf,'PaperPosition', myfiguresize);
% 
% set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
% set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
% set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
% set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
% 
% 
% a=find(P1_label==1); a1=P1_view1(a,:);
% b=find(P1_label==2); b1=P1_view1(b,:);
% c=find(P1_label==3); c1=P1_view1(c,:);
% 
% 
% scatter(a1(:,1),a1(:,2),5,'bo');
% hold on
% scatter(b1(:,1),b1(:,2),5,'ro');
% scatter(c1(:,1),c1(:,2),5,'mo');
% hold off
% 
% 
% xlabel('x_{[1]1}^{1}');
% ylabel('x_{[1]2}^{1}');
% grid on
% 
% 
% c=figure;
% 
% % Defaults for this blog post
% width = 4;     % Width in inches
% height = 3;    % Height in inches
% alw = 0.75;    % AxesLineWidth
% fsz = 11;      % Fontsize
% lw = 1.5;      % LineWidth
% msz = 8;       % MarkerSize
% 
% % Set Tick Marks
% set(gca,'XTick',-3:3);
% set(gca,'YTick',0:10);
% 
% % Here we preserve the size of the image when we save it.
% set(gcf,'InvertHardcopy','on');
% set(gcf,'PaperUnits', 'inches');
% papersize = get(gcf, 'PaperSize');
% left = (papersize(1)- width)/2;
% bottom = (papersize(2)- height)/2;
% myfiguresize = [left, bottom, width, height];
% set(gcf,'PaperPosition', myfiguresize);
% 
% set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
% set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
% set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
% set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
% 
% 
% a=find(P1_label==1); a1=P1_view2(a,:);
% b=find(P1_label==2); b1=P1_view2(b,:);
% c=find(P1_label==3); c1=P1_view2(c,:);
% 
% scatter(a1(:,1),a1(:,2),5,'bo');
% hold on
% scatter(b1(:,1),b1(:,2),5,'ro');
% scatter(c1(:,1),c1(:,2),5,'mo');
% hold off
% 
% 
% xlabel('x_{[1]1}^{2}');
% ylabel('x_{[1]2}^{2}');
% grid on
% 
% 
% c=figure;
% 
% % Defaults for this blog post
% width = 4;     % Width in inches
% height = 3;    % Height in inches
% alw = 0.75;    % AxesLineWidth
% fsz = 11;      % Fontsize
% lw = 1.5;      % LineWidth
% msz = 8;       % MarkerSize
% 
% % Set Tick Marks
% set(gca,'XTick',-3:3);
% set(gca,'YTick',0:10);
% 
% % Here we preserve the size of the image when we save it.
% set(gcf,'InvertHardcopy','on');
% set(gcf,'PaperUnits', 'inches');
% papersize = get(gcf, 'PaperSize');
% left = (papersize(1)- width)/2;
% bottom = (papersize(2)- height)/2;
% myfiguresize = [left, bottom, width, height];
% set(gcf,'PaperPosition', myfiguresize);
% 
% set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
% set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
% set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
% set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
% 
% 
% a=find(P2_label==3); a1=P2_view1(a,:);
% b=find(P2_label==4); b1=P2_view1(b,:);
% 
% 
% scatter(a1(:,1),a1(:,2),5,'mo');
% hold on
% scatter(b1(:,1),b1(:,2),5,'go');
% 
% hold off
% 
% 
% xlabel('x_{[2]1}^{1}');
% ylabel('x_{[2]2}^{1}');
% grid on
% 
% 
% c=figure;
% 
% % Defaults for this blog post
% width = 4;     % Width in inches
% height = 3;    % Height in inches
% alw = 0.75;    % AxesLineWidth
% fsz = 11;      % Fontsize
% lw = 1.5;      % LineWidth
% msz = 8;       % MarkerSize
% 
% % Set Tick Marks
% set(gca,'XTick',-3:3);
% set(gca,'YTick',0:10);
% 
% % Here we preserve the size of the image when we save it.
% set(gcf,'InvertHardcopy','on');
% set(gcf,'PaperUnits', 'inches');
% papersize = get(gcf, 'PaperSize');
% left = (papersize(1)- width)/2;
% bottom = (papersize(2)- height)/2;
% myfiguresize = [left, bottom, width, height];
% set(gcf,'PaperPosition', myfiguresize);
% 
% set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
% set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
% set(0,'defaultLineLineWidth',lw);   % set the default line width to lw
% set(0,'defaultLineMarkerSize',msz); % set the default line marker size to msz
% 
% 
% a=find(P2_label==3); a1=P2_view2(a,:);
% b=find(P2_label==4); b1=P2_view2(b,:);
% 
% scatter(a1(:,1),a1(:,2),5,'m','*');
% hold on
% scatter(b1(:,1),b1(:,2),5,'g','mdiamond');
% hold off
% 
% 
% xlabel('x_{[2]1}^{2}');
% ylabel('x_{[2]2}^{2}');
% grid on
% 
% x = 0:1:10;
% y = sin(x);
% 
% plot(x,y,'-r')
% hold on
% for ii = 1:length(x)
%     text(x(ii),y(ii),num2str(ii),'Color','r')
% end