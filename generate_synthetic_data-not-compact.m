%%   This code is provided (written and created) by Kristina P. Sinaga 
%%   Copyright (c) 2025 Kristina P. Sinaga 
%%   All rights reserved
%%   Contact :  krist.p.sinaga@gmail.com
%%   ------------------------------------------------------------------- %%
%%   Modified: Aug 13, 2025 - Enhanced to generate distinct cluster shapes
%%   Two-view, four-cluster multi-view data with different geometric shapes

clear all;close all;clc

%% Generate View 1 - Four clusters with distinct shapes
rng(45); 
points_per_cluster = 2500;  % 10000 total points / 4 clusters

% Cluster 1: Circular shape
center1 = [2, 2];
theta1 = 2*pi*rand(points_per_cluster, 1);
r1 = 0.5 * sqrt(rand(points_per_cluster, 1));
cluster1 = [center1(1) + r1.*cos(theta1), center1(2) + r1.*sin(theta1)];

% Cluster 2: Elongated horizontal ellipse  
center2 = [8, 2];
theta2 = 2*pi*rand(points_per_cluster, 1);
r2 = sqrt(rand(points_per_cluster, 1));
a2 = 1.5; b2 = 0.4;  % Semi-axes
cluster2 = [center2(1) + a2*r2.*cos(theta2), center2(2) + b2*r2.*sin(theta2)];

% Cluster 3: Crescent/banana shape
center3 = [2, 8];
t3 = linspace(-pi/3, pi/3, points_per_cluster)' + 0.1*randn(points_per_cluster, 1);
r3_outer = 1.2 + 0.1*randn(points_per_cluster, 1);
r3_inner = 0.6 + 0.1*randn(points_per_cluster, 1);
% Outer arc
x3_outer = center3(1) + r3_outer .* cos(t3);
y3_outer = center3(2) + r3_outer .* sin(t3);
% Inner arc (shifted to create crescent)
x3_inner = center3(1) + 0.4 + r3_inner .* cos(t3);
y3_inner = center3(2) + r3_inner .* sin(t3);
cluster3 = [x3_outer x3_inner];



% Cluster 4: S-curve/spiral shape
center4 = [8, 8];
t4 = linspace(0, 2*pi, points_per_cluster)' + 0.05*randn(points_per_cluster, 1);
r4 = 0.3 + 0.3*sin(3*t4) + 0.1*randn(points_per_cluster, 1);
cluster4 = [center4(1) + r4.*cos(t4), center4(2) + r4.*sin(t4)];

% Combine all clusters for View 1
points{1} = [cluster1; cluster2; cluster3; cluster4];
label = [ones(size(cluster1,1),1); 2*ones(size(cluster2,1),1); 
         3*ones(size(cluster3,1),1); 4*ones(size(cluster4,1),1)];



%% Generate View 2 - Four clusters with different distinct shapes

% Cluster 1: Diamond/rhombus shape
center1_v2 = [2, 2];
t1_v2 = 2*pi*rand(points_per_cluster, 1);
r1_v2 = 0.5 + 0.3*abs(cos(4*t1_v2)) + 0.1*randn(points_per_cluster, 1);  % Diamond pattern
cluster1_v2 = [center1_v2(1) + r1_v2.*cos(t1_v2), center1_v2(2) + r1_v2.*sin(t1_v2)];

% Cluster 2: Ring/donut shape
center2_v2 = [6, 6];
theta2_v2 = 2*pi*rand(points_per_cluster, 1);
r2_inner = 0.8; r2_outer = 1.3;
r2_v2 = r2_inner + (r2_outer - r2_inner)*rand(points_per_cluster, 1);
cluster2_v2 = [center2_v2(1) + r2_v2.*cos(theta2_v2), center2_v2(2) + r2_v2.*sin(theta2_v2)];

% Cluster 3: Cross/plus shape
center3_v2 = [6, -3];
% Horizontal bar
x3_h = center3_v2(1) + 2*(rand(points_per_cluster/2, 1) - 0.5);
y3_h = center3_v2(2) + 0.3*randn(points_per_cluster/2, 1);
% Vertical bar  
x3_v = center3_v2(1) + 0.3*randn(points_per_cluster/2, 1);
y3_v = center3_v2(2) + 2*(rand(points_per_cluster/2, 1) - 0.5);
cluster3_v2 = [x3_h, y3_h; x3_v, y3_v];

% Cluster 4: Heart shape
center4_v2 = [-2, -2];
t4_v2 = linspace(0, 2*pi, points_per_cluster)' + 0.1*randn(points_per_cluster, 1);
% Parametric heart equation
scale = 0.3;
x4_heart = center4_v2(1) + scale * (16*sin(t4_v2).^3);
y4_heart = center4_v2(2) + scale * (13*cos(t4_v2) - 5*cos(2*t4_v2) - 2*cos(3*t4_v2) - cos(4*t4_v2));
cluster4_v2 = [x4_heart + 0.1*randn(points_per_cluster, 1), y4_heart + 0.1*randn(points_per_cluster, 1)];

% Combine all clusters for View 2
points{2} = [cluster1_v2; cluster2_v2; cluster3_v2; cluster4_v2];

clear cluster1 cluster2 cluster3 cluster4 cluster1_v2 cluster2_v2 cluster3_v2 cluster4_v2
clear center1 center2 center3 center4 center1_v2 center2_v2 center3_v2 center4_v2
clear theta1 theta2 t3 t4 theta2_v2 t1_v2 t4_v2 r1 r2 r3_outer r3_inner r4 r1_v2 r2_v2
clear a2 b2 x3_outer y3_outer x3_inner y3_inner x3_h y3_h x3_v y3_v 
clear x4_heart y4_heart r2_inner r2_outer scale points_per_cluster 

clear label_ori mix mu obj points_n points_ori sigma temp_points
save data_distinct_shapes.mat

%% Save global indices and original labels for alignment
global_indices = (1:length(label))';
original_labels = label;
save('data_distinct_shapes.mat', 'points', 'label', 'global_indices', 'original_labels', '-append');

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


% Extract clusters for visualization
a=find(label==1); a1=points{1}(a,:);
b=find(label==2); b1=points{1}(b,:);
c=find(label==3); c1=points{1}(c,:);
d=find(label==4); d1=points{1}(d,:);

% Plot with different markers to highlight distinct shapes
scatter(a1(:,1),a1(:,2),8,'b','o','filled');  % Cluster 1: Blue circles
hold on
scatter(b1(:,1),b1(:,2),8,'r','s','filled');  % Cluster 2: Red squares  
scatter(c1(:,1),c1(:,2),8,'m','^','filled');  % Cluster 3: Magenta triangles
scatter(d1(:,1),d1(:,2),8,'g','d','filled');  % Cluster 4: Green diamonds
hold off

% Add legend and title
legend('Cluster 1 (Circle)', 'Cluster 2 (H-Ellipse)', 'Cluster 3 (Crescent)', 'Cluster 4 (Spiral)', 'Location', 'best');
title('View 1: Multi-View Data with Distinct Cluster Shapes');
xlabel('x_{1}^{1}');
ylabel('x_{2}^{1}');
grid on


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


% Extract clusters for View 2 visualization
a=find(label==1); a2=points{2}(a,:);
b=find(label==2); b2=points{2}(b,:);
c=find(label==3); c2=points{2}(c,:);
d=find(label==4); d2=points{2}(d,:);

% Plot with different markers and colors for View 2
scatter(a2(:,1),a2(:,2),8,'b','o','filled');    % Cluster 1: Blue circles
hold on
scatter(b2(:,1),b2(:,2),8,'r','s','filled');    % Cluster 2: Red squares
scatter(c2(:,1),c2(:,2),8,'m','^','filled');    % Cluster 3: Magenta triangles  
scatter(d2(:,1),d2(:,2),8,'g','d','filled');    % Cluster 4: Green diamonds
hold off

% Add legend and title
legend('Cluster 1 (Diamond)', 'Cluster 2 (Ring)', 'Cluster 3 (Cross)', 'Cluster 4 (Heart)', 'Location', 'best');
title('View 2: Multi-View Data with Different Distinct Shapes');
xlabel('x_{1}^{2}');
ylabel('x_{2}^{2}');
grid on

%% Partition
load data_distinct_shapes.mat
P1_view1= points{1}([1:8500],:);
P1_view2= points{2}([1:8500],:);
P1_label= label([1:8500]);
P2_view1= points{1}([8501:10000],:);
P2_view2= points{2}([8501:10000],:);
P2_label= label([8501:10000]);
