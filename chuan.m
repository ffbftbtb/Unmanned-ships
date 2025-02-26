clc; clear all; close all;
%% 系统参数初始化
M = [6020.3   0       0       0       332.7269 0;
     0        9548.7  0       -380.0698 0      -473.4289;
     0        0       23320   0       2682.8   0;
     0        -380.0698 0     4129.0   13.58   84.6732;
     332.7269 0       2682.8  13.58    49134   -13.58;
     0        -473.4289 0     84.6732 -13.58   20696]; % 惯性矩阵
D = diag([0.5434, 1436.5, 4213.8, 4339.8, 26829, 6313.5]); % 阻尼矩阵
C = [0, 0, 0, 1.0541, 6.4737, 11.9362;
     0, 0, 0, -6.4737, 1.0541, -85.9782;
     0, 0, 0, -12.1241, 88.1113, 0;
     -1.0541, 6.4737, 12.1241, 0, 15.7173, -138.4340;
     -6.4737, -1.0541, -88.1113, -15.7173, 0, -0.7677;
     -11.9362, 85.9782, 0, 138.4340, 0.7677, 0]; % 科里奥利矩阵
%% 神经网络参数初始化
n_hidden = 20;                  % 隐藏层节点数（增加节点提升逼近能力）
input_dim = 6;                  % 输入维度（h_tilde是6x1）
W_hat = zeros(n_hidden, 6);     % 权重矩阵：20x6
gamma = 0.005;                  % 学习率（减小以避免振荡）
sigma = 5.0;                    % RBF宽度参数（扩大覆盖范围）
mu_range = [-10, 10];           % RBF中心初始化范围
mu = rand(input_dim, n_hidden) * (mu_range(2)-mu_range(1)) + mu_range(1); % 均匀分布初始化

%% 初始状态与参考轨迹
h0 = [10; 30; -3; 0; 0; 0];     % 初始位置和姿态
nu0 = zeros(6,1);               % 初始速度
% hd = [20; 40; 0; 0; 0; 0];      % 期望位置和姿态
% hd_dot = zeros(6,1);            % 期望速度
% hd_ddot = zeros(6,1);           % 期望加速度
  
%% 控制参数
a = 1;          % 滑模面参数
T = 1;          % 预定义时间
k = 50;         % 控制增益（增大以加快收敛）
epsilon = 1e-6; % 避免除以零的小量
c = 1.0;        % 自适应律参数
d = 0.5;        % 自适应律参数
T2 = 1.0;       % 权重更新时间常数

%% 仿真参数
tspan = [0 20];  % 缩短仿真时间至20秒（更快观察稳态）
dt = 0.01;       % 时间步长
t = tspan(1):dt:tspan(2);
N = length(t);

%% 初始化变量
h = h0;
nu = nu0;
h_traj = zeros(6, N);
nu_traj = zeros(6, N);
W_norm_history = zeros(1, N);   % 记录权重范数

%% 变换矩阵初始化（动态更新）
psi = 0; % 初始偏航角（假设为0）
J = [cos(psi), -sin(psi), 0, 0, 0, 0;
     sin(psi),  cos(psi), 0, 0, 0, 0;
     0,         0,        1, 0, 0, 0;
     0,         0,        0, 1, 0, 0;
     0,         0,        0, 0, 1, 0;
     0,         0,        0, 0, 0, 1]; % 初始J矩阵
J_prev = J;
J_dot = zeros(6); % J的导数（需动态计算）

%% 主循环（修正后）
for i = 1:N
    hd = [20 + 5*sin(0.1*t(i)); 
        40 + 3*cos(0.1*t(i));
        -3 + 0.5*sin(0.2*t(i));
        0.1*sin(0.1*t(i));
        0.05*cos(0.1*t(i));
        0.2*t(i)]; % 时变6-DOF轨迹-期望位置和姿态
    hd_dot =[0.5*cos(0.1*t(i)); 
        -0.3*sin(0.1*t(i));
        0.1*cos(0.2*t(i));
        0.01*cos(0.1*t(i));
        -0.005*sin(0.1*t(i));
        0.2]; % 期望速度
    hd_ddot =[-0.05*sin(0.1*t(i)); 
        -0.03*cos(0.1*t(i));
        -0.02*sin(0.2*t(i));
        -0.001*sin(0.1*t(i));
        -0.0005*cos(0.1*t(i));
        0]; % 期望加速度
    % 跟踪误差计算
    h_tilde = h - hd;
    h_tilde_dot = nu - hd_dot;
    % --- 动态更新变换矩阵J ---
    psi = 0.1 * t(i);
    J = [cos(psi), -sin(psi), 0, 0, 0, 0;
         sin(psi),  cos(psi), 0, 0, 0, 0;
         0,         0,        1, 0, 0, 0;
         0,         0,        0, 1, 0, 0;
         0,         0,        0, 0, 1, 0;
         0,         0,        0, 0, 0, 1];
    J_dot = (J - J_prev) / dt;
    J_prev = J;
    
    % --- 滑模面计算（公式19）---
    norm_h_tilde = norm(h_tilde) + epsilon;
    s = h_tilde_dot + (h_tilde / (a * T)) * exp(norm_h_tilde^a) / norm_h_tilde^a;
    
    % --- 神经网络扰动估计 ---
    z = h_tilde;
    z = z / (norm(z) + epsilon);
    diff = z - mu;
    h_rbf = exp(-sum(diff.^2, 1)' / (2*sigma^2));
    tau_d_hat = W_hat' * h_rbf;
    
    % --- 权重更新（公式29）---
    W_norm = norm(W_hat, 'fro');
    term1 = gamma * c * norm(s)^(c-2) * h_rbf * s';
    term2 = (1/(d*T2)) * (W_norm^(1-d)) * exp(min(W_norm^d, 10)) * W_hat;
    W_hat = W_hat - term1 * dt + term2 * dt;
    
    % --- 控制输入计算（公式21）---
    inv_J = inv(J);
    
    % term1_control: 6×1 向量
    term1_control = (C * inv_J + D * inv_J - M * inv_J * (J_dot * inv_J)) * nu;
    
    % term2_control: 6×1 向量
    term2_control = M * inv_J * hd_ddot;
    
    % term3_control: 6×1 向量
    term3_control = M * inv_J * k * sign(s);
    
    % term4_control: 6×1 向量
    term4_part1 = (a * (h_tilde * h_tilde') / norm_h_tilde^2) * h_tilde_dot;
    term4_part2 = (eye(6) - a * (h_tilde * h_tilde') / norm_h_tilde^2) * ...
                  (1 / (a * T * norm_h_tilde^a)) * exp(norm_h_tilde^a) * h_tilde_dot;
    term4_control = M * inv_J * (term4_part1 + term4_part2);
    
    % tau: 6×1 向量
    tau = term1_control + term2_control - term3_control - term4_control - tau_d_hat;

    % --- 系统动力学更新 ---
    h_dot = J * nu;          % 6×1 向量
    nu_dot = inv(M) * (tau - C * nu - D * nu + tau_d_hat); % 6×1 向量
    h = h + h_dot * dt;      % 6×1 向量
    nu = nu + nu_dot * dt;   % 6×1 向量

    % 存储轨迹
    h_traj(:, i) = h;
    nu_traj(:, i) = nu;      % 确保nu是6×1的列向量
end
%% 绘图与诊断
figure;
subplot(3,1,1);
plot(t, h_traj(1:3,:));
legend('Surge', 'Sway', 'Heave');
title('位置跟踪');

subplot(3,1,2);
plot(t, h_traj(4:6,:));
legend('Roll', 'Pitch', 'Yaw');
title('姿态跟踪');

subplot(3,1,3);
plot(t, W_norm_history);
title('权重矩阵范数');
xlabel('时间 (s)');

figure;
plot(t, nu_traj(1:3,:));
legend('Surge速度', 'Sway速度', 'Heave速度');
title('速度响应');
