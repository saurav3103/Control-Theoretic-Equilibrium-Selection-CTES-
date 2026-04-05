`%% 3-Link Robot — CTES Framework (Static Analysis Focus)

clear; clc; close all;

%% ── 1. Parameters ─────────────────────────────────────────

l1=1; l2=1; l3=1;
m1=1; m2=1; m3=1;
g=9.81;

x_t = 1.5; y_t = 0.5;

w1=20; w2=1; w3=0.1;

fprintf('=== 3-Link CTES Framework ===\n');

%% Forward Kinematics
fk = @(q) [ ...
    l1*cos(q(1)) + l2*cos(q(1)+q(2)) + l3*cos(sum(q));
    l1*sin(q(1)) + l2*sin(q(1)+q(2)) + l3*sin(sum(q)) ];

%% Potential Energy
V = @(q) ...
    (m1+m2+m3)*g*l1*sin(q(1)) + ...
    (m2+m3)*g*l2*sin(q(1)+q(2)) + ...
    m3*g*l3*sin(sum(q));

%% Cost
f = @(q) ...
    w1*norm(fk(q)-[x_t;y_t])^2 + ...
    w2*V(q) + ...
    w3*norm(q)^2;

%% ── 2. Candidate Generation (IK) ─────────────────────────

N_samples = 60;
candidates = [];

for i=1:N_samples
    q = (rand(3,1)-0.5)*2*pi;

    for k=1:100
        J = jacobian_3link(q,l1,l2,l3);
        e = fk(q) - [x_t;y_t];

        lambda = 0.05;
        q = q - J' * ((J*J' + lambda*eye(2)) \ e);
    end

    if norm(fk(q)-[x_t;y_t]) < 1e-3
        candidates = [candidates, q];
    end
end

% Remove duplicates
candidates = uniquetol(candidates','ByRows',true)';
N = size(candidates,2);

fprintf('Candidates found: %d\n', N);

%% ── 3. CTES Energy Evaluation ───────────────────────────

f_vals = zeros(1,N);
energies = zeros(1,N);

for i=1:N
    f_vals(i) = f(candidates(:,i));
end

f_max = max(f_vals); f_min = min(f_vals);

for i=1:N
    q = candidates(:,i);
    H = numerical_hessian(f,q);

    if min(eig(H)) <= 0
        energies(i) = inf;
        continue
    end

    depth = (f_max - f_vals(i)) / (f_max - f_min + 1e-6);

    A = -H;
    B = eye(3);
    Q = max(depth*H,1e-8*eye(3));
    R = eye(3);

    P = care(A,B,Q,R);
    energies(i) = -trace(P);

    fprintf('Sol %d: f=%.3f depth=%.3f energy=%.4f\n',...
        i,f_vals(i),depth,energies(i));
end

[~,idx] = min(energies);
q_star = candidates(:,idx);

fprintf('\nSelected equilibrium:\n');
disp(q_star');

%% ── 4. STATIC VISUALIZATIONS ───────────────────────────

theta = linspace(0,2*pi,200);

% 1. Workspace
figure; hold on; grid on;
plot(3*cos(theta),3*sin(theta),'k--');

for i=1:N
    q = candidates(:,i);

    x1 = l1*cos(q(1)); y1 = l1*sin(q(1));
    x2 = x1 + l2*cos(q(1)+q(2)); y2 = y1 + l2*sin(q(1)+q(2));
    x3 = x2 + l3*cos(sum(q)); y3 = y2 + l3*sin(sum(q));

    if i == idx
        plot([0 x1 x2 x3],[0 y1 y2 y3],'g-','LineWidth',3);
    else
        plot([0 x1 x2 x3],[0 y1 y2 y3],'Color',[0.7 0.7 0.7]);
    end
end

plot(x_t,y_t,'r*','MarkerSize',15);
title('Workspace: All IK Solutions (Green = Selected)');
axis equal;

% 2. Joint Space Distribution
figure;
subplot(1,3,1)
scatter(candidates(1,:), candidates(2,:), 80, energies, 'filled');
xlabel('q1'); ylabel('q2'); grid on; colorbar;

subplot(1,3,2)
scatter(candidates(2,:), candidates(3,:), 80, energies, 'filled');
xlabel('q2'); ylabel('q3'); grid on; colorbar;

subplot(1,3,3)
scatter(candidates(1,:), candidates(3,:), 80, energies, 'filled');
xlabel('q1'); ylabel('q3'); grid on; colorbar;

sgtitle('Joint Space (Color = Energy)');

% 3. Energy vs Cost
figure;
scatter(energies,f_vals,120,'filled'); grid on;
xlabel('Energy'); ylabel('Cost');
title('Energy vs Cost');
hold on;
plot(energies(idx),f_vals(idx),'gp','MarkerSize',15,'LineWidth',2);

% 4. Depth vs Energy
depths = (f_max - f_vals) ./ (f_max - f_min + 1e-6);

figure;
scatter(depths, energies, 120, 'filled');
xlabel('Depth'); ylabel('Energy');
title('Depth vs Energy');
grid on;

% 5. Hessian Eigenvalues
eig_min = zeros(1,N);
eig_max = zeros(1,N);

for i=1:N
    H = numerical_hessian(f, candidates(:,i));
    e = eig(H);
    eig_min(i) = min(e);
    eig_max(i) = max(e);
end

figure;
subplot(1,2,1)
bar(eig_min); title('Min Eigenvalues'); grid on;

subplot(1,2,2)
bar(eig_max); title('Max Eigenvalues'); grid on;

% 6. Decision Bar Plot
figure;
b = bar(energies); hold on;

b.FaceColor = 'flat';
for i=1:N
    if i==idx
        b.CData(i,:) = [0 0.8 0];
    else
        b.CData(i,:) = [0.7 0.7 0.7];
    end
end

title('CTES Selection (Green = Selected)');
xlabel('Solution'); ylabel('Energy');
grid on;

%% ── 5. SINGLE SIMULATION (Animation Only) ──────────────

Kp = diag([40 40 40]);
Kd = diag([10 10 10]);

x0 = [0.2; -0.3; 0.5; 0; 0; 0];
tspan = [0 10];

[t,X] = ode45(@(t,x) dynamics(t,x,q_star,Kp,Kd,l1,l2,l3,m1,m2,m3,g), tspan, x0);

q_traj = X(:,1:3);

figure;
pause_time = 0.04;

for k=1:length(t)
    q = q_traj(k,:)';

    x1 = l1*cos(q(1)); y1 = l1*sin(q(1));
    x2 = x1 + l2*cos(q(1)+q(2)); y2 = y1 + l2*sin(q(1)+q(2));
    x3 = x2 + l3*cos(sum(q)); y3 = y2 + l3*sin(sum(q));

    cla;
    plot(3*cos(theta),3*sin(theta),'k--'); hold on;
    plot([0 x1 x2 x3],[0 y1 y2 y3],'o-','LineWidth',3);
    plot(x_t,y_t,'r*','MarkerSize',15);

    axis equal;
    xlim([-3 3]); ylim([-3 3]);
    grid on;

    title(sprintf('t=%.2f sec',t(k)));

    drawnow;
    pause(pause_time);
end

%% ── FUNCTIONS ───────────────────────────────────────────

function J = jacobian_3link(q,l1,l2,l3)
    s1 = sin(q(1)); c1 = cos(q(1));
    s12 = sin(q(1)+q(2)); c12 = cos(q(1)+q(2));
    s123 = sin(sum(q)); c123 = cos(sum(q));

    J = [ ...
        -l1*s1 - l2*s12 - l3*s123,  -l2*s12 - l3*s123,  -l3*s123;
         l1*c1 + l2*c12 + l3*c123,   l2*c12 + l3*c123,   l3*c123];
end

function H = numerical_hessian(f,q)
    h=1e-5;
    n=length(q);
    H=zeros(n);
    for i=1:n
        for j=1:n
            dq=zeros(n,1); dq2=dq;
            dq(i)=h; dq2(j)=h;
            H(i,j) = (f(q+dq+dq2)-f(q+dq-dq2)-f(q-dq+dq2)+f(q-dq-dq2))/(4*h^2);
        end
    end
end

function dx = dynamics(~,x,q_star,Kp,Kd,l1,l2,l3,m1,m2,m3,g)

    q = x(1:3);
    qd = x(4:6);

    M = eye(3);

    G = [(m1+m2+m3)*g*l1*cos(q(1));
         (m2+m3)*g*l2*cos(q(1)+q(2));
         m3*g*l3*cos(sum(q))];

    v = -Kp*(q - q_star) - Kd*qd;
    tau = M*v + G;

    qdd = M\(tau - G);

    dx = [qd; qdd];
end