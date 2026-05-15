%% CTES Inverted Pendulum - Greedy LQR Equilibrium Selection
% Uses CTES framework (v8 philosophy) to select optimal physical parameters
% and Q/R ratio for tracking a 50 Hz square wave (Gibbs-disturbed) input.
%
% Greedy search order: m -> L (g fixed at 9.81)
% CTES proxy: -trace(P) from infinite-horizon LQR (ARE)
% Square wave: 50 Hz, amplitude ~2.24V (10 dBm into 50 ohm), 4 odd harmonics
%
% State: x = [theta; theta_dot]
% Linearized upright equilibrium: theta* = 0

clear; clc; close all;

%% ════════════════════════════════════════════════════════════════════════
%% PARAMETERS & GRIDS
%% ════════════════════════════════════════════════════════════════════════

g_fixed = 9.81;                          % m/s^2 fixed

m_grid  = [0.1, 0.3, 0.5, 0.7, 1.0];   % kg
L_grid  = [0.5, 0.75, 1.0, 1.5, 2.0];  % m
QR_grid = [0.1, 1, 10, 100, 1000];      % Q/R ratio (Q = ratio*I, R = 1)

% Square wave: 50 Hz, 10 dBm into 50 ohm -> P=10mW -> V=sqrt(P*R)=sqrt(0.01*50)
V_amp = sqrt(0.01 * 50);                 % ~2.2361 V
f0    = 50;                              % Hz fundamental
t_sim = 0.2;                             % seconds simulation window
dt    = 1e-5;                            % time step
t     = 0:dt:t_sim;

% Square wave via 4 odd Fourier harmonics (Gibbs included)
square_ref = zeros(size(t));
for k = 1:4
    n = 2*k - 1;   % 1,3,5,7 -> 50,150,250,350 Hz
    square_ref = square_ref + (4*V_amp/pi) * (1/n) * sin(2*pi*n*f0*t);
end

fprintf('════════════════════════════════════════════════════════════\n');
fprintf('  CTES INVERTED PENDULUM - Greedy LQR Selection\n');
fprintf('  Square wave: %.0f Hz, Amp=%.4f V, 4 harmonics\n', f0, V_amp);
fprintf('  CTES proxy: -trace(P)  |  Infinite horizon LQR\n');
fprintf('════════════════════════════════════════════════════════════\n\n');

%% ════════════════════════════════════════════════════════════════════════
%% GREEDY STAGE 1: Fix L=1.0m, sweep m, find best QR at each m
%% ════════════════════════════════════════════════════════════════════════

fprintf('─────────────────────────────────────────────\n');
fprintf('  STAGE 1: Sweep m  (L=1.0 m fixed)\n');
fprintf('─────────────────────────────────────────────\n');
fprintf('%-8s %-8s %-12s %-12s\n', 'm(kg)', 'QR', '-trace(P)', 'LQR Cost');
fprintf('%s\n', repmat('-', 1, 44));

L_fixed = 1.0;
best_m_energy = inf;
best_m = m_grid(1);
best_QR_stage1 = QR_grid(1);
stage1_results = struct('m',{},'L',{},'qr',{},'P',{},'energy',{},'cost',{});
s1_idx = 0;

for mi = 1:numel(m_grid)
    m = m_grid(mi);
    [A, B] = pendulum_linearize(m, L_fixed, g_fixed);

    best_e_for_m = inf;
    best_qr_for_m = QR_grid(1);

    for qi = 1:numel(QR_grid)
        qr = QR_grid(qi);
        Q  = qr * eye(2);
        R  = 1;

        try
            P = care(A, B, Q, R);
        catch
            continue
        end

        energy = -trace(P);   % CTES proxy: most negative = most robust

        % Simulate tracking cost
        K = lqr(A, B, Q, R);
        cost = simulate_lqr_cost(A, B, K, square_ref, dt, Q, R);

        s1_idx = s1_idx + 1;
        stage1_results(s1_idx).m      = m;
        stage1_results(s1_idx).L      = L_fixed;
        stage1_results(s1_idx).qr     = qr;
        stage1_results(s1_idx).P      = P;
        stage1_results(s1_idx).energy = energy;
        stage1_results(s1_idx).cost   = cost;

        fprintf('%-8.2f %-8.1f %-12.4f %-12.4f\n', m, qr, energy, cost);

        if energy < best_e_for_m   % min -trace(P) = most negative = most robust
            best_e_for_m  = energy;
            best_qr_for_m = qr;
        end
    end

    if best_e_for_m < best_m_energy
        best_m_energy = best_e_for_m;
        best_m         = m;
        best_QR_stage1 = best_qr_for_m;
    end
end

fprintf('\n  >> Stage 1 best: m=%.2f kg, QR=%.1f, CTES energy=%.4f\n\n', ...
    best_m, best_QR_stage1, best_m_energy);

%% ════════════════════════════════════════════════════════════════════════
%% GREEDY STAGE 2: Fix m=best_m, sweep L, find best QR at each L
%% ════════════════════════════════════════════════════════════════════════

fprintf('─────────────────────────────────────────────\n');
fprintf('  STAGE 2: Sweep L  (m=%.2f kg fixed)\n', best_m);
fprintf('─────────────────────────────────────────────\n');
fprintf('%-8s %-8s %-12s %-12s\n', 'L(m)', 'QR', '-trace(P)', 'LQR Cost');
fprintf('%s\n', repmat('-', 1, 44));

best_L_energy = inf;
best_L  = L_grid(1);
best_QR_final = best_QR_stage1;
stage2_results = struct('m',{},'L',{},'qr',{},'P',{},'energy',{},'cost',{});
s2_idx = 0;

for li = 1:numel(L_grid)
    L = L_grid(li);
    [A, B] = pendulum_linearize(best_m, L, g_fixed);

    best_e_for_L = inf;
    best_qr_for_L = QR_grid(1);

    for qi = 1:numel(QR_grid)
        qr = QR_grid(qi);
        Q  = qr * eye(2);
        R  = 1;

        try
            P = care(A, B, Q, R);
        catch
            continue
        end

        energy = -trace(P);
        K = lqr(A, B, Q, R);
        cost = simulate_lqr_cost(A, B, K, square_ref, dt, Q, R);

        s2_idx = s2_idx + 1;
        stage2_results(s2_idx).m      = best_m;
        stage2_results(s2_idx).L      = L;
        stage2_results(s2_idx).qr     = qr;
        stage2_results(s2_idx).P      = P;
        stage2_results(s2_idx).energy = energy;
        stage2_results(s2_idx).cost   = cost;

        fprintf('%-8.3f %-8.1f %-12.4f %-12.4f\n', L, qr, energy, cost);

        if energy < best_e_for_L   % min -trace(P) = most negative = most robust
            best_e_for_L  = energy;
            best_qr_for_L = qr;
        end
    end

    if best_e_for_L < best_L_energy
        best_L_energy = best_e_for_L;
        best_L         = L;
        best_QR_final  = best_qr_for_L;
    end
end

fprintf('\n  >> Stage 2 best: L=%.3f m, QR=%.1f, CTES energy=%.4f\n\n', ...
    best_L, best_QR_final, best_L_energy);

%% ════════════════════════════════════════════════════════════════════════
%% FINAL CONFIGURATION
%% ════════════════════════════════════════════════════════════════════════

m_opt  = best_m;
L_opt  = best_L;
QR_opt = best_QR_final;
Q_opt  = QR_opt * eye(2);
R_opt  = 1;

[A_opt, B_opt] = pendulum_linearize(m_opt, L_opt, g_fixed);
P_opt  = care(A_opt, B_opt, Q_opt, R_opt);
K_opt  = lqr(A_opt, B_opt, Q_opt, R_opt);

fprintf('════════════════════════════════════════════════════════════\n');
fprintf('  OPTIMAL CONFIGURATION (CTES Selected)\n');
fprintf('  m    = %.2f kg\n',   m_opt);
fprintf('  L    = %.3f m\n',    L_opt);
fprintf('  g    = %.2f m/s²\n', g_fixed);
fprintf('  Q/R  = %.1f\n',      QR_opt);
fprintf('  K    = [%.4f  %.4f]\n', K_opt(1), K_opt(2));
fprintf('  -trace(P) = %.4f  (CTES energy)\n', -trace(P_opt));
fprintf('  Eigenvalues of (A-BK): %.4f, %.4f\n', ...
    eig(A_opt - B_opt*K_opt));
fprintf('════════════════════════════════════════════════════════════\n\n');

%% ════════════════════════════════════════════════════════════════════════
%% SIMULATION: Tracking square wave with optimal config
%% ════════════════════════════════════════════════════════════════════════

N   = numel(t);
x   = zeros(2, N);
u   = zeros(1, N);
x(:,1) = [0.01; 0];   % small initial perturbation from upright

for k = 1:N-1
    e       = square_ref(k) - x(1,k);   % tracking error on theta
    u(k)    = -K_opt * x(:,k) + QR_opt * e;  % LQR + feedforward gain
    x(:,k+1) = x(:,k) + dt * (A_opt*x(:,k) + B_opt*u(k));
end
u(N) = u(N-1);

% Infinite horizon cost accumulation
running_cost = cumsum(dt * (diag(x' * Q_opt * x)' + R_opt * u.^2));

%% ════════════════════════════════════════════════════════════════════════
%% VISUALIZATION
%% ════════════════════════════════════════════════════════════════════════

all_m   = [stage1_results.m];
all_e1  = [stage1_results.energy];
u_m     = unique(all_m);
best_e_per_m = arrayfun(@(mv) min(all_e1(all_m==mv)), u_m);
[~,bmi] = min(best_e_per_m);

all_L   = [stage2_results.L];
all_e2  = [stage2_results.energy];
u_L     = unique(all_L);
best_e_per_L = arrayfun(@(lv) min(all_e2(all_L==lv)), u_L);
[~,bli] = min(best_e_per_L);

s2_bestL = stage2_results([stage2_results.L] == best_L);
qr_vals  = [s2_bestL.qr];
e_vals   = [s2_bestL.energy];
[~,qi_best] = min(e_vals);

% 1. Stage 1: CTES Energy vs m
figure; hold on; grid on;
b1 = bar(u_m, best_e_per_m, 0.5);
b1.FaceColor = 'flat';
for i = 1:numel(u_m)
    b1.CData(i,:) = [0.7 0.7 0.7];
end
b1.CData(bmi,:) = [0 0.8 0];
xlabel('m (kg)'); ylabel('-trace(P)');
title('Stage 1: CTES Energy vs m (Green = Selected)');

% 2. Stage 2: CTES Energy vs L
figure; hold on; grid on;
b2 = bar(u_L, best_e_per_L, 0.5);
b2.FaceColor = 'flat';
for i = 1:numel(u_L)
    b2.CData(i,:) = [0.7 0.7 0.7];
end
b2.CData(bli,:) = [0 0.8 0];
xlabel('L (m)'); ylabel('-trace(P)');
title('Stage 2: CTES Energy vs L (Green = Selected)');

% 3. CTES Energy vs Q/R (best L)
figure; hold on; grid on;
plot(qr_vals, e_vals, 'o-', 'LineWidth', 2);
plot(qr_vals(qi_best), e_vals(qi_best), 'gp', 'MarkerSize', 15, 'LineWidth', 2);
set(gca, 'XScale', 'log');
xlabel('Q/R ratio'); ylabel('-trace(P)');
title('CTES Energy vs Q/R (Green = Selected)');

% 4. Square wave reference
figure; hold on; grid on;
plot(t*1000, square_ref, 'LineWidth', 1.5);
xlabel('Time (ms)'); ylabel('Amplitude (V)');
title('Reference: 50 Hz Square Wave (4 Harmonics + Gibbs)');
xlim([0 t_sim*1000]);

% 5. Theta tracking
figure; hold on; grid on;
plot(t*1000, square_ref, '--', 'LineWidth', 1.2);
plot(t*1000, x(1,:), 'LineWidth', 1.5);
xlabel('Time (ms)'); ylabel('\theta (rad)');
title(sprintf('Tracking: m=%.1f kg, L=%.2f m, QR=%.1f', m_opt, L_opt, QR_opt));
legend({'\theta_{ref}', '\theta(t)'}, 'Location', 'best');
xlim([0 t_sim*1000]);

% 6. Angular velocity
figure; hold on; grid on;
plot(t*1000, x(2,:), 'LineWidth', 1.5);
xlabel('Time (ms)'); ylabel('\theta_{dot} (rad/s)');
title('Angular Velocity');
xlim([0 t_sim*1000]);

% 7. Control effort
figure; hold on; grid on;
plot(t*1000, u, 'LineWidth', 1.5);
xlabel('Time (ms)'); ylabel('u(t) (N·m)');
title('Control Effort (Torque)');
xlim([0 t_sim*1000]);

% 8. Tracking error
figure; hold on; grid on;
plot(t*1000, square_ref - x(1,:), 'LineWidth', 1.5);
xlabel('Time (ms)'); ylabel('e(t) = r - \theta');
title('Tracking Error');
xlim([0 t_sim*1000]);

% 9. Cumulative LQR cost
figure; hold on; grid on;
plot(t*1000, running_cost, 'r-', 'LineWidth', 1.5);
xlabel('Time (ms)'); ylabel('J(t)');
title(sprintf('Cumulative LQR Cost   J_{final} = %.4f', running_cost(end)));
xlim([0 t_sim*1000]);

%% ════════════════════════════════════════════════════════════════════════
%% LOCAL FUNCTIONS
%% ════════════════════════════════════════════════════════════════════════

function [A, B] = pendulum_linearize(m, L, g)
    % Linearized inverted pendulum at upright equilibrium (theta=0)
    % State: [theta; theta_dot]
    % x_dot = A*x + B*u
    % A = [0 1; g/L 0]  (unstable: positive g/L term)
    % B = [0; 1/(m*L^2)]
    A = [0, 1; g/L, 0];
    B = [0; 1/(m * L^2)];
end

function cost = simulate_lqr_cost(A, B, K, ref, dt, Q, R)
    % Simulate LQR tracking and return total infinite-horizon cost
    N = numel(ref);
    x = zeros(2, 1);
    x(1) = 0.01;   % small perturbation
    cost = 0;
    for k = 1:N-1
        e    = ref(k) - x(1);
        u    = -K * x + (K(1) + 0.1) * e;   % proportional feedforward
        cost = cost + dt * (x' * Q * x + R * u^2);
        x    = x + dt * (A*x + B*u);
        if norm(x) > 1e4; cost = 1e10; return; end  % diverged
    end
end
