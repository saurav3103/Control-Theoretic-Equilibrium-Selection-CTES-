%% LQR Energy-Based Global Optimization - v7 (Rastrigin + Fix 3)
% Fix for Rastrigin failure in v6:
%   v6 used Q = |f(x*)| * H → fails when f(x*) > 0 everywhere
%   v7 uses Q = 1/(|f(x*) - f_ref| + ε) * H  (inverted encoding)
%
% Inversion logic: smaller f(x*) → larger Q → larger P → less negative -P
% Global minima (f≈0) gets largest Q, shallowest well gets smallest Q
%
% f_ref = 0 (Rastrigin global minima is exactly 0)
% ε = 0.1 (prevents Q→∞ at true global minima)

clear; clc; close all;

%% ── 1. Define Rastrigin function ─────────────────────────────────────────

A = 10;
f_scalar  = @(x) A + x^2 - A*cos(2*pi*x);
dfdx      = @(x) 2*x + 2*pi*A*sin(2*pi*x);
d2fdx2    = @(x) 2 + 4*pi^2*A*cos(2*pi*x);     % analytical Hessian

x_plot = linspace(-5, 5, 2000);

% Fix 3 parameters
f_ref = 0;      % reference level — global minima of Rastrigin is 0
epsilon = 0.1;  % prevents Q→∞ when f(x*) = f_ref

fprintf('=== Rastrigin Stress Test v7: Fix 3 (Inverted Q) ===\n');
fprintf('    Q = 1/(|f(x*) - f_ref| + ε) * H\n');
fprintf('    f_ref = %.1f,  ε = %.2f\n\n', f_ref, epsilon);

%% ── 2. Grid search + fzero refinement ───────────────────────────────────

grid_pts  = linspace(-5, 5, 500);    % fine grid needed for periodic function
grad_vals = arrayfun(dfdx, grid_pts);

% Sign changes from - to + = local minima
sign_changes = find(diff(sign(grad_vals)) > 0);
fprintf('Grid search found %d candidate minima regions\n', numel(sign_changes));

local_minima = zeros(1, numel(sign_changes));
valid = true(1, numel(sign_changes));
for i = 1:numel(sign_changes)
    xa = grid_pts(sign_changes(i));
    xb = grid_pts(sign_changes(i) + 1);
    try
        local_minima(i) = fzero(dfdx, [xa, xb]);
    catch
        valid(i) = false;
    end
end
local_minima = local_minima(valid);
local_minima = uniquetol(local_minima, 1e-4);
N = numel(local_minima);
fprintf('Refined to %d distinct local minima\n\n', N);

%% ── 3. LQR at each minima ────────────────────────────────────────────────

results = struct();
valid_count = 0;

fprintf('%-10s %-14s %-12s %-14s %-12s\n', ...
    'x*', 'f(x*)', 'Hessian', 'Q=1/(|f|+ε)*H', 'Energy(-P)');
fprintf('%s\n', repmat('-', 1, 66));

for i = 1:N
    x_star = local_minima(i);
    f_val  = f_scalar(x_star);
    H      = d2fdx2(x_star);

    % Skip if not a valid minima
    if H <= 0
        fprintf('  Skipping x*=%.4f: H=%.4f (not positive definite)\n', x_star, H);
        continue
    end

    A_sys = -H;
    B     = 1;
    % Fix 3: inverted encoding — smaller f(x*) → larger Q → larger P
    % Global minima (f≈f_ref) gets Q→1/ε * H (largest)
    % Shallow wells (f>>f_ref) get Q→0 (smallest)
    Q = (1 / (abs(f_val - f_ref) + epsilon)) * H;
    R = 1;

    % No special case needed — epsilon handles f(x*)=f_ref gracefully

    try
        P      = care(A_sys, B, Q, R);
        energy = -P;
    catch
        fprintf('  Skipping x*=%.4f: ARE failed\n', x_star);
        continue
    end

    valid_count = valid_count + 1;
    results(valid_count).x_star  = x_star;
    results(valid_count).f_val   = f_val;
    results(valid_count).hessian = H;
    results(valid_count).P       = P;
    results(valid_count).energy  = energy;

    fprintf('%-10.4f %-14.6f %-12.6f %-14.6f %-12.6f\n', ...
        x_star, f_val, H, (1/(abs(f_val-f_ref)+epsilon))*H, energy);
end
N = valid_count;

%% ── 4. Rank + Spearman correlation ──────────────────────────────────────

energies = [results.energy];
f_vals   = [results.f_val];
x_stars  = [results.x_star];

[~, energy_rank] = sort(energies);
[~, fval_rank]   = sort(f_vals);

fprintf('\n── Ranking by -P Energy ──\n');
for k = 1:N
    idx = energy_rank(k);
    fprintf('  Rank %d: x* = %7.4f,  f(x*) = %8.5f,  Energy = %9.6f\n', ...
        k, x_stars(idx), f_vals(idx), energies(idx));
end

fprintf('\n── Ranking by True f(x*) ──\n');
for k = 1:N
    idx = fval_rank(k);
    fprintf('  Rank %d: x* = %7.4f,  f(x*) = %8.5f,  Energy = %9.6f\n', ...
        k, x_stars(idx), f_vals(idx), energies(idx));
end

% Spearman rank correlation (manual)
[~, re] = sort(energies); rank_e = zeros(1,N); rank_e(re) = 1:N;
[~, rf] = sort(f_vals);   rank_f = zeros(1,N); rank_f(rf) = 1:N;
d2_sum  = sum((rank_e - rank_f).^2);
rho     = 1 - 6*d2_sum / (N*(N^2-1));

fprintf('\nSpearman Rank Correlation: %.4f\n', rho);

% Global minima check
[~, e_idx] = min(energies);
[~, f_idx] = min(f_vals);
fprintf('\n-P identifies global minima at:  x* = %.6f,  f = %.6f\n', ...
    x_stars(e_idx), f_vals(e_idx));
fprintf('True global minima at:           x* = %.6f,  f = %.6f\n', ...
    x_stars(f_idx), f_vals(f_idx));

if abs(x_stars(e_idx) - x_stars(f_idx)) < 0.1
    fprintf('✓ Correct global minima identified.\n');
else
    fprintf('✗ Wrong global minima identified.\n');
    fprintf('  Gap: |x_predicted - x_true| = %.4f\n', ...
        abs(x_stars(e_idx) - x_stars(f_idx)));
end

% Count ranking errors
mismatches = sum(energy_rank ~= fval_rank);
fprintf('\nRanking mismatches: %d / %d\n', mismatches, N);
if rho > 0.9
    fprintf('✓ Strong correlation — proxy tracks depth well.\n');
elseif rho > 0.5
    fprintf('~ Moderate correlation — partial support.\n');
else
    fprintf('✗ Weak correlation — proxy unreliable on Rastrigin.\n');
end

%% ── 5. Visualize ─────────────────────────────────────────────────────────

figure('Name','LQR v6 - Rastrigin','Position',[80 80 1200 600]);

% -- Top: Rastrigin + detected minima
subplot(2,3,[1 2 3]);
plot(x_plot, arrayfun(f_scalar, x_plot), 'b-', 'LineWidth', 2); hold on;
for i = 1:N
    if i == f_idx
        plot(results(i).x_star, results(i).f_val, 'k*', ...
            'MarkerSize', 16, 'LineWidth', 2.5);   % true global
    elseif i == e_idx
        plot(results(i).x_star, results(i).f_val, 'g^', ...
            'MarkerSize', 12, 'LineWidth', 2);      % predicted global
    else
        plot(results(i).x_star, results(i).f_val, 'ro', ...
            'MarkerSize', 8, 'LineWidth', 1.5);
    end
end
yline(0, 'k--', 'Alpha', 0.3);
xlabel('x'); ylabel('f(x)');
title(sprintf('Rastrigin: f(x) = 10 + x^2 - 10cos(2\\pix)   |   \\rho = %.3f   |   %d wells', ...
    rho, N), 'FontSize', 11);
legend('f(x)', 'True global (★)', 'Predicted global (▲)', 'Local minima', ...
    'Location', 'north');
grid on; xlim([-5 5]);

% -- Bottom left: f(x*) per well
subplot(2,3,4);
bar(1:N, f_vals(fval_rank), 'FaceColor', [0.3 0.6 0.9]);
xlabel('Rank (by true depth)'); ylabel('f(x*)');
title('True depth ranking');
grid on;

% -- Bottom middle: Energy per well (same order)
subplot(2,3,5);
% Reorder energies by true depth rank for comparison
energy_in_fval_order = energies(fval_rank);
bar(1:N, energy_in_fval_order, 'FaceColor', [0.9 0.5 0.2]);
xlabel('Rank (by true depth)'); ylabel('-P energy');
title('Energy in true-depth order (should be monotone ↓)');
grid on;

% -- Bottom right: Scatter
subplot(2,3,6);
scatter(energies, f_vals, 100, 'filled', 'MarkerFaceColor', [0.8 0.2 0.2]);
hold on;
p_fit = polyfit(energies, f_vals, 1);
x_fit = linspace(min(energies), max(energies), 100);
plot(x_fit, polyval(p_fit, x_fit), 'k--', 'LineWidth', 1.5);
% Mark global minima
scatter(energies(e_idx), f_vals(e_idx), 200, 'g^', 'filled');
scatter(energies(f_idx), f_vals(f_idx), 200, 'k*', 'LineWidth', 2);
xlabel('-P (energy proxy)'); ylabel('True f(x*)');
title(sprintf('Spearman \\rho = %.4f', rho));
grid on;

sgtitle('LQR Energy Hypothesis — v7: Rastrigin + Fix 3  Q = 1/(|f - f_{ref}| + \epsilon) \cdot H', ...
    'FontSize', 12, 'FontWeight', 'bold');