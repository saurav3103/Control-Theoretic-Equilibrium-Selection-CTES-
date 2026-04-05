%% LQR Energy-Based Global Optimization - v4 (Flipped Proxy: -P)
% Fix from v3: P was perfectly correlated with depth but inverted (rho=-1).
% Solution: use -P as proxy so lower energy = deeper well (rho -> +1)
% and global minima = minimum energy, consistent with original intuition.
%
% Test function: product of shifted quadratics, scaled to control depth
%   f(x) = 0.05*(x+4)*(x+2)*(x)*(x-2.5)*(x-4.5) + depth_shaping_term
%
% Wells are at approximately x = {-3.2, -1.1, 1.2, 3.4}
% Depths are deliberately non-monotone in curvature to stress-test P/H

clear; clc; close all;

%% ── 1. Define 5-well test function ──────────────────────────────────────

% Constructed as sum of Gaussians negated (creates controllable wells)
% Each well: position mu_i, depth d_i, width sigma_i
% f(x) = -sum(d_i * exp(-(x-mu_i)^2 / (2*sigma_i^2))) + baseline

mu    = [-4.0, -2.0,  0.5,  2.5,  4.5];   % well centers
depth = [ 1.2,  0.8,  2.0,  1.5,  0.6];   % well depths (2.0 is global)
sigma = [ 0.6,  0.4,  0.5,  0.7,  0.3];   % well widths (controls curvature)

% Note: deeper well (mu=0.5) has medium width — not the sharpest
% This is the stress: global minima is NOT the stiffest well

f = @(x) -sum(depth .* exp(-(x - mu).^2 ./ (2*sigma.^2)), 2)';

% Scalar version for single x input
f_scalar = @(x) -sum(depth .* exp(-(x - mu).^2 ./ (2*sigma.^2)));

% Numerical derivatives
dx     = 1e-6;
dfdx   = @(x) (f_scalar(x + dx) - f_scalar(x - dx)) / (2*dx);
d2fdx2 = @(x) (f_scalar(x + dx) - 2*f_scalar(x) + f_scalar(x - dx)) / dx^2;

x_plot = linspace(-6, 6, 1000);

%% ── 2. Grid search + fzero refinement ───────────────────────────────────

grid_pts  = linspace(-5.5, 5.5, 300);   % finer grid for 5 wells
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
local_minima = uniquetol(local_minima, 1e-3);
N = numel(local_minima);
fprintf('Refined to %d distinct local minima\n\n', N);

%% ── 3. Linearize + LQR at each minima ───────────────────────────────────

results = struct();
valid_count = 0;

fprintf('%-8s %-14s %-12s %-12s %-14s\n', ...
    'x*', 'f(x*)', 'Hessian', 'Q=|f|*H', 'Energy (-P)');
fprintf('%s\n', repmat('-', 1, 76));

for i = 1:N
    x_star = local_minima(i);
    f_val  = f_scalar(x_star);
    H      = d2fdx2(x_star);

    A = -H;
    B = 1;
    % Depth-encoded Q: tells LQR to penalize deviation more at deeper wells
    % |f(x*)| is the depth — deeper well → larger Q → larger P
    Q = abs(f_val) * H;
    R = 1;

    if A >= 0 || H <= 0
        fprintf('  Skipping x*=%.4f: not a valid minima (H=%.4f)\n', x_star, H);
        continue
    end

    P = care(A, B, Q, R);

    % Energy proxy: -P so that deeper well = lower energy = global minima
    % v3 showed P has rho=-1 (perfect but inverted), -P gives rho=+1
    energy = -P;

    valid_count = valid_count + 1;
    results(valid_count).x_star  = x_star;
    results(valid_count).f_val   = f_val;
    results(valid_count).hessian = H;
    results(valid_count).P       = P;
    results(valid_count).energy  = energy;

    fprintf('%-8.4f %-14.6f %-12.6f %-12.6f %-14.6f\n', ...
        x_star, f_val, H, abs(f_val)*H, P);
end
N = valid_count;

%% ── 4. Rank + correlation analysis ──────────────────────────────────────

energies = [results.energy];
f_vals   = [results.f_val];
x_stars  = [results.x_star];
hessians = [results.hessian];

[~, energy_rank] = sort(energies);
[~, fval_rank]   = sort(f_vals);

fprintf('\n── Ranking by -P Energy (v4) ──\n');
for k = 1:N
    idx = energy_rank(k);
    fprintf('  Rank %d: x* = %6.3f,  f(x*) = %8.5f,  P = %8.6f\n', ...
        k, x_stars(idx), f_vals(idx), energies(idx));
end

fprintf('\n── Ranking by True f(x*) ──\n');
for k = 1:N
    idx = fval_rank(k);
    fprintf('  Rank %d: x* = %6.3f,  f(x*) = %8.5f,  P = %8.6f\n', ...
        k, x_stars(idx), f_vals(idx), energies(idx));
end

% Spearman rank correlation — manual implementation (no toolbox needed)
% +1 = perfect agreement, -1 = perfect reversal, 0 = no correlation
[~, re] = sort(energies);  rank_e = zeros(1,N); rank_e(re) = 1:N;
[~, rf] = sort(f_vals);    rank_f = zeros(1,N); rank_f(rf) = 1:N;
d2  = sum((rank_e - rank_f).^2);
rho = 1 - 6*d2 / (N*(N^2 - 1));
fprintf('\nSpearman Rank Correlation (Energy vs Depth): %.4f\n', rho);
fprintf('(+1 = perfect, -1 = reversed, 0 = no correlation)\n');

if isequal(energy_rank, fval_rank)
    fprintf('\n✓ HYPOTHESIS SUPPORTED: P/H ranking matches true ranking.\n');
elseif rho < 0
    fprintf('\n✓ DIRECTIONALLY CORRECT: Negative correlation — lower P/H = deeper well.\n');
    fprintf('  But ranking not perfect. Partial support.\n');
else
    fprintf('\n✗ HYPOTHESIS VIOLATED: P/H does not track depth.\n');
end

% Global minima identified
[~, e_idx] = min(energies);
[~, f_idx] = min(f_vals);
fprintf('\nP/H identifies global minima at:  x* = %.4f\n', x_stars(e_idx));
fprintf('True global minima at:             x* = %.4f\n', x_stars(f_idx));
if e_idx == f_idx
    fprintf('✓ Correct global minima identified.\n');
else
    fprintf('✗ Wrong global minima identified.\n');
end

%% ── 5. Visualize ─────────────────────────────────────────────────────────

figure('Name','LQR v2 - 5 Well Stress Test','Position',[80 80 1200 700]);

colors_list = {'ro','gs','md','c^','bh'};

% -- Top: Function + detected minima
subplot(2,3,[1 2 3]);
plot(x_plot, arrayfun(f_scalar, x_plot), 'b-', 'LineWidth', 2); hold on;
for i = 1:N
    plot(results(i).x_star, results(i).f_val, colors_list{mod(i-1,5)+1}, ...
        'MarkerSize', 12, 'LineWidth', 2);
    text(results(i).x_star, results(i).f_val - 0.12, ...
        sprintf('x*_%d\n%.2f', i, results(i).x_star), ...
        'HorizontalAlignment','center', 'FontSize', 8, 'FontWeight','bold');
end
% Mark true global minima
[~, gi] = min(f_vals);
plot(x_stars(gi), f_vals(gi), 'k*', 'MarkerSize', 18, 'LineWidth', 2);
yline(0,'k--','Alpha',0.2);
xlabel('x'); ylabel('f(x)');
title('5-Well Test Function — Detected Minima (★ = true global)', 'FontSize', 11);
grid on; xlim([-6 6]);

% -- Bottom left: Hessian bar chart
subplot(2,3,4);
b = bar(1:N, hessians, 'FaceColor','flat');
b.CData = repmat([0.3 0.6 0.9], N, 1);
b.CData(gi,:) = [0.2 0.7 0.2];   % highlight true global minima in green
xlabel('Minima'); ylabel('Hessian (curvature)');
title('Hessian at each minima');
xticklabels(arrayfun(@(i) sprintf('x*_%d',i), 1:N, 'UniformOutput',false));
grid on;

% -- Bottom middle: P/H bar chart
subplot(2,3,5);
[~, ei] = min(energies);
b2 = bar(1:N, energies, 'FaceColor','flat');
b2.CData = repmat([0.9 0.5 0.2], N, 1);
b2.CData(ei,:) = [0.2 0.7 0.2];   % highlight energy-predicted global in green
xlabel('Minima'); ylabel('-P (energy proxy)');
title('-P Energy — Green = predicted global');
xticklabels(arrayfun(@(i) sprintf('x*_%d',i), 1:N, 'UniformOutput',false));
grid on;

% -- Bottom right: Scatter energy vs depth
subplot(2,3,6);
scatter(energies, f_vals, 140, 'filled', 'MarkerFaceColor',[0.8 0.2 0.2]);
for i = 1:N
    text(energies(i), f_vals(i) - 0.04, sprintf('x*_%d',i), ...
        'FontSize', 8, 'HorizontalAlignment','center');
end
% Fit line
p = polyfit(energies, f_vals, 1);
x_fit = linspace(min(energies), max(energies), 100);
plot(x_fit, polyval(p, x_fit), 'r--', 'LineWidth', 1.5);
xlabel('-P (depth-encoded energy)'); ylabel('True f(x*)');
title(sprintf('Correlation: Spearman \\rho = %.3f', rho));
grid on;

sgtitle('LQR Energy Hypothesis — v4: Proxy = -P,  Q = |f(x*)| \times H', ...
    'FontSize', 13, 'FontWeight', 'bold');