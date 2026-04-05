%% LQR Energy-Based Global Optimization - v5 (2D Extension)
% Extends v4 to 2D: f(x1, x2) — a scalar function of a 2D state.
%
% Key changes from v4 (1D):
%   x      : vector [x1, x2]       (was scalar)
%   H      : 2x2 Hessian matrix    (was scalar d2f/dx2)
%   A = -H : 2x2 matrix            (was scalar)
%   B = I2 : 2x2 identity          (was scalar 1)
%   Q = |f(x*)| * H : 2x2 matrix   (was scalar)
%   P      : 2x2 ARE solution      (was scalar)
%   proxy  : -trace(P)             (was -P)
%
% Test function: 2D sum-of-Gaussians (5 wells at different locations/depths)
% Stress: global minima is NOT the stiffest or most central well

clear; clc; close all;

%% ── 1. Define 2D test function ───────────────────────────────────────────

% Well centers [x1, x2], depths, and widths (sigma controls Hessian scale)
mu    = [-3.0 -3.0;   % well 1
         -2.0  2.5;   % well 2
          0.5  0.5;   % well 3 — global (deepest)
          2.5 -1.5;   % well 4
          3.5  3.0];  % well 5

depth = [1.2, 0.9, 2.0, 1.5, 0.7];    % well 3 is global (depth=2.0)
sigma = [0.7, 0.5, 0.6, 0.8, 0.4];    % well 3 has medium width

N_wells = size(mu, 1);

% Scalar function f(x) where x = [x1, x2] column vector
f_scalar = @(x) f_eval(x, mu, depth, sigma);

% Vectorized for surface plotting — explicit loop, no arrayfun/cellfun
% (inline lambdas cause uniform output issues with nested vector inputs)

%% ── 2. Numerical gradient and Hessian ───────────────────────────────────

h = 1e-4;   % finite difference step

% Gradient: [df/dx1, df/dx2]
grad_f = @(x) [ (f_scalar(x + [h;0]) - f_scalar(x - [h;0])) / (2*h); ...
                (f_scalar(x + [0;h]) - f_scalar(x - [0;h])) / (2*h) ];

% Hessian: explicitly built 2x2 matrix
hessian_f = @(x) build_hessian(x, f_scalar, h);

%% ── 3. Find local minima via 2D grid search + fminsearch ─────────────────

% Grid search to find candidate starting points
[X1g, X2g] = meshgrid(linspace(-5,5,40), linspace(-5,5,40));
Fg = eval_grid(X1g, X2g, mu, depth, sigma);

% Find local minima: points lower than all 8 neighbors
candidates = [];
[nr, nc] = size(Fg);
for r = 2:nr-1
    for c = 2:nc-1
        neighborhood = Fg(r-1:r+1, c-1:c+1);
        if Fg(r,c) == min(neighborhood(:))
            candidates = [candidates; X1g(r,c), X2g(r,c)];
        end
    end
end
fprintf('Grid search found %d candidate minima regions\n', size(candidates,1));

% Refine each candidate using gradient-based minimization (no toolbox)
% Using simple gradient descent with line search
local_minima = [];
for i = 1:size(candidates,1)
    x = candidates(i,:)';
    for iter = 1:2000
        g = grad_f(x);
        if norm(g) < 1e-8; break; end
        % Backtracking line search
        alpha = 0.1;
        for ls = 1:20
            if double(f_scalar(x - alpha*g)) < double(f_scalar(x)); break; end
            alpha = alpha * 0.5;
        end
        x = x - alpha * g;
    end
    local_minima = [local_minima; x'];
end

% Remove duplicates
if ~isempty(local_minima)
    keep = true(size(local_minima,1),1);
    for i = 1:size(local_minima,1)
        for j = i+1:size(local_minima,1)
            if norm(local_minima(i,:) - local_minima(j,:)) < 0.2
                keep(j) = false;
            end
        end
    end
    local_minima = local_minima(keep,:);
end
N = size(local_minima, 1);
fprintf('Refined to %d distinct local minima\n\n', N);

%% ── 4. LQR at each minima ────────────────────────────────────────────────

results = struct();
valid_count = 0;

fprintf('%-20s %-12s %-14s %-14s %-14s\n', ...
    'x* = [x1, x2]', 'f(x*)', 'trace(H)', 'Q scale', 'Energy(-trP)');
fprintf('%s\n', repmat('-', 1, 76));

for i = 1:N
    x_star = local_minima(i,:)';
    f_val  = f_scalar(x_star);
    H      = hessian_f(x_star);

    % Check H is positive definite (valid minima, not saddle)
    eig_H = eig(H);
    if any(eig_H <= 0)
        fprintf('  Skipping [%.2f, %.2f]: not positive definite (saddle/maxima)\n', ...
            x_star(1), x_star(2));
        continue
    end

    A = -H;             % 2x2: gradient flow linearization
    B = eye(2);         % 2x2: full state control
    Q = abs(f_val) * H; % 2x2: depth-encoded cost matrix
    R = eye(2);         % 2x2: unit control cost

    % Solve continuous ARE: A'P + PA - PBR^{-1}B'P + Q = 0
    P = care(A, B, Q, R);

    % Scalar proxy: -trace(P) so deeper well = lower energy
    energy = -trace(P);

    valid_count = valid_count + 1;
    results(valid_count).x_star  = x_star;
    results(valid_count).f_val   = f_val;
    results(valid_count).H       = H;
    results(valid_count).P       = P;
    results(valid_count).energy  = energy;

    fprintf('[%5.2f, %5.2f]      %-12.6f %-14.6f %-14.6f %-14.6f\n', ...
        x_star(1), x_star(2), f_val, trace(H), abs(f_val)*trace(H), energy);
end
N = valid_count;

%% ── 5. Rank + Spearman correlation ──────────────────────────────────────

energies = [results.energy];
f_vals   = [results.f_val];

[~, energy_rank] = sort(energies);
[~, fval_rank]   = sort(f_vals);

fprintf('\n── Ranking by -trace(P) Energy ──\n');
for k = 1:N
    idx = energy_rank(k);
    fprintf('  Rank %d: x*=[%5.2f,%5.2f]  f(x*)=%8.5f  Energy=%8.5f\n', ...
        k, results(idx).x_star(1), results(idx).x_star(2), ...
        results(idx).f_val, results(idx).energy);
end

fprintf('\n── Ranking by True f(x*) ──\n');
for k = 1:N
    idx = fval_rank(k);
    fprintf('  Rank %d: x*=[%5.2f,%5.2f]  f(x*)=%8.5f  Energy=%8.5f\n', ...
        k, results(idx).x_star(1), results(idx).x_star(2), ...
        results(idx).f_val, results(idx).energy);
end

% Spearman rank correlation (manual)
[~, re] = sort(energies); rank_e = zeros(1,N); rank_e(re) = 1:N;
[~, rf] = sort(f_vals);   rank_f = zeros(1,N); rank_f(rf) = 1:N;
d2  = sum((rank_e - rank_f).^2);
rho = 1 - 6*d2 / (N*(N^2-1));
fprintf('\nSpearman Rank Correlation: %.4f\n', rho);

[~, e_idx] = min(energies);
[~, f_idx] = min(f_vals);
fprintf('\n-trace(P) identifies global minima at: [%.4f, %.4f]\n', ...
    results(e_idx).x_star(1), results(e_idx).x_star(2));
fprintf('True global minima at:                 [%.4f, %.4f]\n', ...
    results(f_idx).x_star(1), results(f_idx).x_star(2));
if e_idx == f_idx
    fprintf('✓ Correct global minima identified.\n');
else
    fprintf('✗ Wrong global minima identified.\n');
end

%% ── 6. Visualize ─────────────────────────────────────────────────────────

figure('Name','LQR v5 - 2D','Position',[80 80 1200 500]);

% -- Left: 2D contour + minima locations
subplot(1,3,1);
x1v = linspace(-5,5,120);
x2v = linspace(-5,5,120);
[X1,X2] = meshgrid(x1v, x2v);
F = eval_grid(X1, X2, mu, depth, sigma);
contourf(X1, X2, F, 25, 'LineColor','none'); hold on;
colorbar; colormap(gca, 'parula');
colors_list = {'ro','gs','md','c^','bh','wx'};
for i = 1:N
    plot(results(i).x_star(1), results(i).x_star(2), ...
        colors_list{mod(i-1,6)+1}, 'MarkerSize',12,'LineWidth',2);
    text(results(i).x_star(1)+0.15, results(i).x_star(2), ...
        sprintf('x*_%d',i), 'Color','w','FontSize',8,'FontWeight','bold');
end
% Mark true global
plot(results(f_idx).x_star(1), results(f_idx).x_star(2), ...
    'k*','MarkerSize',18,'LineWidth',2);
xlabel('x_1'); ylabel('x_2');
title('2D Landscape + Minima (★ = true global)');
grid on;

% -- Middle: Energy bar chart
subplot(1,3,2);
[~, ei] = min(energies);
b = bar(1:N, energies, 'FaceColor','flat');
b.CData = repmat([0.9 0.5 0.2], N, 1);
b.CData(ei,:) = [0.2 0.7 0.2];
xlabel('Minima'); ylabel('-trace(P)');
title('-trace(P) Energy — Green = predicted global');
xticklabels(arrayfun(@(i) sprintf('x*_%d',i), 1:N, 'UniformOutput',false));
grid on;

% -- Right: Scatter energy vs depth
subplot(1,3,3);
scatter(energies, f_vals, 140, 'filled', 'MarkerFaceColor',[0.8 0.2 0.2]);
hold on;
for i = 1:N
    text(energies(i), f_vals(i)-0.04, sprintf('x*_%d',i), ...
        'FontSize',8,'HorizontalAlignment','center');
end
p = polyfit(energies, f_vals, 1);
x_fit = linspace(min(energies), max(energies), 100);
plot(x_fit, polyval(p,x_fit), 'k--', 'LineWidth', 1.5);
xlabel('-trace(P)'); ylabel('True f(x*)');
title(sprintf('Spearman \\rho = %.3f', rho));
grid on;

sgtitle('LQR Energy Hypothesis — v5: 2D Extension,  proxy = -trace(P)', ...
    'FontSize', 13, 'FontWeight', 'bold');

%% ── Helper function ──────────────────────────────────────────────────────

function val = f_eval(x, mu, depth, sigma)
% Guaranteed scalar output — sum of Gaussians at point x=[x1;x2]
    d = depth(:)';    % 1xN row
    s = sigma(:)';    % 1xN row
    dists = (x(1) - mu(:,1)').^2 + (x(2) - mu(:,2)').^2;  % 1xN
    val = double(-sum( d .* exp(-dists ./ (2*s.^2)) ));     % true scalar
end

function F = eval_grid(X1, X2, mu, depth, sigma)
% Evaluate sum-of-Gaussians on a grid — explicit loop avoids arrayfun issues
    F = zeros(size(X1));
    for r = 1:size(X1,1)
        for c = 1:size(X1,2)
            x1 = X1(r,c); x2 = X2(r,c);
            F(r,c) = -sum(depth .* exp( ...
                -((x1 - mu(:,1)').^2 + (x2 - mu(:,2)').^2) ./ (2*sigma.^2)));
        end
    end
end

function H = build_hessian(x, f, h)
% Explicitly build 2x2 Hessian via finite differences
    f0  = double(f(x));
    % Second derivatives on diagonal
    H11 = double((f(x+[h;0]) - 2*f0 + f(x-[h;0])) / h^2);
    H22 = double((f(x+[0;h]) - 2*f0 + f(x-[0;h])) / h^2);
    % Cross derivative (symmetric)
    H12 = double((f(x+[h;h]) - f(x+[h;-h]) - f(x-[h;h]) + f(x-[h;-h])) / (4*h^2));
    % Assemble — force exact symmetry
    H = [H11(1), H12(1); H12(1), H22(1)];
end
