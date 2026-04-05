%% LQR Energy-Based Global Optimization - v8 (Unified Formula)
% Unifies v4 (Gaussian wells) and v7 (Rastrigin) into one sign-agnostic encoding.
%
% Fix 4: normalized range depth
%   depth(x*) = (f_max - f(x*)) / (f_max - f_min + ε)   ∈ [0,1]
%   Q = depth(x*) · H
%   proxy = -P
%
% Global minima always gets depth≈1 (largest Q), shallowest gets depth≈0.
% Works for any sign of f(x*) — no f_ref needed.
%
% Tests:
%   A: 5-well Gaussian  (f < 0, from v4)
%   B: Rastrigin [-5,5] (f ≥ 0, from v6)

clear; clc; close all;

epsilon = 1e-6;   % range normalization guard

%% ════════════════════════════════════════════════════════════════════════
%% TEST A: 5-well Gaussian
%% ════════════════════════════════════════════════════════════════════════

fprintf('════════════════════════════════════════\n');
fprintf('  TEST A: 5-Well Gaussian  (f < 0)\n');
fprintf('════════════════════════════════════════\n\n');

mu_A    = [-4.0, -2.0, 0.5, 2.5, 4.5];
depth_A = [ 1.2,  0.9, 2.0, 1.5, 0.7];
sigma_A = [ 0.7,  0.5, 0.6, 0.8, 0.4];

f_A    = @(x) f_eval(x, mu_A, depth_A, sigma_A);
dfdx_A = @(x) (f_A(x+1e-5) - f_A(x-1e-5)) / 2e-5;
H_A    = @(x) (f_A(x+1e-4) - 2*f_A(x) + f_A(x-1e-4)) / 1e-8;

[rho_A, correct_A, results_A] = run_lqr_optim( ...
    f_A, dfdx_A, H_A, -5.5, 5.5, 300, epsilon);

%% ════════════════════════════════════════════════════════════════════════
%% TEST B: Rastrigin
%% ════════════════════════════════════════════════════════════════════════

fprintf('\n════════════════════════════════════════\n');
fprintf('  TEST B: Rastrigin  (f >= 0)\n');
fprintf('════════════════════════════════════════\n\n');

Ar = 10;
f_B    = @(x) Ar + x^2 - Ar*cos(2*pi*x);
dfdx_B = @(x) 2*x + 2*pi*Ar*sin(2*pi*x);
H_B    = @(x) 2 + 4*pi^2*Ar*cos(2*pi*x);

[rho_B, correct_B, results_B] = run_lqr_optim( ...
    f_B, dfdx_B, H_B, -5, 5, 500, epsilon);

%% ════════════════════════════════════════════════════════════════════════
%% SUMMARY
%% ════════════════════════════════════════════════════════════════════════

fprintf('\n════════════════════════════════════════\n');
fprintf('  UNIFIED FORMULA SUMMARY\n');
fprintf('  Q = depth(x*) * H\n');
fprintf('  depth = (f_max-f) / (f_max-f_min+e)\n');
fprintf('════════════════════════════════════════\n');
fprintf('  %-22s  rho=%6.4f  Global: %s\n', 'Test A (Gaussian)', rho_A, yn(correct_A));
fprintf('  %-22s  rho=%6.4f  Global: %s\n', 'Test B (Rastrigin)', rho_B, yn(correct_B));
if correct_A && correct_B && rho_A > 0.9 && rho_B > 0.9
    fprintf('\n  UNIFIED FORMULA WORKS ON BOTH FUNCTION CLASSES\n');
else
    fprintf('\n  UNIFIED FORMULA FAILS ON AT LEAST ONE CLASS\n');
end
fprintf('════════════════════════════════════════\n');

%% ════════════════════════════════════════════════════════════════════════
%% VISUALIZATION
%% ════════════════════════════════════════════════════════════════════════

figure('Name','LQR v8 - Unified Formula','Position',[60 60 1300 600]);

plot_test(results_A, rho_A, 'Test A: 5-Well Gaussian', ...
    linspace(-5.5,5.5,500), f_A, [1,2,3]);
plot_test(results_B, rho_B, 'Test B: Rastrigin', ...
    linspace(-5,5,2000), f_B, [4,5,6]);

sgtitle('LQR v8: Unified Formula  depth=(f_{max}-f)/(f_{max}-f_{min}+e)', ...
    'FontSize', 12, 'FontWeight', 'bold');

%% ════════════════════════════════════════════════════════════════════════
%% LOCAL FUNCTIONS
%% ════════════════════════════════════════════════════════════════════════

function [rho, correct, results] = run_lqr_optim(f, dfdx, Hfun, ...
    x_lo, x_hi, n_grid, epsilon)

    % Grid search
    grid_pts  = linspace(x_lo, x_hi, n_grid);
    grad_vals = arrayfun(dfdx, grid_pts);
    sc = find(diff(sign(grad_vals)) > 0);
    fprintf('Grid search: %d candidate regions\n', numel(sc));

    % fzero refinement
    mins = [];
    for i = 1:numel(sc)
        try
            mins(end+1) = fzero(dfdx, [grid_pts(sc(i)), grid_pts(sc(i)+1)]);
        catch; end
    end
    mins = uniquetol(mins, 1e-4);
    N    = numel(mins);
    fprintf('Refined: %d distinct local minima\n\n', N);

    % Evaluate f at all minima first — needed for range normalization
    f_vals_all = arrayfun(f, mins);
    f_max  = max(f_vals_all);
    f_min  = min(f_vals_all);
    range  = f_max - f_min + epsilon;
    fprintf('  f_min=%.4f  f_max=%.4f  range=%.4f\n\n', f_min, f_max, range);

    fprintf('%-10s %-12s %-8s %-10s %-12s\n', ...
        'x*','f(x*)','depth','Q=d*H','Energy(-P)');
    fprintf('%s\n', repmat('-',1,56));

    results = struct();
    vc = 0;

    for i = 1:N
        x_star = mins(i);
        f_val  = f_vals_all(i);
        H      = Hfun(x_star);

        if H <= 0; continue; end

        % Unified depth encoding — sign agnostic
        depth  = (f_max - f_val) / range;    % in [0,1], global minima gets ~1
        A_sys  = -H;
        B      = 1;
        Q      = max(depth * H, 1e-10);      % guard against Q=0
        R      = 1;

        try
            P      = care(A_sys, B, Q, R);
            energy = -P;
        catch
            continue
        end

        vc = vc + 1;
        results(vc).x_star  = x_star;
        results(vc).f_val   = f_val;
        results(vc).hessian = H;
        results(vc).depth   = depth;
        results(vc).P       = P;
        results(vc).energy  = energy;

        fprintf('%-10.4f %-12.6f %-8.4f %-10.4f %-12.6f\n', ...
            x_star, f_val, depth, depth*H, energy);
    end
    N = vc;

    energies = [results.energy];
    f_vals   = [results.f_val];
    x_stars  = [results.x_star];

    % Spearman rank correlation (manual)
    [~,re] = sort(energies); re_r = zeros(1,N); re_r(re) = 1:N;
    [~,rf] = sort(f_vals);   rf_r = zeros(1,N); rf_r(rf) = 1:N;
    d2  = sum((re_r - rf_r).^2);
    rho = 1 - 6*d2/(N*(N^2-1));

    [~,e_idx] = min(energies);
    [~,f_idx] = min(f_vals);
    correct = abs(x_stars(e_idx) - x_stars(f_idx)) < 0.15;

    fprintf('\nSpearman rho = %.4f\n', rho);
    fprintf('Predicted global: x*=%.4f  f=%.5f\n', x_stars(e_idx), f_vals(e_idx));
    fprintf('True global:      x*=%.4f  f=%.5f\n', x_stars(f_idx), f_vals(f_idx));
    if correct
        fprintf('Correct global minima identified.\n');
    else
        fprintf('Wrong global minima identified.\n');
    end
end

function plot_test(results, rho, ttl, x_plot, f, sid)
    if isempty(results); return; end
    N = numel(results);
    energies = [results.energy];
    f_vals   = [results.f_val];
    [~,e_idx] = min(energies);
    [~,f_idx] = min(f_vals);
    [~,fr]    = sort(f_vals);

    subplot(2,3,sid(1));
    plot(x_plot, arrayfun(f,x_plot), 'b-', 'LineWidth',2); hold on;
    for i = 1:N
        if i==f_idx
            plot(results(i).x_star,results(i).f_val,'k*','MarkerSize',16,'LineWidth',2);
        elseif i==e_idx
            plot(results(i).x_star,results(i).f_val,'g^','MarkerSize',12,'LineWidth',2);
        else
            plot(results(i).x_star,results(i).f_val,'ro','MarkerSize',7);
        end
    end
    xlabel('x'); ylabel('f(x)'); title(ttl,'FontSize',10);
    legend('f(x)','True *','Predicted ^','Local','Location','best');
    grid on;

    subplot(2,3,sid(2));
    depths = [results.depth];
    bar(1:N, depths(fr), 'FaceColor',[0.3 0.6 0.9]);
    xlabel('Rank (true depth)'); ylabel('depth');
    title('Depth weights (should decrease)'); grid on;

    subplot(2,3,sid(3));
    scatter(energies, f_vals, 100, 'filled','MarkerFaceColor',[0.8 0.2 0.2]);
    hold on;
    p = polyfit(energies,f_vals,1);
    xf = linspace(min(energies),max(energies),100);
    plot(xf,polyval(p,xf),'k--','LineWidth',1.5);
    scatter(energies(e_idx),f_vals(e_idx),180,'g^','filled');
    scatter(energies(f_idx),f_vals(f_idx),180,'k*','LineWidth',2);
    xlabel('-P'); ylabel('f(x*)');
    title(sprintf('rho = %.4f', rho)); grid on;
end

function s = yn(b)
    if b; s = 'Correct'; else; s = 'Wrong'; end
end

function val = f_eval(x, mu, depth, sigma)
    d = depth(:)'; s = sigma(:)';
    val = double(-sum(d .* exp(-(x-mu).^2 ./ (2*s.^2))));
end