% run_comparison.m
%
% Task 2, Part 3: compare two optimisation techniques -- the custom GA and
% custom PSO from ../shared -- on two CEC'2005-style benchmark functions
% (Shifted Sphere = F1, unimodal; Shifted Rastrigin = F9, highly
% multimodal), at D=2 and D=10, 15 independent runs each, reporting
% mean/std/best/worst and convergence curves, per the brief's requirements.
%
% Results are reported as ERROR TO THE KNOWN OPTIMUM, f(x) - f(x*), which
% is the convention used throughout the CEC'2005 special session and in the
% literature this comparison is meant to be readable against. Because both
% functions here are shifted but not rotated or composed, f(x*) is exactly
% the bias constant (-450 for F1, -330 for F9), so the error is known in
% closed form and no separate "true optimum" estimate is needed. Reporting
% the error rather than the raw fitness also makes a log-scale convergence
% plot meaningful (the raw values are negative and cannot be log-plotted).

clear; clc;
this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir, '..', 'shared'));

functions = {'sphere', 'rastrigin'};
domains = struct('sphere', [-100 100], 'rastrigin', [-5 5]);
biases  = struct('sphere', -450,       'rastrigin', -330);   % f(x*) for each function
dimensions = [2, 10];
n_runs = 15;

% Iteration budgets kept equal between GA and PSO for a fair comparison
% (both see essentially the same number of fitness evaluations per run):
%   GA:  pop_size 40  x (60 generations + 1 initial) = 2440 evaluations
%   PSO: n_particles 30 x (80 iterations + 1 initial) = 2430 evaluations
ga_opts_base = struct('pop_size', 40, 'n_generations', 60);
pso_opts_base = struct('n_particles', 30, 'n_iterations', 80);

results = table();
per_run = table();
convergence_curves = struct();

for fi = 1:length(functions)
    fname = functions{fi};
    bounds = domains.(fname);
    f_opt  = biases.(fname);

    for D = dimensions
        lb = repmat(bounds(1), 1, D);
        ub = repmat(bounds(2), 1, D);
        fitness_fn = @(x) cec_functions(fname, x);

        % --- GA runs ---
        ga_best = zeros(n_runs, 1);
        ga_hist = zeros(ga_opts_base.n_generations+1, n_runs);
        for r = 1:n_runs
            opts = ga_opts_base; opts.seed = 1000 + r;
            [~, best_f, hist] = custom_ga(fitness_fn, lb, ub, opts);
            ga_best(r) = best_f - f_opt;              % error to optimum
            ga_hist(:, r) = hist - f_opt;
        end

        % --- PSO runs ---
        pso_best = zeros(n_runs, 1);
        pso_hist = zeros(pso_opts_base.n_iterations+1, n_runs);
        for r = 1:n_runs
            opts = pso_opts_base; opts.seed = 2000 + r;
            [~, best_f, hist] = custom_pso(fitness_fn, lb, ub, opts);
            pso_best(r) = best_f - f_opt;             % error to optimum
            pso_hist(:, r) = hist - f_opt;
        end

        % --- Statistical comparison of the two optimisers ---
        % 15 runs per optimiser is a small sample and the error
        % distributions are strongly right-skewed (a few unlucky runs stuck
        % in a local optimum dominate the mean), so a non-parametric
        % Wilcoxon rank-sum test on the two sets of 15 final errors is the
        % appropriate test rather than a t-test on the means.
        if exist('ranksum', 'file') == 2
            p_value = ranksum(ga_best, pso_best);
        else
            p_value = NaN;   % Statistics & Machine Learning Toolbox not installed
        end
        if median(ga_best) < median(pso_best)
            winner = 'GA';
        elseif median(pso_best) < median(ga_best)
            winner = 'PSO';
        else
            winner = 'tie';
        end

        % --- Record summary statistics ---
        results = [results; table({fname}, D, {'GA'}, mean(ga_best), std(ga_best), ...
            min(ga_best), max(ga_best), median(ga_best), p_value, {winner}, ...
            'VariableNames', {'Function','Dimension','Optimizer','MeanError','StdError', ...
                              'BestError','WorstError','MedianError','RanksumP','BetterMedian'})]; %#ok<AGROW>
        results = [results; table({fname}, D, {'PSO'}, mean(pso_best), std(pso_best), ...
            min(pso_best), max(pso_best), median(pso_best), p_value, {winner}, ...
            'VariableNames', {'Function','Dimension','Optimizer','MeanError','StdError', ...
                              'BestError','WorstError','MedianError','RanksumP','BetterMedian'})]; %#ok<AGROW>

        % --- Per-run errors, so the appendix can show all 15 runs, not
        %     just their summary (the brief asks for the runs themselves) ---
        per_run = [per_run; table(repmat({fname}, n_runs, 1), repmat(D, n_runs, 1), ...
            (1:n_runs)', ga_best, pso_best, ...
            'VariableNames', {'Function','Dimension','Run','GA_Error','PSO_Error'})]; %#ok<AGROW>

        key = sprintf('%s_D%d', fname, D);
        convergence_curves.(key).ga = ga_hist;
        convergence_curves.(key).pso = pso_hist;

        % --- Convergence plot (median curve, both optimisers, log scale) ---
        % The MEDIAN across the 15 runs is plotted rather than the mean:
        % with a log axis a single catastrophic run would otherwise drag the
        % mean curve up by orders of magnitude and misrepresent typical
        % behaviour. A floor of 1e-12 keeps exact-zero errors plottable.
        figure('Visible', 'off');
        ga_curve  = max(median(ga_hist, 2), 1e-12);
        pso_curve = max(median(pso_hist, 2), 1e-12);
        semilogy(0:ga_opts_base.n_generations, ga_curve, 'LineWidth', 1.5, 'DisplayName', 'GA');
        hold on;
        semilogy(0:pso_opts_base.n_iterations, pso_curve, 'LineWidth', 1.5, 'DisplayName', 'PSO');
        xlabel('Generation / Iteration');
        ylabel('Error to optimum, f(x) - f(x^*)  (median of 15 runs)');
        title(sprintf('%s, D=%d: GA vs PSO convergence', fname, D));
        legend('show'); grid on;
        saveas(gcf, fullfile(this_dir, sprintf('convergence_%s.png', key)));
        close(gcf);

        % --- Box plot of the 15 final errors per optimiser: shows spread
        %     and outliers, which the mean/std alone hide ---
        % boxplot() lives in the Statistics & Machine Learning Toolbox, which
        % is not guaranteed to be installed. Guard it so a missing toolbox
        % costs one optional figure rather than aborting the whole run
        % mid-loop and losing every result before the CSVs are written.
        if exist('boxplot', 'file') == 2
            figure('Visible', 'off');
            boxplot(max([ga_best, pso_best], 1e-12), 'Labels', {'GA', 'PSO'});
            set(gca, 'YScale', 'log');
            ylabel('Final error to optimum (15 runs)');
            title(sprintf('%s, D=%d: distribution of final error', fname, D));
            grid on;
            saveas(gcf, fullfile(this_dir, sprintf('boxplot_%s.png', key)));
            close(gcf);
        end

        fprintf('Done: %s, D=%d  (GA median %.4g vs PSO median %.4g, ranksum p=%.4g)\n', ...
            fname, D, median(ga_best), median(pso_best), p_value);
    end
end

writetable(results, fullfile(this_dir, 'optimizer_comparison_results.csv'));
writetable(per_run, fullfile(this_dir, 'optimizer_per_run_errors.csv'));
save(fullfile(this_dir, 'convergence_curves.mat'), 'convergence_curves');

fprintf('\n=== Summary: error to optimum, mean +/- std, [best, worst] ===\n');
for i = 1:height(results)
    fprintf('%-10s D=%-3d %-4s : %11.4g +/- %10.4g  [%10.4g, %10.4g]\n', ...
        results.Function{i}, results.Dimension(i), results.Optimizer{i}, ...
        results.MeanError(i), results.StdError(i), results.BestError(i), results.WorstError(i));
end
fprintf('\nSaved optimizer_comparison_results.csv, optimizer_per_run_errors.csv,\n');
fprintf('convergence_curves.mat, and one convergence_*.png + boxplot_*.png per\n');
fprintf('function/dimension combination.\n');
