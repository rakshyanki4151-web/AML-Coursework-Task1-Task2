% ga_tune_flc.m
%
% Task 2, Part 2: optimise the FLC's membership functions with a Genetic
% Algorithm, keeping the FLC's structure (same fuzzy sets, same 15 rules)
% identical to Part 1, per the brief's instruction to keep the same
% structure and only adjust membership functions.
%
% Run build_flc.m and generate_reference_dataset.m first.

clear; clc;
this_dir = fileparts(mfilename('fullpath'));
addpath(fullfile(this_dir, '..', 'shared'));
addpath(fullfile(this_dir, '..', 'part1_flc'));

base_fis = readfis(fullfile(this_dir, '..', 'part1_flc', 'thermostat_flc.fis'));
data = readtable(fullfile(this_dir, 'reference_dataset.csv'));
X = [data.TempError, data.RateChange];
y = data.TargetPower;

%% Chromosome bounds (see apply_chromosome.m for the full explanation)
%          TempErr_shift TempErr_scale RateChg_shift RateChg_scale Out_shift Out_scale
lb = [       -2.0,          0.6,          -0.4,          0.6,       -15,       0.6];
ub = [        2.0,          1.6,           0.4,          1.6,        15,       1.6];

%% Fitness function: MSE between tuned-FLC output and the reference dataset
fitness_fn = @(chromosome) flc_mse(chromosome, base_fis, X, y);

%% Baseline: Part 1's un-tuned FLC MSE, for a before/after comparison
baseline_mse = flc_mse([0 1 0 1 0 1], base_fis, X, y); % identity chromosome
fprintf('Part 1 (untuned) FLC MSE on reference dataset: %.3f\n', baseline_mse);

%% Run the custom GA (Section: Genetic Algorithm Design in the report)
ga_opts = struct('pop_size', 40, 'n_generations', 60, 'seed', 42);
[best_chromosome, best_mse, history] = custom_ga(fitness_fn, lb, ub, ga_opts);

fprintf('GA-tuned FLC MSE on reference dataset: %.3f (%.1f%% reduction)\n', ...
    best_mse, 100*(baseline_mse-best_mse)/baseline_mse);
fprintf('Best chromosome [TE_shift TE_scale RC_shift RC_scale Out_shift Out_scale]:\n');
disp(best_chromosome);

%% Save the tuned FIS
tuned_fis = apply_chromosome(base_fis, best_chromosome);
if exist('writeFIS', 'file')
    writeFIS(tuned_fis, fullfile(this_dir, 'thermostat_flc_tuned'));
else
    writefis(tuned_fis, fullfile(this_dir, 'thermostat_flc_tuned'));
end
fprintf('Saved thermostat_flc_tuned.fis\n');

%% Plot 1: GA convergence curve
figure('Name', 'GA Convergence');
plot(0:length(history)-1, history, 'LineWidth', 1.5);
xlabel('Generation'); ylabel('Best MSE so far');
title('GA convergence: FLC membership-function tuning');
grid on;
saveas(gcf, fullfile(this_dir, 'ga_convergence.png'));

%% Plot 2: Before/after membership functions, for the report's justification
figure('Name', 'Before vs After Tuning', 'Position', [100 100 900 700]);
subplot(3,2,1); plotmf(base_fis, 'input', 1); title('TempError -- before (Part 1)');
subplot(3,2,2); plotmf(tuned_fis, 'input', 1); title('TempError -- after GA tuning');
subplot(3,2,3); plotmf(base_fis, 'input', 2); title('RateChange -- before (Part 1)');
subplot(3,2,4); plotmf(tuned_fis, 'input', 2); title('RateChange -- after GA tuning');
subplot(3,2,5); plotmf(base_fis, 'output', 1); title('Output -- before (Part 1)');
subplot(3,2,6); plotmf(tuned_fis, 'output', 1); title('Output -- after GA tuning');
saveas(gcf, fullfile(this_dir, 'before_after_mfs.png'));

fprintf('Saved ga_convergence.png and before_after_mfs.png\n');

%% Plot 3: control surfaces, before vs after tuning
figure('Name', 'Control Surface: Before vs After', 'Position', [100 100 1000 420]);
subplot(1,2,1); gensurf(base_fis);  title('Control surface -- before (Part 1)');
subplot(1,2,2); gensurf(tuned_fis); title('Control surface -- after GA tuning');
saveas(gcf, fullfile(this_dir, 'before_after_surface.png'));
fprintf('Saved before_after_surface.png\n');

%% Closed-loop validation of the tuned controller
%
% The GA's fitness is dataset MSE against the reference law, which is what
% the brief specifies. That is an OPEN-LOOP criterion, though: it measures
% how closely the FLC reproduces a set of (input, output) examples, not how
% the controller behaves once it is actually driving a room. This section
% therefore validates the tuned FLC on the same closed-loop plant used in
% Part 1 (../shared/simulate_room.m) -- an evaluation the GA never saw, so
% an improvement here is genuine evidence the tuning helped the controller
% and not merely evidence that the GA fitted its own objective.

scenario_opts = struct('setpoint', 22, 'T0', 16, 'T_out', 5, ...
                        'tau', 60, 'k_act', 0.5, 'dt', 1, 't_end', 180, 'band', 0.5);

[t_b, T_b, u_b, m_b] = simulate_room(base_fis,  scenario_opts);
[t_a, T_a, u_a, m_a] = simulate_room(tuned_fis, scenario_opts);

fprintf('\n--- Closed-loop validation (held out from the GA fitness) ---\n');
fprintf('%-26s %14s %14s\n', 'Metric', 'Untuned (P1)', 'GA-tuned');
fprintf('%-26s %14.2f %14.2f\n', 'Settling time (min)',       m_b.settling_time_min,        m_a.settling_time_min);
fprintf('%-26s %14.3f %14.3f\n', 'Overshoot (degC)',          m_b.overshoot_degC,           m_a.overshoot_degC);
fprintf('%-26s %14.3f %14.3f\n', 'Steady-state error (degC)', m_b.steady_state_error_degC,  m_a.steady_state_error_degC);
fprintf('%-26s %14.1f %14.1f\n', 'IAE (degC.min)',            m_b.IAE,                      m_a.IAE);
fprintf('%-26s %14.1f %14.1f\n', 'Energy (pct.min)',          m_b.energy_pct_min,           m_a.energy_pct_min);

cl = table( ...
    {'Settling time (min)'; 'Overshoot (degC)'; 'Steady-state error (degC)'; 'IAE (degC.min)'; 'Energy (pct.min)'}, ...
    [m_b.settling_time_min; m_b.overshoot_degC; m_b.steady_state_error_degC; m_b.IAE; m_b.energy_pct_min], ...
    [m_a.settling_time_min; m_a.overshoot_degC; m_a.steady_state_error_degC; m_a.IAE; m_a.energy_pct_min], ...
    'VariableNames', {'Metric', 'Untuned', 'GATuned'});
writetable(cl, fullfile(this_dir, 'closed_loop_before_after.csv'));

figure('Name', 'Closed-loop: Untuned vs GA-tuned', 'Position', [100 100 900 600]);
subplot(2,1,1);
plot(t_b, T_b, '--', 'LineWidth', 1.4); hold on;
plot(t_a, T_a, 'LineWidth', 1.6);
yline(scenario_opts.setpoint, ':k', 'Setpoint');
xlabel('Time (min)'); ylabel('Room temperature (^\circC)');
title('Closed-loop response before and after GA tuning');
legend('Untuned (Part 1)', 'GA-tuned', 'Location', 'southeast'); grid on;
subplot(2,1,2);
plot(t_b, u_b, '--', 'LineWidth', 1.4); hold on;
plot(t_a, u_a, 'LineWidth', 1.6);
yline(0, ':k');
xlabel('Time (min)'); ylabel('HeaterCoolerPower (%)');
title('Actuator command'); legend('Untuned (Part 1)', 'GA-tuned', 'Location', 'northeast'); grid on;
saveas(gcf, fullfile(this_dir, 'closed_loop_before_after.png'));

fprintf('\nSaved closed_loop_before_after.png and closed_loop_before_after.csv\n');

%% Save every number the report needs, in one place
summary = table( ...
    {'Untuned FLC (Part 1)'; 'GA-tuned FLC'}, ...
    [baseline_mse; best_mse], ...
    'VariableNames', {'Controller', 'ReferenceDatasetMSE'});
writetable(summary, fullfile(this_dir, 'ga_tuning_summary.csv'));
writematrix(best_chromosome, fullfile(this_dir, 'best_chromosome.csv'));
fprintf('Saved ga_tuning_summary.csv and best_chromosome.csv\n');

function mse = flc_mse(chromosome, base_fis, X, y)
    fis = apply_chromosome(base_fis, chromosome);
    y_pred = evalfis(fis, X);
    mse = mean((y_pred - y).^2);
end
