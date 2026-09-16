% analyze_flc.m
%
% Task 2, Part 1: analysis of the FLC's output behaviour.
% Run build_flc.m first to generate thermostat_flc.fis.
%
% Produces, all saved automatically as PNG (no GUI screenshot needed):
%   1. Membership function plots for both inputs and the output
%   2. The control surface (Output vs. both inputs) -- the required
%      "control surface plot" evidence
%   3. A worked example table of rule activation at several representative
%      operating points, printed to the console (and saved as CSV) for
%      the "rule activation" evidence
%
% Combine these with the two manual GUI screenshots (fuzzyLogicDesigner,
% ruleview) listed at the end of build_flc.m for the full Part 1 evidence
% pack.

clear; clc;
this_dir = fileparts(mfilename('fullpath'));
% NOTE: the Fuzzy Logic Toolbox naming is asymmetric -- writefis was
% renamed to writeFIS, but the reader is still spelled readfis (lowercase).
% There is no readFIS, in any release.
fis = readfis(fullfile(this_dir, 'thermostat_flc.fis'));

%% 1. Membership function plots
figure('Name', 'Membership Functions', 'Position', [100 100 900 700]);

subplot(3,1,1);
plotmf(fis, 'input', 1);
title('Input 1: TempError (room temperature minus setpoint, ^\circC)');

subplot(3,1,2);
plotmf(fis, 'input', 2);
title('Input 2: RateChange (deg C per minute)');

subplot(3,1,3);
plotmf(fis, 'output', 1);
title('Output: HeaterCoolerPower (%, negative = cooling, positive = heating)');

saveas(gcf, fullfile(this_dir, 'membership_functions.png'));
fprintf('Saved membership_functions.png\n');

%% 2. Control surface plot
figure('Name', 'Control Surface', 'Position', [100 100 800 600]);
gensurf(fis);
title('Control surface: HeaterCoolerPower vs. TempError and RateChange');
saveas(gcf, fullfile(this_dir, 'control_surface.png'));
fprintf('Saved control_surface.png\n');

%% 3. Worked operational scenarios (rule activation evidence)
% Each row: [TempError, RateChange, description]
scenarios = {
    -8.0, -1.5, 'Cold room, still cooling (worst-case heating demand)';
    -8.0,  1.5, 'Cold room, but already warming back up';
    -0.5,  0.0, 'At setpoint, stable';
     0.0,  1.0, 'At setpoint, but warming (should start easing toward cooling)';
     8.0,  1.5, 'Hot room, still heating up (worst-case cooling demand)';
     8.0, -1.5, 'Hot room, but already cooling back down';
};

fprintf('\n%-12s %-12s %-45s %-10s\n', 'TempError', 'RateChange', 'Scenario', 'Output(%)');
results = cell(size(scenarios,1), 4);
for i = 1:size(scenarios,1)
    e  = scenarios{i,1};
    de = scenarios{i,2};
    desc = scenarios{i,3};
    out = evalfis(fis, [e de]);
    fprintf('%-12.2f %-12.2f %-45s %-10.2f\n', e, de, desc, out);
    results(i,:) = {e, de, desc, out};
end

% Save as CSV for inclusion in the report appendix
T = cell2table(results, 'VariableNames', {'TempError','RateChange','Scenario','HeaterCoolerPower'});
writetable(T, fullfile(this_dir, 'scenario_results.csv'));
fprintf('\nSaved scenario_results.csv\n');

fprintf('\nFor the rule-activation screenshot, open ruleview(fis) and enter the\n');
fprintf('first and fifth scenario''s [TempError RateChange] values above --\n');
fprintf('these two show clearly different subsets of the 15 rules firing.\n');

%% 4. Rule activation strengths (numeric evidence, not just a screenshot)
%
% The brief's Part 1 asks for "analysis of the output behaviour of the
% controller showing the rules of activation". A GUI screenshot of the rule
% viewer shows this qualitatively; this section extracts the actual firing
% strength of each of the 15 rules at two contrasting operating points, so
% the report can state exactly which rules drive each control decision and
% with what weight -- quantitative evidence a screenshot cannot give.

probe_points = [-8.0 -1.5;    % cold room, still cooling  -> max heating
                 7.0  1.0];   % hot room, still warming   -> max cooling
probe_labels = {'Cold & Falling [-8, -1.5]', 'Hot & Rising [7, 1]'};

rule_names = arrayfun(@(r) sprintf('R%d', r), 1:numel(fis.Rules), 'UniformOutput', false)';
firing_table = table(rule_names, 'VariableNames', {'Rule'});

fprintf('\n--- Rule activation strengths ---\n');
for p = 1:size(probe_points, 1)
    % evalfis can return per-rule firing strengths as its 5th output, but
    % that signature is not available in every release. Try it, and fall
    % back to computing the firing strengths directly from the membership
    % functions -- which uses only property access already exercised
    % elsewhere in this project, so it cannot depend on a toolbox signature.
    out = evalfis(fis, probe_points(p,:));
    try
        [~, ~, ~, ~, ruleFiring] = evalfis(fis, probe_points(p,:));
        firing = ruleFiring(:);
    catch
        firing = local_firing_strengths(fis, probe_points(p,:));
    end
    if numel(firing) ~= numel(fis.Rules)
        firing = local_firing_strengths(fis, probe_points(p,:));
    end
    firing_table.(sprintf('Firing_P%d', p)) = firing;

    fprintf('\n%s  ->  HeaterCoolerPower = %.2f %%\n', probe_labels{p}, out);
    active = find(firing > 1e-6);
    for a = active'
        fprintf('   Rule %2d fires at %.3f\n', a, firing(a));
    end
    if isempty(active)
        fprintf('   (no rules fired -- this would indicate a coverage gap)\n');
    end
end

writetable(firing_table, fullfile(this_dir, 'rule_activation.csv'));
fprintf('\nSaved rule_activation.csv\n');

% Bar chart of the firing strengths at both probe points, side by side --
% the figure to put next to the rule-viewer screenshot in the report.
figure('Name', 'Rule Activation', 'Position', [100 100 900 400]);
bar([firing_table.Firing_P1, firing_table.Firing_P2]);
set(gca, 'XTick', 1:numel(fis.Rules), 'XTickLabel', rule_names);
xlabel('Rule'); ylabel('Firing strength');
legend(probe_labels, 'Location', 'best');
title('Rule activation at two contrasting operating points');
grid on;
saveas(gcf, fullfile(this_dir, 'rule_activation.png'));
fprintf('Saved rule_activation.png\n');

%% 5. Closed-loop operational scenario
% Wrapped in try/catch: the membership-function plots, control surface and
% scenario table above are the Part 1 essentials and are already saved by
% this point. If anything here fails on a given MATLAB release, the script
% should say so and still exit cleanly rather than discarding everything
% that already succeeded.
try
%
% The brief asks the analysis to demonstrate "how the controller achieves
% the specified behaviours in relation to an operational scenario". The
% static probe points above show what the controller commands at an
% instant; this section closes the loop around a first-order room thermal
% model (see ../shared/simulate_room.m) and shows what the controller
% actually DOES over time -- the behaviour that matters to a resident.
%
% A classical linear PD controller (the same reference law Part 2 tunes
% against) is simulated on the identical plant for comparison, so the
% report can argue concretely for the FLC rather than asserting it.

addpath(fullfile(this_dir, '..', 'shared'));

Kp = 8; Kd = 15;   % same gains as Part 2's reference law
pd_controller = @(e, de) max(min(-Kp*e - Kd*de, 100), -100);

scenario_opts = struct( ...
    'setpoint', 22, 'T0', 16, 'T_out', 5, ...   % cold Karnali winter morning
    'tau', 60, 'k_act', 0.5, 'dt', 1, 't_end', 180, 'band', 0.5);

[t_f, T_f, u_f, m_f] = simulate_room(fis,           scenario_opts);
[t_p, T_p, u_p, m_p] = simulate_room(pd_controller, scenario_opts);

fprintf('\n--- Closed-loop scenario: cold morning, 16 -> 22 degC, outdoor 5 degC ---\n');
fprintf('%-22s %12s %12s\n', 'Metric', 'Fuzzy FLC', 'Linear PD');
fprintf('%-22s %12.2f %12.2f\n', 'Settling time (min)',  m_f.settling_time_min,       m_p.settling_time_min);
fprintf('%-22s %12.3f %12.3f\n', 'Overshoot (degC)',     m_f.overshoot_degC,          m_p.overshoot_degC);
fprintf('%-22s %12.3f %12.3f\n', 'Steady-state err (degC)', m_f.steady_state_error_degC, m_p.steady_state_error_degC);
fprintf('%-22s %12.1f %12.1f\n', 'IAE (degC.min)',       m_f.IAE,                     m_p.IAE);
fprintf('%-22s %12.1f %12.1f\n', 'Energy (pct.min)',     m_f.energy_pct_min,          m_p.energy_pct_min);

closed_loop = table( ...
    {'Settling time (min)'; 'Overshoot (degC)'; 'Steady-state error (degC)'; 'IAE (degC.min)'; 'Energy (pct.min)'}, ...
    [m_f.settling_time_min; m_f.overshoot_degC; m_f.steady_state_error_degC; m_f.IAE; m_f.energy_pct_min], ...
    [m_p.settling_time_min; m_p.overshoot_degC; m_p.steady_state_error_degC; m_p.IAE; m_p.energy_pct_min], ...
    'VariableNames', {'Metric', 'FuzzyFLC', 'LinearPD'});
writetable(closed_loop, fullfile(this_dir, 'closed_loop_metrics.csv'));

figure('Name', 'Closed-loop Step Response', 'Position', [100 100 900 600]);
subplot(2,1,1);
plot(t_f, T_f, 'LineWidth', 1.6); hold on;
plot(t_p, T_p, '--', 'LineWidth', 1.4);
yline(scenario_opts.setpoint, ':k', 'Setpoint');
yline(scenario_opts.setpoint + scenario_opts.band, ':', 'Color', [.6 .6 .6]);
yline(scenario_opts.setpoint - scenario_opts.band, ':', 'Color', [.6 .6 .6]);
xlabel('Time (min)'); ylabel('Room temperature (^\circC)');
title('Closed-loop response: cold morning (16 \rightarrow 22 ^\circC, outdoor 5 ^\circC)');
legend('Fuzzy FLC', 'Linear PD', 'Location', 'southeast'); grid on;

subplot(2,1,2);
plot(t_f, u_f, 'LineWidth', 1.6); hold on;
plot(t_p, u_p, '--', 'LineWidth', 1.4);
yline(0, ':k');
xlabel('Time (min)'); ylabel('HeaterCoolerPower (%)');
title('Actuator command (positive = heating, negative = cooling)');
legend('Fuzzy FLC', 'Linear PD', 'Location', 'northeast'); grid on;

saveas(gcf, fullfile(this_dir, 'closed_loop_response.png'));
fprintf('\nSaved closed_loop_response.png and closed_loop_metrics.csv\n');
catch closed_loop_err
    fprintf(2, '\n[WARNING] Closed-loop section failed: %s\n', closed_loop_err.message);
    fprintf(2, 'Everything above (MF plots, control surface, scenarios, rule activation)\n');
    fprintf(2, 'was saved successfully. Send me this message and I will fix it.\n');
end

%% ---------------------------------------------------------------
function firing = local_firing_strengths(fis, x)
% Per-rule firing strength computed from the FIS structure directly, using
% only membership-function parameters and rule antecedents. Works on any
% release that can build the FIS at all, so it is a safe fallback when the
% five-output evalfis signature is unavailable.
    nR = numel(fis.Rules);
    firing = zeros(nR, 1);
    for r = 1:nR
        ant = fis.Rules(r).Antecedent;
        w = 1;
        for iv = 1:numel(ant)
            k = ant(iv);
            if k == 0            % this input is unused by this rule
                continue
            end
            mf = fis.Inputs(iv).MembershipFunctions(abs(k));
            mu = local_mf_value(mf.Type, mf.Parameters, x(iv));
            if k < 0             % NOT
                mu = 1 - mu;
            end
            w = min(w, mu);      % AND = min, matching fis.AndMethod
        end
        firing(r) = w;
    end
end

function y = local_mf_value(mftype, prm, x)
% Triangular / trapezoidal membership value. Degenerate shoulders
% (a==b or c==d, i.e. the saturating end sets) cannot divide by zero here:
% those branches simply never fire.
    switch lower(char(mftype))
        case 'trimf'
            a = prm(1); b = prm(2); c = prm(2); d = prm(3);
        case 'trapmf'
            a = prm(1); b = prm(2); c = prm(3); d = prm(4);
        otherwise
            error('local_mf_value:type', 'Unsupported MF type: %s', char(mftype));
    end
    y = 0;
    if x >= b && x <= c
        y = 1;
    elseif x > a && x < b
        y = (x - a) / (b - a);
    elseif x > c && x < d
        y = (d - x) / (d - c);
    end
end
