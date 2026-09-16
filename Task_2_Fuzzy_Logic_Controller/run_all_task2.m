function run_all_task2()
% RUN_ALL_TASK2  One-command driver for the whole of Task 2.
%
% Runs Part 1 (build + analyse the FLC), Part 2 (GA membership-function
% tuning) and Part 3 (GA vs PSO on the CEC'2005 benchmarks) in order, then
% prints a single consolidated summary of every number the report needs.
%
% USAGE -- from the MATLAB Command Window:
%     cd Task_2_Fuzzy_Logic_Controller
%     run_all_task2
%
% Requires the Fuzzy Logic Toolbox. The Global Optimization Toolbox is NOT
% required (the GA and PSO are hand-written in shared/). The Statistics
% Toolbox is optional -- without it Part 3 skips the box plots and reports
% the Wilcoxon p-value as NaN, but everything else still runs.
%
% Expect roughly 5-25 minutes in total, almost all of it Part 3 (120
% optimisation runs). Progress prints as it goes.
%
% NOTE: this driver does NOT take the two GUI screenshots Part 1 needs for
% the 16-mark "diagrams and screenshots" criterion -- those cannot be
% scripted. It prints the exact commands at the end; take them yourself.

    % The sub-scripts below each begin with `clear`, which wipes this
    % function's local variables -- so no path can be held in a variable
    % across calls. The fix is to put this folder and its subfolders on the
    % MATLAB path up front: `addpath` is global state that `clear` does not
    % touch, so every script can then be called by name with no `cd` at all,
    % and `which` keeps resolving this file for the summary at the end.
    %
    % (An earlier version cd'd into each folder and re-derived the root with
    % which() each time. That broke: once the working directory moved away
    % from this folder, and with the folder not on the path, which() returned
    % empty and the cd target collapsed to a relative path.)
    ROOT = fileparts(mfilename('fullpath'));
    addpath(ROOT, ...
            fullfile(ROOT, 'shared'), ...
            fullfile(ROOT, 'part1_flc'), ...
            fullfile(ROOT, 'part2_ga_tuning'), ...
            fullfile(ROOT, 'part3_optimizer_comparison'));

    fprintf('\n========================================\n');
    fprintf('  TASK 2 -- FULL RUN\n');
    fprintf('========================================\n');
    t_all = tic;

    check_toolbox();

    % Each script resolves its own folder internally via mfilename, so it
    % writes its outputs beside itself regardless of the working directory.
    %% ---------------- PART 1 ----------------
    banner('PART 1 -- Build and analyse the FLC');
    build_flc
    analyze_flc

    %% ---------------- PART 2 ----------------
    banner('PART 2 -- GA tuning of the membership functions');
    generate_reference_dataset
    ga_tune_flc

    %% ---------------- PART 3 ----------------
    banner('PART 3 -- GA vs PSO on CEC''2005 benchmarks (this is the slow one)');
    run_comparison

    %% ---------------- SUMMARY ----------------
    summarise(toc(t_all));
end

% =====================================================================

function banner(msg)
    fprintf('\n\n----------------------------------------\n');
    fprintf('  %s\n', msg);
    fprintf('----------------------------------------\n');
end

function check_toolbox()
    if isempty(ver('fuzzy'))
        error('run_all_task2:noFuzzyToolbox', ...
            ['The Fuzzy Logic Toolbox is not installed. Install it from the ' ...
             'Add-Ons button (puzzle-piece icon in the toolbar) -- it is ' ...
             'included with your university licence.']);
    end
    if exist('ranksum', 'file') ~= 2
        fprintf(['\nNOTE: Statistics & Machine Learning Toolbox not found. Part 3 will\n' ...
                 'still run, but the GA-vs-PSO Wilcoxon p-value will be NaN and the\n' ...
                 'box plots will be skipped. Everything else is unaffected.\n']);
    end
end

function summarise(elapsed_s)
    % Safe now: addpath at the top of run_all_task2 keeps this resolvable
    % no matter what the working directory has become.
    root = fileparts(which('run_all_task2'));
    fprintf('\n\n========================================\n');
    fprintf('  EVERYTHING FINISHED in %.1f minutes\n', elapsed_s/60);
    fprintf('========================================\n');

    fprintf('\n--- PART 1: closed-loop performance ---\n');
    show_csv(fullfile(root, 'part1_flc', 'closed_loop_metrics.csv'));
    fprintf('\n--- PART 1: controller output at the 6 probe points ---\n');
    show_csv(fullfile(root, 'part1_flc', 'scenario_results.csv'));

    fprintf('\n--- PART 2: reference-dataset MSE, before vs after tuning ---\n');
    show_csv(fullfile(root, 'part2_ga_tuning', 'ga_tuning_summary.csv'));
    fprintf('\n--- PART 2: closed-loop check (the GA never optimised for this) ---\n');
    show_csv(fullfile(root, 'part2_ga_tuning', 'closed_loop_before_after.csv'));

    fprintf('\n--- PART 3: GA vs PSO, error to the known optimum ---\n');
    show_csv(fullfile(root, 'part3_optimizer_comparison', 'optimizer_comparison_results.csv'));

    fprintf('\n\n========================================\n');
    fprintf('  WHAT IS LEFT FOR YOU TO DO BY HAND\n');
    fprintf('========================================\n');
    fprintf(['\nThe two GUI screenshots cannot be scripted. Run each line, wait for\n' ...
             'the window, screenshot it (Mac: Cmd+Shift+4, then drag over the window):\n\n']);
    fprintf('    fis = readfis(''%s'');\n', ...
            fullfile(root, 'part1_flc', 'thermostat_flc.fis'));
    fprintf('    fuzzyLogicDesigner(fis)   %% -> screenshot_fis_editor.png\n');
    fprintf(['\nThen open the "Rule Inference" tab in that same window and set\n' ...
             'Input values to [-8 -1.5] (screenshot as screenshot_ruleview_cold.png),\n' ...
             'then [7 1] (screenshot as screenshot_ruleview_hot.png).\n']);
    fprintf(['\nThen upload every PNG from the three part folders, plus those\n' ...
             'screenshots, into the figures/ folder of the Overleaf project, and\n' ...
             'fill in the red \\NUM{...} placeholders in task2_report.tex using the\n' ...
             'tables printed above.\n\n']);
end

function show_csv(path)
    if exist(path, 'file') ~= 2
        fprintf('  [not found: %s]\n', path);
        return
    end
    try
        disp(readtable(path));
    catch
        type(path);
    end
end
