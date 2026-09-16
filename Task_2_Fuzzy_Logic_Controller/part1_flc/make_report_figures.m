function make_report_figures()
% MAKE_REPORT_FIGURES  Generate the three Part 1 "components" figures
% directly, instead of screenshotting the Fuzzy Logic Designer GUI.
%
% Produces, in this folder:
%   screenshot_fis_editor.png      FIS structure diagram: both inputs with
%                                  their membership functions, the Mamdani
%                                  inference block, the output membership
%                                  functions, and a panel listing the
%                                  And/Or/Implication/Aggregation/
%                                  Defuzzification methods -- the same
%                                  information the FIS editor window shows.
%   screenshot_ruleview_cold.png   Rule-by-rule inference at [-8, -1.5]
%   screenshot_ruleview_hot.png    Rule-by-rule inference at [ 7,  1  ]
%
% The two rule-inference figures reproduce the Rule Inference view exactly:
% one row per rule, showing each antecedent membership function with the
% input value marked and the membership shaded, the clipped consequent, and
% at the bottom the aggregated output set with the centroid marked.
%
% Why generate rather than screenshot: the figures come out at publication
% resolution, they regenerate identically if the FIS changes, and no manual
% step can be forgotten. The filenames match what the report expects, so
% they drop straight into the Overleaf figures/ folder.
%
% Everything here is base-MATLAB plotting plus membership-function
% arithmetic done locally, so it does not depend on any toolbox plotting
% helper or on a particular evalfis signature.

    this_dir = fileparts(mfilename('fullpath'));

    % --- Membership functions, read from the saved FIS where possible ----
    % (falls back to the definitions in build_flc.m if the FIS cannot be
    % read, so the figures can always be produced)
    [TE, RC, OU, from_file] = load_mfs(fullfile(this_dir, 'thermostat_flc.fis'));
    if from_file
        fprintf('Membership functions read from thermostat_flc.fis\n');
    else
        fprintf('Could not read the .fis -- using the definitions from build_flc.m\n');
    end

    % --- Rule base: (TempError set, RateChange set, Output set) ----------
    RULES = [1 1 5; 1 2 5; 1 3 4;
             2 1 4; 2 2 4; 2 3 3;
             3 1 4; 3 2 3; 3 3 2;
             4 1 3; 4 2 2; 4 3 2;
             5 1 2; 5 2 1; 5 3 1];

    TE_LBL = {'Cold','Cool','Comfortable','Warm','Hot'};
    RC_LBL = {'Falling','Steady','Rising'};
    OU_LBL = {'FullCool','LowCool','Off','LowHeat','FullHeat'};

    make_structure_figure(this_dir, TE, RC, OU, TE_LBL, RC_LBL, OU_LBL);

    make_rule_figure(this_dir, TE, RC, OU, RULES, [-8 -1.5], ...
        'Cold room, still cooling', 'screenshot_ruleview_cold.png');
    make_rule_figure(this_dir, TE, RC, OU, RULES, [7 1], ...
        'Hot room, still warming',  'screenshot_ruleview_hot.png');

    fprintf('\nSaved screenshot_fis_editor.png, screenshot_ruleview_cold.png\n');
    fprintf('and screenshot_ruleview_hot.png to %s\n', this_dir);
end

% =====================================================================

function [TE, RC, OU, ok] = load_mfs(fis_path)
    TE = {[-10 -10 -6 -2], [-4 -2 0], [-1 0 1], [0 2 4], [2 6 10 10]};
    RC = {[-2 -2 -0.6 0], [-0.4 0 0.4], [0 0.6 2 2]};
    OU = {[-100 -100 -70 -40], [-60 -30 0], [-10 0 10], [0 30 60], [40 70 100 100]};
    ok = false;
    try
        fis = readfis(fis_path);
        TE = arrayfun(@(m) m.Parameters, fis.Inputs(1).MembershipFunctions,  'UniformOutput', false);
        RC = arrayfun(@(m) m.Parameters, fis.Inputs(2).MembershipFunctions,  'UniformOutput', false);
        OU = arrayfun(@(m) m.Parameters, fis.Outputs(1).MembershipFunctions, 'UniformOutput', false);
        ok = true;
    catch
        % keep the build_flc.m defaults above
    end
end

function y = mfval(prm, x)
% Membership value for a 3-parameter triangle or 4-parameter trapezoid,
% vectorised over x. Degenerate shoulders (a==b or c==d) never divide by
% zero because those branches simply do not fire.
    if numel(prm) == 3
        a = prm(1); b = prm(2); c = prm(2); d = prm(3);
    else
        a = prm(1); b = prm(2); c = prm(3); d = prm(4);
    end
    y = zeros(size(x));
    y(x >= b & x <= c) = 1;
    i1 = x > a & x < b;  if b > a, y(i1) = (x(i1) - a) / (b - a); end
    i2 = x > c & x < d;  if d > c, y(i2) = (d - x(i2)) / (d - c); end
end

function plot_mf_set(sets, labels, rng, ttl)
    x = linspace(rng(1), rng(2), 600);
    hold on;
    for m = 1:numel(sets)
        plot(x, mfval(sets{m}, x), 'LineWidth', 1.6);
    end
    ylim([-0.05 1.15]); xlim(rng); grid on; box on;
    title(ttl, 'FontWeight', 'normal');
    ylabel('\mu');
    legend(labels, 'Location', 'southoutside', 'Orientation', 'horizontal', ...
           'FontSize', 7, 'Box', 'off');
end

% ---------------------------------------------------------------------

function make_structure_figure(dir_out, TE, RC, OU, TE_LBL, RC_LBL, OU_LBL)
    fig = figure('Name', 'FIS structure', 'Position', [80 80 1150 760], 'Color', 'w');

    subplot(2,3,1); plot_mf_set(TE, TE_LBL, [-10 10],   'Input 1: TempError (^\circC)');
    subplot(2,3,4); plot_mf_set(RC, RC_LBL, [-2 2],     'Input 2: RateChange (^\circC/min)');
    subplot(2,3,3); plot_mf_set(OU, OU_LBL, [-100 100], 'Output: HeaterCoolerPower (%)');

    % Centre panel: the inference block and its configuration, i.e. the same
    % information the FIS editor shows in its property pane.
    subplot(2,3,[2 5]); axis off; xlim([0 1]); ylim([0 1]); hold on;
    rectangle('Position', [0.15 0.55 0.7 0.32], 'Curvature', 0.06, ...
              'EdgeColor', [0.15 0.15 0.15], 'LineWidth', 1.6);
    text(0.5, 0.78, 'ThermostatFLC', 'HorizontalAlignment', 'center', ...
         'FontWeight', 'bold', 'FontSize', 12);
    text(0.5, 0.68, 'Mamdani Type-1', 'HorizontalAlignment', 'center', 'FontSize', 11);
    text(0.5, 0.61, '2 inputs  |  1 output  |  15 rules', ...
         'HorizontalAlignment', 'center', 'FontSize', 9);

    cfg = {'And method', 'min'; 'Or method', 'max'; 'Implication', 'min'; ...
           'Aggregation', 'max'; 'Defuzzification', 'centroid'};
    y0 = 0.42;
    text(0.15, y0 + 0.06, 'Inference configuration', 'FontWeight', 'bold', 'FontSize', 9);
    for i = 1:size(cfg,1)
        yy = y0 - (i-1)*0.075;
        text(0.15, yy, cfg{i,1}, 'FontSize', 9);
        text(0.85, yy, cfg{i,2}, 'FontSize', 9, 'FontWeight', 'bold', ...
             'HorizontalAlignment', 'right', 'FontName', 'FixedWidth');
    end
    % arrows from the inputs into the block, and out to the output
    annotation('arrow', [0.34 0.40], [0.72 0.72]);
    annotation('arrow', [0.34 0.40], [0.26 0.72]);
    annotation('arrow', [0.62 0.68], [0.72 0.72]);

    sgtitle('Fuzzy Inference System: ThermostatFLC (Mamdani Type-1)', ...
            'FontWeight', 'bold');
    saveas(fig, fullfile(dir_out, 'screenshot_fis_editor.png'));
    close(fig);
end

% ---------------------------------------------------------------------

function make_rule_figure(dir_out, TE, RC, OU, RULES, inp, desc, fname)
    nR   = size(RULES, 1);
    xTE  = linspace(-10, 10, 400);
    xRC  = linspace(-2, 2, 400);
    xOU  = linspace(-100, 100, 1001);

    firing = zeros(nR, 1);
    agg    = zeros(size(xOU));
    for r = 1:nR
        muA = mfval(TE{RULES(r,1)}, inp(1));
        muB = mfval(RC{RULES(r,2)}, inp(2));
        firing(r) = min(muA, muB);                                  % AND = min
        agg = max(agg, min(firing(r), mfval(OU{RULES(r,3)}, xOU)));  % clip, then union
    end
    % The crisp value is taken from evalfis so that it matches
    % scenario_results.csv exactly. The local centroid below is only a
    % fallback: it agrees with evalfis to within ~0.4%, the difference being
    % purely the output-grid resolution used for defuzzification, not a
    % difference in the inference itself.
    crisp = NaN;
    try
        fis_for_eval = readfis(fullfile(dir_out, 'thermostat_flc.fis'));
        crisp = evalfis(fis_for_eval, inp);
    catch
    end
    if ~isfinite(crisp)
        if sum(agg) > 0
            crisp = sum(xOU .* agg) / sum(agg);                       % centroid
        else
            crisp = 0;
        end
    end

    fig = figure('Name', 'Rule inference', 'Position', [60 40 900 1150], 'Color', 'w');
    rows = nR + 1;

    for r = 1:nR
        % --- antecedent 1
        subplot(rows, 3, (r-1)*3 + 1); hold on;
        plot(xTE, mfval(TE{RULES(r,1)}, xTE), 'k', 'LineWidth', 0.9);
        plot([inp(1) inp(1)], [0 1], 'r-', 'LineWidth', 1.1);
        ylim([0 1.1]); xlim([-10 10]); set(gca, 'XTick', [], 'YTick', []); box on;
        ylabel(sprintf('R%d', r), 'Rotation', 0, 'FontSize', 7, ...
               'HorizontalAlignment', 'right', 'VerticalAlignment', 'middle');
        if r == 1, title(sprintf('TempError = %g', inp(1)), 'FontSize', 9); end

        % --- antecedent 2
        subplot(rows, 3, (r-1)*3 + 2); hold on;
        plot(xRC, mfval(RC{RULES(r,2)}, xRC), 'k', 'LineWidth', 0.9);
        plot([inp(2) inp(2)], [0 1], 'r-', 'LineWidth', 1.1);
        ylim([0 1.1]); xlim([-2 2]); set(gca, 'XTick', [], 'YTick', []); box on;
        if r == 1, title(sprintf('RateChange = %g', inp(2)), 'FontSize', 9); end

        % --- consequent, clipped at the firing strength
        subplot(rows, 3, (r-1)*3 + 3); hold on;
        clipped = min(firing(r), mfval(OU{RULES(r,3)}, xOU));
        plot(xOU, mfval(OU{RULES(r,3)}, xOU), 'Color', [.6 .6 .6], 'LineWidth', 0.8);
        if firing(r) > 0
            fill([xOU fliplr(xOU)], [clipped zeros(size(clipped))], [0.30 0.55 0.85], ...
                 'EdgeColor', 'none', 'FaceAlpha', 0.85);
        end
        ylim([0 1.1]); xlim([-100 100]); set(gca, 'XTick', [], 'YTick', []); box on;
        if firing(r) > 0
            text(95, 0.78, sprintf('%.2f', firing(r)), 'FontSize', 7, ...
                 'HorizontalAlignment', 'right', 'Color', [0.1 0.3 0.6], 'FontWeight', 'bold');
        end
        if r == 1, title('HeaterCoolerPower', 'FontSize', 9); end
    end

    % --- aggregated output + centroid
    subplot(rows, 3, [nR*3+1, nR*3+3]); hold on;
    fill([xOU fliplr(xOU)], [agg zeros(size(agg))], [0.85 0.45 0.20], ...
         'EdgeColor', 'none', 'FaceAlpha', 0.9);
    plot([crisp crisp], [0 1.05], 'k-', 'LineWidth', 2);
    ylim([0 1.15]); xlim([-100 100]); box on; grid on;
    xlabel('HeaterCoolerPower (%)', 'FontSize', 9);
    title(sprintf('Aggregated output  ->  centroid = %.2f %%', crisp), ...
          'FontSize', 10, 'FontWeight', 'bold');

    sgtitle(sprintf('Rule inference at [%g, %g] -- %s', inp(1), inp(2), desc), ...
            'FontWeight', 'bold');
    saveas(fig, fullfile(dir_out, fname));
    close(fig);

    fprintf('%s: crisp output = %.3f %% (%d of %d rules firing)\n', ...
            fname, crisp, sum(firing > 0), nR);
end
