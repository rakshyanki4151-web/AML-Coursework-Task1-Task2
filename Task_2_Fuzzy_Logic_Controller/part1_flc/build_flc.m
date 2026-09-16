% build_flc.m
%
% Task 2, Part 1: Design and construct a Mamdani Fuzzy Logic Controller
% (FLC) for temperature regulation in one room of an intelligent
% assistive-care flat, controlling a combined heater/cooler actuator.
%
% Control behaviour chosen: a classic fuzzy PD (proportional-derivative)
% thermostat. Two inputs describe the current thermal state (how far off
% the setpoint the room is, and whether it is getting warmer or colder),
% and one output drives the heater/cooler.
%
% Inputs:
%   TempError   = room_temperature - setpoint_temperature   (deg C, range [-10, 10])
%                 negative = room is colder than desired; positive = hotter.
%   RateChange  = d(TempError)/dt over the last sample interval (deg C/min, range [-2, 2])
%                 negative = room is cooling further; positive = warming further.
%
% Output:
%   HeaterCoolerPower = actuator command, range [-100, 100] %
%                 negative = cooling power; positive = heating power; 0 = off.
%
% This design is deliberately a fuzzy PD controller: TempError plays the
% role of the "proportional" term and RateChange the "derivative" term,
% which is the standard, well-justified way to give a thermostat both
% current-error awareness and trend awareness (so it eases off heating
% just before reaching setpoint, rather than overshooting).
%
% Run this script in MATLAB (Fuzzy Logic Toolbox required) to build and
% save the FIS. Part 1's "diagrams and screenshots" requirement is met by:
%   (a) the auto-saved membership-function plots and control-surface plot
%       this script produces (see analyze_flc.m), and
%   (b) two GUI screenshots you take yourself (instructions printed at the
%       end of this script): the FIS editor and the rule viewer.

clear; clc;

%% 1. Create the Mamdani FIS
fis = mamfis('Name', 'ThermostatFLC');

%% 2. Input 1: TempError, range [-10, 10] deg C, 5 fuzzy sets
fis = addInput(fis, [-10 10], 'Name', 'TempError');
fis = addMF(fis, 'TempError', 'trapmf', [-10 -10 -6 -2], 'Name', 'Cold');       % NB
fis = addMF(fis, 'TempError', 'trimf',  [-4 -2 0],       'Name', 'Cool');       % NS
fis = addMF(fis, 'TempError', 'trimf',  [-1 0 1],        'Name', 'Comfortable');% ZE
fis = addMF(fis, 'TempError', 'trimf',  [0 2 4],         'Name', 'Warm');       % PS
fis = addMF(fis, 'TempError', 'trapmf', [2 6 10 10],     'Name', 'Hot');        % PB

%% 3. Input 2: RateChange, range [-2, 2] deg C/min, 3 fuzzy sets
fis = addInput(fis, [-2 2], 'Name', 'RateChange');
fis = addMF(fis, 'RateChange', 'trapmf', [-2 -2 -0.6 0], 'Name', 'Falling');
fis = addMF(fis, 'RateChange', 'trimf',  [-0.4 0 0.4],   'Name', 'Steady');
fis = addMF(fis, 'RateChange', 'trapmf', [0 0.6 2 2],    'Name', 'Rising');

%% 4. Output: HeaterCoolerPower, range [-100, 100] %, 5 fuzzy sets
fis = addOutput(fis, [-100 100], 'Name', 'HeaterCoolerPower');
fis = addMF(fis, 'HeaterCoolerPower', 'trapmf', [-100 -100 -70 -40], 'Name', 'FullCool');
fis = addMF(fis, 'HeaterCoolerPower', 'trimf',  [-60 -30 0],         'Name', 'LowCool');
fis = addMF(fis, 'HeaterCoolerPower', 'trimf',  [-10 0 10],          'Name', 'Off');
fis = addMF(fis, 'HeaterCoolerPower', 'trimf',  [0 30 60],           'Name', 'LowHeat');
fis = addMF(fis, 'HeaterCoolerPower', 'trapmf', [40 70 100 100],     'Name', 'FullHeat');

%% 5. Rule base (5 TempError sets x 3 RateChange sets = 15 rules)
%
% Design logic (a standard fuzzy PD table): heat harder when the room is
% cold AND still cooling; ease off heating as it warms back toward
% setpoint (RateChange = Rising) even while still Cold/Cool, to reduce
% overshoot; mirror the same logic for cooling on the hot side.
rules = [
    "TempError==Cold & RateChange==Falling => HeaterCoolerPower=FullHeat (1)"
    "TempError==Cold & RateChange==Steady  => HeaterCoolerPower=FullHeat (1)"
    "TempError==Cold & RateChange==Rising  => HeaterCoolerPower=LowHeat (1)"

    "TempError==Cool & RateChange==Falling => HeaterCoolerPower=LowHeat (1)"
    "TempError==Cool & RateChange==Steady  => HeaterCoolerPower=LowHeat (1)"
    "TempError==Cool & RateChange==Rising  => HeaterCoolerPower=Off (1)"

    "TempError==Comfortable & RateChange==Falling => HeaterCoolerPower=LowHeat (1)"
    "TempError==Comfortable & RateChange==Steady  => HeaterCoolerPower=Off (1)"
    "TempError==Comfortable & RateChange==Rising  => HeaterCoolerPower=LowCool (1)"

    "TempError==Warm & RateChange==Falling => HeaterCoolerPower=Off (1)"
    "TempError==Warm & RateChange==Steady  => HeaterCoolerPower=LowCool (1)"
    "TempError==Warm & RateChange==Rising  => HeaterCoolerPower=LowCool (1)"

    "TempError==Hot & RateChange==Falling => HeaterCoolerPower=LowCool (1)"
    "TempError==Hot & RateChange==Steady  => HeaterCoolerPower=FullCool (1)"
    "TempError==Hot & RateChange==Rising  => HeaterCoolerPower=FullCool (1)"
];
fis = addRule(fis, rules);

%% 6. Inference / defuzzification method (explicit, for the write-up)
fis.AndMethod = 'min';        % rule antecedent conjunction: min (Mamdani standard)
fis.OrMethod = 'max';
fis.ImplicationMethod = 'min';   % rule implication (clipping)
fis.AggregationMethod = 'max';   % combine all fired rules' output sets
fis.DefuzzificationMethod = 'centroid'; % centre-of-gravity defuzzification

%% 7. Save the FIS for use by analyze_flc.m and Part 2's GA tuning
save_dir = fileparts(mfilename('fullpath'));
if exist('writeFIS', 'file')
    writeFIS(fis, fullfile(save_dir, 'thermostat_flc'));   % R2018b and later
else
    writefis(fis, fullfile(save_dir, 'thermostat_flc'));   % legacy spelling
end
fprintf('Saved thermostat_flc.fis to %s\n', save_dir);

%% 8. Screenshots you need to take manually (GUI-only, cannot be scripted)
fprintf('\n--- Manual screenshot checklist for the report ---\n');
fprintf('1. Run:  fuzzyLogicDesigner(fis)   -> screenshot the FIS editor\n');
fprintf('         (shows the And/Or/Implication/Aggregation/Defuzz methods)\n');
fprintf('2. In that same window, open the "Rule Inference" tab -> screenshot it.\n');
fprintf('         Set Input values to [-8 -1.5] (cold & falling), screenshot, then\n');
fprintf('         [7 1] (hot & rising) and screenshot again -- two clearly different\n');
fprintf('         operating points firing different subsets of the rules.\n');
fprintf('         (The old ruleview(fis) command was removed in recent releases.)\n');
fprintf('Then run analyze_flc.m for the auto-generated membership-function and\n');
fprintf('control-surface plots (no GUI needed, saved directly as PNG).\n');
