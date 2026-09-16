% generate_reference_dataset.m
%
% Task 2, Part 2 needs "a data set of n input-output examples (xi, yi)"
% to evaluate/tune the FLC against. Since this is a simulated smart-home
% controller with no real deployed sensor log, the dataset is generated
% from a simple analytic reference control law -- a textbook linear PD
% controller, u_ref = -Kp*e - Kd*de, clipped to the actuator's [-100,100]
% range -- sampled across the same operating envelope the FLC covers, with
% added Gaussian sensor noise. This is a standard, defensible way to get an
% evaluation dataset for FLC tuning when no logged deployment data exists:
% the GA in ga_tune_flc.m then adjusts the FLC's membership functions so
% its fuzzy output matches this reference law as closely as possible.
%
% NOTE FOR THE REPORT: state this dataset-generation choice explicitly
% (Experimental Setup section) -- it is a modelling assumption, not real
% sensor data, and should be described as such rather than implied to be
% measured.

clear; clc;
rng(7); % fixed seed for a reproducible dataset

n = 300;
Kp = 8;   % proportional gain
Kd = 15;  % derivative gain
noise_std = 3; % percent, simulates sensor/actuator noise

TempError  = -10 + 20*rand(n,1);   % uniform over [-10, 10]
RateChange = -2  + 4*rand(n,1);    % uniform over [-2, 2]

TargetPower = -Kp*TempError - Kd*RateChange + noise_std*randn(n,1);
TargetPower = max(min(TargetPower, 100), -100); % actuator saturation

T = table(TempError, RateChange, TargetPower);
this_dir = fileparts(mfilename('fullpath'));
writetable(T, fullfile(this_dir, 'reference_dataset.csv'));

fprintf('Saved reference_dataset.csv with %d examples.\n', n);
fprintf('Reference law: u = -%.1f*TempError - %.1f*RateChange + noise(std=%.1f)\n', ...
    Kp, Kd, noise_std);
