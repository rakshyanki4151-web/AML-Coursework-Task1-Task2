function [t, T, u, metrics] = simulate_room(controller, opts)
% SIMULATE_ROOM  Closed-loop simulation of one room in the assistive-care
% flat under a given controller, used for Part 1's "operational scenario"
% analysis and Part 2's before/after validation of the GA-tuned FLC.
%
% Plant (first-order lumped thermal model of a room + heater/cooler):
%
%   dT/dt = (T_out - T)/tau  +  k_act * u/100
%
%   T      room air temperature (deg C)
%   T_out  outdoor temperature (deg C) -- the disturbance the controller fights
%   tau    thermal time constant of the room (min): how fast it leaks heat
%   k_act  actuator authority (deg C/min at full +-100% power)
%   u      controller command in [-100, 100] % (negative = cooling)
%
% This is the standard first-order RC model of a room's thermal envelope;
% it is deliberately simple but captures the two things that make the
% control problem non-trivial: the room's thermal lag (which is what makes
% a derivative/RateChange term worth having, since a pure proportional
% controller overshoots) and a continuous outdoor heat-loss disturbance
% (which is what forces a non-zero steady-state actuator command).
%
% Inputs:
%   controller : either
%                  - a FIS object/struct (evaluated with evalfis), or
%                  - a function handle u = controller(e, de)
%                to allow a classical PD controller to be compared against
%                the FLC on the identical plant.
%   opts       : struct, optional fields (defaults shown):
%                  setpoint  = 22     desired room temperature (deg C)
%                  T0        = 16     initial room temperature (deg C)
%                  T_out     = 5      outdoor temperature (deg C)
%                  tau       = 60     room thermal time constant (min)
%                  k_act     = 0.5    deg C/min at full power
%
% REACHABILITY CONSTRAINT -- check this whenever you change the plant:
% at equilibrium dT/dt = 0 with u = 100%, so the hottest the room can ever
% reach is  T_out + tau*k_act.  The setpoint MUST sit below that, with
% margin, or the controller saturates at full power, never settles, and the
% comparison between two controllers becomes meaningless (both just sit at
% 100%). With the defaults here: 5 + 60*0.5 = 35 degC ceiling against a
% 22 degC setpoint, needing about 57% power to hold -- comfortably inside
% the actuator's range.
%                  dt        = 1      simulation step (min)
%                  t_end     = 180    simulation length (min)
%                  band      = 0.5    settling band around setpoint (deg C)
%                  noise_std = 0      sensor noise std on the temperature reading
%
% Outputs:
%   t, T, u  : time (min), room temperature trace, actuator command trace
%   metrics  : struct with settling_time_min, overshoot_degC,
%              steady_state_error_degC, IAE (integral of absolute error),
%              and energy_pct_min (integral of |u| -- a proxy for the
%              energy the controller spends, which matters directly for the
%              assistive-care brief's power-management motivation).

    if nargin < 2, opts = struct(); end
    setpoint  = getopt(opts, 'setpoint', 22);
    T0        = getopt(opts, 'T0', 16);
    T_out     = getopt(opts, 'T_out', 5);
    tau       = getopt(opts, 'tau', 60);
    k_act     = getopt(opts, 'k_act', 0.5);
    dt        = getopt(opts, 'dt', 1);
    t_end     = getopt(opts, 't_end', 180);
    band      = getopt(opts, 'band', 0.5);
    noise_std = getopt(opts, 'noise_std', 0);

    n = round(t_end / dt) + 1;
    t = (0:n-1)' * dt;
    T = zeros(n, 1);
    u = zeros(n, 1);
    T(1) = T0;

    e_prev = T0 - setpoint;   % TempError uses the same sign convention as the FLC:
                              % negative = room colder than desired.
    for k = 1:n-1
        T_meas = T(k) + noise_std * randn();
        e  = T_meas - setpoint;
        de = (e - e_prev) / dt;                 % deg C per minute

        % Saturate to the ranges the FLC's universes of discourse cover, so
        % an out-of-range excursion cannot produce an unfired (NaN) rule base.
        e_sat  = max(min(e,  10), -10);
        de_sat = max(min(de,  2),  -2);

        if isa(controller, 'function_handle')
            u(k) = controller(e_sat, de_sat);
        else
            u(k) = evalfis(controller, [e_sat de_sat]);
        end
        u(k) = max(min(u(k), 100), -100);       % actuator saturation

        % Forward-Euler integration of the plant
        dT = (T_out - T(k)) / tau + k_act * u(k) / 100;
        T(k+1) = T(k) + dt * dT;
        e_prev = e;
    end
    u(n) = u(n-1);

    %% Performance metrics
    err = T - setpoint;
    metrics.IAE = sum(abs(err)) * dt;
    metrics.energy_pct_min = sum(abs(u)) * dt;

    % Overshoot: the largest excursion PAST the setpoint, in the direction
    % of travel (here the room starts cold, so overshoot means going above).
    if T0 < setpoint
        metrics.overshoot_degC = max(0, max(T) - setpoint);
    else
        metrics.overshoot_degC = max(0, setpoint - min(T));
    end

    % Settling time: first time after which |error| stays inside the band.
    outside = find(abs(err) > band);
    if isempty(outside)
        metrics.settling_time_min = 0;
    elseif outside(end) == n
        metrics.settling_time_min = NaN;        % never settled within t_end
    else
        metrics.settling_time_min = t(outside(end) + 1);
    end

    % Steady-state error: mean error over the final 20% of the run.
    tail = max(1, round(0.8*n)) : n;
    metrics.steady_state_error_degC = mean(err(tail));
end

function v = getopt(opts, field, default)
    if isfield(opts, field)
        v = opts.(field);
    else
        v = default;
    end
end
