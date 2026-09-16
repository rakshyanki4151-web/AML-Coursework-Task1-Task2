function [best_x, best_f, history] = custom_pso(fitness_fn, lb, ub, opts)
% CUSTOM_PSO  Particle Swarm Optimisation for continuous minimisation,
% following the same velocity/position update formulation used in Task 1's
% Python PSO (src/pso.py), reimplemented in MATLAB for Task 2 Part 3's
% optimiser comparison. Kept general-purpose (arbitrary dimension D) so it
% can be reused directly on the CEC'2005 benchmark functions.
%
% Inputs:
%   fitness_fn : function handle, fitness_fn(x) -> scalar (to MINIMISE),
%                x is a 1 x D row vector.
%   lb, ub     : 1 x D row vectors, lower/upper bounds per dimension.
%   opts       : struct with optional fields (defaults shown):
%                  n_particles   = 30
%                  n_iterations  = 100
%                  inertia       = 0.6
%                  cognitive     = 1.5
%                  social        = 1.5
%                  velocity_clip = 0.2   (fraction of search-box span)
%                  seed          = 42
%
% Outputs:
%   best_x, best_f : best position/fitness found (global best)
%   history        : n_iterations+1 x 1 vector, best-so-far fitness per iteration

    if nargin < 4, opts = struct(); end
    n_particles   = getopt(opts, 'n_particles', 30);
    n_iterations  = getopt(opts, 'n_iterations', 100);
    inertia       = getopt(opts, 'inertia', 0.6);
    cognitive     = getopt(opts, 'cognitive', 1.5);
    social        = getopt(opts, 'social', 1.5);
    velocity_clip = getopt(opts, 'velocity_clip', 0.2);
    seed          = getopt(opts, 'seed', 42);

    rng(seed);
    D = numel(lb);
    span = ub - lb;

    positions = lb + rand(n_particles, D) .* span;
    velocities = (rand(n_particles, D) - 0.5) .* span * 0.1;

    fitness = arrayfun(@(i) fitness_fn(positions(i,:)), (1:n_particles)');
    pbest_pos = positions;
    pbest_fit = fitness;
    [gbest_fit, gidx] = min(pbest_fit);
    gbest_pos = pbest_pos(gidx, :);

    history = zeros(n_iterations+1, 1);
    history(1) = gbest_fit;

    v_clip = velocity_clip * span;

    for iter = 1:n_iterations
        r1 = rand(n_particles, D);
        r2 = rand(n_particles, D);

        velocities = inertia .* velocities ...
            + cognitive .* r1 .* (pbest_pos - positions) ...
            + social    .* r2 .* (gbest_pos - positions);
        velocities = max(min(velocities, v_clip), -v_clip);

        positions = positions + velocities;
        positions = max(min(positions, ub), lb);

        fitness = arrayfun(@(i) fitness_fn(positions(i,:)), (1:n_particles)');

        improved = fitness < pbest_fit;
        pbest_pos(improved, :) = positions(improved, :);
        pbest_fit(improved) = fitness(improved);

        [iter_best_fit, iter_idx] = min(pbest_fit);
        if iter_best_fit < gbest_fit
            gbest_fit = iter_best_fit;
            gbest_pos = pbest_pos(iter_idx, :);
        end
        history(iter+1) = gbest_fit;
    end

    best_x = gbest_pos;
    best_f = gbest_fit;
end

function v = getopt(opts, field, default)
    if isfield(opts, field)
        v = opts.(field);
    else
        v = default;
    end
end
