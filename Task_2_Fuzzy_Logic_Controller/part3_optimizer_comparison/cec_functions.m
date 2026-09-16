function f = cec_functions(name, x)
% CEC_FUNCTIONS  Shifted benchmark functions in the style of the CEC'2005
% special-session suite (Sphere = F1, Rastrigin = F9).
%
% IMPORTANT, STATE THIS IN THE REPORT: the official CEC'2005 suite ships
% specific bias data files and shift vectors (o_i) downloaded from the
% competition's own archive. Since that download was not available in this
% environment, this file instead uses the officially documented bias
% constants (-450 for F1, -330 for F9) together with a fixed,
% reproducible pseudo-random shift vector (seeded, see below), which
% preserves the core property the shift is meant to test -- that the
% optimum is not at the trivial origin, so an optimiser cannot exploit an
% implicit "search near zero" bias -- while not exactly reproducing the
% published data files. This is a documented, deliberate simplification,
% not an oversight.
%
% Usage:
%   f = cec_functions('sphere', x)     % x: 1 x D or N x D
%   f = cec_functions('rastrigin', x)
%
% Domains (also documented here for run_comparison.m):
%   sphere:    x in [-100, 100]^D
%   rastrigin: x in [-5, 5]^D

    D = size(x, 2);
    o = shift_vector(name, D);

    switch name
        case 'sphere'
            bias = -450;
            z = x - o;
            f = sum(z.^2, 2) + bias;

        case 'rastrigin'
            bias = -330;
            z = x - o;
            f = sum(z.^2 - 10*cos(2*pi*z) + 10, 2) + bias;

        otherwise
            error('Unknown function name: %s', name);
    end
end

function o = shift_vector(name, D)
% Fixed, reproducible shift vector per function name and dimension, so the
% same optimum location is used across every run/optimiser/dimension
% comparison (a different seed would just relocate the optimum, not change
% the difficulty, but keeping it fixed makes cross-run comparison exact).
%
% IMPLEMENTATION NOTE: this uses a private RandStream rather than rng(seed)
% deliberately -- this function is called on every single fitness
% evaluation from inside custom_ga/custom_pso's optimisation loop, and
% those optimisers rely on MATLAB's *global* random stream for their own
% crossover/mutation/velocity randomness. Calling rng(seed) here would
% reset that global stream on every fitness call and silently destroy the
% optimiser's own randomness. A local RandStream object is fully isolated
% from the global stream, so it reproduces the same shift vector every
% time without interfering with anything else.
    switch name
        case 'sphere'
            seed = 501;
            domain_half = 80; % keep the shift well inside [-100,100]
        case 'rastrigin'
            seed = 509;
            domain_half = 4;  % keep the shift well inside [-5,5]
        otherwise
            error('Unknown function name: %s', name);
    end
    stream = RandStream('mt19937ar', 'Seed', seed);
    o = -domain_half + 2*domain_half*rand(stream, 1, D);
end
