function [best_x, best_f, history] = custom_ga(fitness_fn, lb, ub, opts)
% CUSTOM_GA  A real-coded Genetic Algorithm for continuous minimisation,
% implemented from first principles (no Global Optimization Toolbox
% dependency), so the same optimiser can be reused for Task 2 Part 2
% (FLC membership-function tuning) and Part 3 (CEC'2005 benchmark
% comparison against PSO), and so every operator is auditable in the
% report rather than treated as a black box.
%
% Inputs:
%   fitness_fn : function handle, fitness_fn(x) -> scalar (to MINIMISE),
%                x is a 1 x D row vector.
%   lb, ub     : 1 x D row vectors, lower/upper bounds per gene.
%   opts       : struct with optional fields (defaults shown):
%                  pop_size    = 40
%                  n_generations = 60
%                  crossover_rate = 0.8
%                  mutation_rate  = 0.1     (per-gene probability)
%                  mutation_scale = 0.1     (fraction of gene range)
%                  tournament_k   = 3
%                  elite_count    = 2
%                  seed           = 42
%
% Outputs:
%   best_x   : 1 x D best solution found
%   best_f   : scalar, its fitness
%   history  : n_generations+1 x 1 vector, best-so-far fitness per generation
%
% Genetic operators used (state explicitly for the report):
%   Selection : tournament selection (size tournament_k)
%   Crossover : BLX-alpha (blend) crossover, alpha = 0.3
%   Mutation  : Gaussian mutation, per-gene probability mutation_rate,
%               std = mutation_scale * (ub-lb)
%   Elitism   : the elite_count best individuals are carried over unchanged

    if nargin < 4, opts = struct(); end
    pop_size       = getopt(opts, 'pop_size', 40);
    n_generations  = getopt(opts, 'n_generations', 60);
    crossover_rate = getopt(opts, 'crossover_rate', 0.8);
    mutation_rate  = getopt(opts, 'mutation_rate', 0.1);
    mutation_scale = getopt(opts, 'mutation_scale', 0.1);
    tournament_k   = getopt(opts, 'tournament_k', 3);
    elite_count    = getopt(opts, 'elite_count', 2);
    seed           = getopt(opts, 'seed', 42);

    rng(seed);
    D = numel(lb);
    span = ub - lb;

    % --- Initialise population uniformly within bounds ---
    pop = lb + rand(pop_size, D) .* span;
    fitness = arrayfun(@(i) fitness_fn(pop(i,:)), (1:pop_size)');

    [best_f, idx] = min(fitness);
    best_x = pop(idx,:);
    history = zeros(n_generations+1, 1);
    history(1) = best_f;

    for gen = 1:n_generations
        % --- Elitism: carry the best individuals forward unchanged ---
        [~, order] = sort(fitness, 'ascend');
        elite_idx = order(1:elite_count);
        new_pop = pop(elite_idx, :);

        % --- Fill the rest of the new population ---
        while size(new_pop, 1) < pop_size
            parent1 = tournament_select(pop, fitness, tournament_k);
            parent2 = tournament_select(pop, fitness, tournament_k);

            if rand() < crossover_rate
                alpha = 0.3;
                lo = min(parent1, parent2) - alpha .* abs(parent1 - parent2);
                hi = max(parent1, parent2) + alpha .* abs(parent1 - parent2);
                child = lo + rand(1, D) .* (hi - lo);
            else
                child = parent1;
            end

            mutate_mask = rand(1, D) < mutation_rate;
            child(mutate_mask) = child(mutate_mask) + ...
                randn(1, sum(mutate_mask)) .* (mutation_scale * span(mutate_mask));

            child = min(max(child, lb), ub); % clip to bounds
            new_pop = [new_pop; child]; %#ok<AGROW>
        end

        pop = new_pop(1:pop_size, :);
        fitness = arrayfun(@(i) fitness_fn(pop(i,:)), (1:pop_size)');

        [gen_best_f, gen_idx] = min(fitness);
        if gen_best_f < best_f
            best_f = gen_best_f;
            best_x = pop(gen_idx, :);
        end
        history(gen+1) = best_f;
    end
end

function parent = tournament_select(pop, fitness, k)
    n = size(pop, 1);
    contenders = randi(n, 1, k);
    [~, best_local] = min(fitness(contenders));
    parent = pop(contenders(best_local), :);
end

function v = getopt(opts, field, default)
    if isfield(opts, field)
        v = opts.(field);
    else
        v = default;
    end
end
