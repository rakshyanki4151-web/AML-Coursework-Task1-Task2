function fis = apply_chromosome(base_fis, chromosome)
% APPLY_CHROMOSOME  Decode a 6-gene chromosome into a modified copy of
% base_fis's membership functions, keeping the same number of fuzzy sets,
% the same MF shape types (trimf/trapmf), and the same rule base as
% Part 1 -- only the *shape* of each variable's membership functions is
% adjusted, via a per-variable affine transform (shift + scale) applied
% uniformly to every breakpoint of every set belonging to that variable.
%
% Chromosome layout (length 6, matching Section "Genetic Algorithm Design"
% in the report):
%   [1] TempError  center shift   (deg C,   bounds [-2.0,  2.0])
%   [2] TempError  spread scale   (unitless, bounds [0.6, 1.6])
%   [3] RateChange center shift   (deg C/min, bounds [-0.4, 0.4])
%   [4] RateChange spread scale   (unitless, bounds [0.6, 1.6])
%   [5] Output     center shift   (%,       bounds [-15,  15])
%   [6] Output     spread scale   (unitless, bounds [0.6, 1.6])
%
% Why this encoding: optimising all 17+11+17=45 raw breakpoints directly
% would risk producing invalid (non-monotonic, or domain-violating)
% membership functions after crossover/mutation, needing repair logic.
% An affine transform (shift + positive scale) applied to every breakpoint
% of a variable's original, valid MF set is guaranteed to remain valid by
% construction (an affine map with positive scale preserves ordering), so
% no repair step is needed, while still giving the GA meaningful control
% over both where each variable's sets are centred and how "sharp" vs.
% "spread out" they are -- exactly the two qualities that matter for
% controller responsiveness (spread) and bias (shift).

    fis = base_fis;
    var_specs = {
        'input',  1, chromosome(1), chromosome(2);   % TempError
        'input',  2, chromosome(3), chromosome(4);   % RateChange
        'output', 1, chromosome(5), chromosome(6);   % HeaterCoolerPower
    };

    for i = 1:size(var_specs, 1)
        io_type = var_specs{i,1};
        io_idx  = var_specs{i,2};
        shift   = var_specs{i,3};
        scale   = var_specs{i,4};

        if strcmp(io_type, 'input')
            domain = base_fis.Inputs(io_idx).Range;
            n_mfs  = numel(base_fis.Inputs(io_idx).MembershipFunctions);
        else
            domain = base_fis.Outputs(io_idx).Range;
            n_mfs  = numel(base_fis.Outputs(io_idx).MembershipFunctions);
        end
        domain_mid = mean(domain);

        for m = 1:n_mfs
            if strcmp(io_type, 'input')
                base_params = base_fis.Inputs(io_idx).MembershipFunctions(m).Parameters;
            else
                base_params = base_fis.Outputs(io_idx).MembershipFunctions(m).Parameters;
            end

            new_params = domain_mid + shift + scale .* (base_params - domain_mid);
            new_params = min(max(new_params, domain(1)), domain(2)); % keep in-domain
            new_params = sort(new_params); % guarantee non-decreasing (safety net)

            % --- Guarantee full coverage of the universe of discourse ---
            % Without this, a chromosome with scale < 1 shrinks every set
            % toward the domain midpoint and leaves the two ends of the
            % universe uncovered: e.g. TempError's 'Cold' trapmf
            % [-10 -10 -6 -2] at scale 0.6 becomes [-6 -6 -3.6 -1.2], so an
            % input of -8 has ZERO membership in every set, no rule fires,
            % and evalfis returns NaN (or the bare range midpoint) instead
            % of a control action. Pinning the outer shoulders of the first
            % and last membership function to the domain edges makes the
            % saturating end sets cover their tails by construction, for
            % every chromosome the GA can generate, so no repair or
            % constraint-handling step is needed in the GA itself.
            if m == 1
                new_params(1:2) = domain(1);
            end
            if m == n_mfs
                new_params(end-1:end) = domain(2);
            end

            if strcmp(io_type, 'input')
                fis.Inputs(io_idx).MembershipFunctions(m).Parameters = new_params;
            else
                fis.Outputs(io_idx).MembershipFunctions(m).Parameters = new_params;
            end
        end
    end
end
