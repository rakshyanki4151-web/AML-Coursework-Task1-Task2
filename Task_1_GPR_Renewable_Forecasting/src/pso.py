"""A hand-written Particle Swarm Optimiser.

Written from scratch (rather than imported from a library) because the
coursework marking criteria reward "correct application" and originality,
and a ~60-line PSO is easy to explain fully in the report's Methods
section with pseudocode.

Standard velocity/position update (Kennedy & Eberhart, 1995):
    v <- w*v + c1*r1*(pbest - x) + c2*r2*(gbest - x)
    x <- x + v
"""
import numpy as np


class PSOResult:
    def __init__(self, best_position, best_score, history):
        self.best_position = best_position
        self.best_score = best_score
        self.history = history  # list of best_score per iteration, for convergence plots


def optimise(fitness_fn, bounds, n_particles=20, n_iterations=30,
             inertia=0.6, cognitive=1.5, social=1.5, seed=42):
    """Minimise ``fitness_fn(position) -> float`` over a box-constrained
    search space.

    Parameters
    ----------
    bounds : list of (low, high) tuples, one per dimension.
    """
    rng = np.random.RandomState(seed)
    dim = len(bounds)
    lows = np.array([b[0] for b in bounds])
    highs = np.array([b[1] for b in bounds])
    span = highs - lows

    positions = lows + rng.rand(n_particles, dim) * span
    velocities = (rng.rand(n_particles, dim) - 0.5) * span * 0.1

    scores = np.array([fitness_fn(p) for p in positions])
    pbest_pos = positions.copy()
    pbest_score = scores.copy()
    gbest_idx = int(np.argmin(pbest_score))
    gbest_pos = pbest_pos[gbest_idx].copy()
    gbest_score = pbest_score[gbest_idx]

    history = [gbest_score]
    for _ in range(n_iterations):
        r1 = rng.rand(n_particles, dim)
        r2 = rng.rand(n_particles, dim)
        velocities = (
            inertia * velocities
            + cognitive * r1 * (pbest_pos - positions)
            + social * r2 * (gbest_pos - positions)
        )
        # Clamp velocity to +-20% of each dimension's span to prevent
        # particles from repeatedly overshooting the box.
        v_clip = 0.2 * span
        velocities = np.clip(velocities, -v_clip, v_clip)

        positions = positions + velocities
        positions = np.clip(positions, lows, highs)

        scores = np.array([fitness_fn(p) for p in positions])
        improved = scores < pbest_score
        pbest_pos[improved] = positions[improved]
        pbest_score[improved] = scores[improved]

        gbest_idx = int(np.argmin(pbest_score))
        if pbest_score[gbest_idx] < gbest_score:
            gbest_score = pbest_score[gbest_idx]
            gbest_pos = pbest_pos[gbest_idx].copy()

        history.append(gbest_score)

    return PSOResult(gbest_pos, gbest_score, history)
