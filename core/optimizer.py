import numpy as np
from cmaes import CMA, SepCMA


class CMAESOptimizer:

    def __init__(
        self,
        objective_function,
        seed=None,
        progress_callback=None,
        max_evals=60000,
        target_similarity=0.95,
        num_restarts=3,
        user_stop_callback=None,
        stagnation_gens=50,
        initial_params=None,
        log_dir=None,
        restart_strategy='bipop'
    ):
        self.objective_function = objective_function
        self.seed = seed
        self.progress_callback = progress_callback
        self.user_stop_callback = user_stop_callback
        self.log_dir = log_dir
        self.restart_strategy = restart_strategy

        # Tracking
        self.best_params = None
        self.best_fitness = float('inf')
        self.total_function_evals = 0
        self.all_restart_solutions = []  # Track all restart solutions

        # Logging history (if log_dir provided)
        self.optimization_history = []  # List of {gen, evals, fitness, similarity, sigma, restart}
        self.metadata = {}  # Optional metadata for logging (target info, etc.)
        self.start_time = None
        self.end_time = None

        # Stop criteria (always whichever comes first)
        self.target_similarity = target_similarity
        self.target_fitness = 1.0 - target_similarity  # Convert similarity to fitness

        # Optimization settings (validated optimal)
        self.N_PARAMS = 37  # All 36 synthesis parameters + note_hold_time (excluding note)
        self.POPULATION_SIZE = 20
        self.MAX_FUNCTION_EVALS = max_evals
        # For 'no_restarts' strategy, force single run (no restarts)
        self.N_RESTARTS = 1 if (restart_strategy == 'no_restarts' or restart_strategy == 'none') else num_restarts
        self.INITIAL_SIGMA = 0.25
        self.POPULATION_MULTIPLIER = 2.0
        self.SIGMA_MULTIPLIER = 0.8
        self.STAGNATION_GENERATIONS = stagnation_gens
        self.TOLERANCE_FUN = 1e-6
        self.PARAM_BOUNDS = (0.0, 1.0)

        # Use provided initial params or default to center
        if initial_params is not None:
            self.INITIAL_MEAN = np.array(initial_params).copy()
            print(f"[OPTIMIZER] Starting from provided initial parameters")
        else:
            self.INITIAL_MEAN = np.full(self.N_PARAMS, 0.5)

    def optimize(self):
        import time
        self.start_time = time.time()

        print("\n" + "="*80)
        print("CMA-ES OPTIMIZATION")
        print("="*80)
        print(f"Parameters: {self.N_PARAMS}")
        print(f"Population: {self.POPULATION_SIZE}")
        print(f"Max evaluations: {self.MAX_FUNCTION_EVALS}")
        print(f"Target similarity: {self.target_similarity:.1%}")
        print(f"Restarts: {self.N_RESTARTS}")
        print(f"Stops at: whichever comes first")
        print("="*80 + "\n")

        current_pop_size = self.POPULATION_SIZE
        current_sigma = self.INITIAL_SIGMA

        for restart_idx in range(self.N_RESTARTS):
            print(f"\n[RESTART {restart_idx + 1}/{self.N_RESTARTS}]")
            print(f"  Population: {current_pop_size}, Sigma: {current_sigma:.4f}")
            print(f"  Evals: {self.total_function_evals}/{self.MAX_FUNCTION_EVALS}\n")

            # Check stop criteria before starting restart
            if self._should_stop_optimization():
                print(f"[STOP CRITERIA MET]")
                break

            restart_best_params, restart_best_fitness = self._run_single_optimization(
                population_size=current_pop_size,
                initial_sigma=current_sigma,
                restart_idx=restart_idx
            )

            # Save this restart's solution
            restart_similarity = 1.0 - restart_best_fitness
            self.all_restart_solutions.append({
                'params': restart_best_params,
                'fitness': restart_best_fitness,
                'similarity': restart_similarity
            })

            if restart_best_fitness < self.best_fitness:
                self.best_fitness = restart_best_fitness
                self.best_params = restart_best_params
                similarity = 1.0 - self.best_fitness
                print(f"\n[NEW BEST] Fitness: {self.best_fitness:.6f} (similarity: {similarity:.2%})")

            # Check if we should stop after this restart
            if self._should_stop_optimization():
                print(f"\n[STOP CRITERIA MET]")
                break

            # Update population and sigma for next restart based on strategy
            current_pop_size, current_sigma = self._get_next_population_and_sigma(
                current_pop_size, current_sigma, restart_idx
            )

        self.end_time = time.time()
        duration = self.end_time - self.start_time

        print("\n" + "="*80)
        print("OPTIMIZATION COMPLETE")
        print("="*80)
        similarity = 1.0 - self.best_fitness
        print(f"Best fitness: {self.best_fitness:.6f} (similarity: {similarity:.2%})")
        print(f"Total evaluations: {self.total_function_evals}")
        print(f"Restart solutions: {len(self.all_restart_solutions)}")
        print(f"Duration: {duration:.1f} seconds")
        print("="*80 + "\n")

        # Save logs if enabled
        if self.log_dir is not None:
            self._save_logs()

        return {
            'best_params': self.best_params,
            'best_fitness': self.best_fitness,
            'total_evals': self.total_function_evals,
            'similarity': similarity,
            'all_solutions': self.all_restart_solutions,
            'history': self.optimization_history
        }

    def _get_next_population_and_sigma(self, current_pop, current_sigma, restart_idx):
        if self.restart_strategy == 'no_restarts' or self.restart_strategy == 'none':
            # No restarts - this shouldn't be called but return unchanged
            return current_pop, current_sigma

        elif self.restart_strategy == 'ipop_1.2':
            # Standard IPOP with modest sigma increase
            new_pop = int(current_pop * 2.0)
            new_sigma = current_sigma * 1.2
            return new_pop, new_sigma

        elif self.restart_strategy == 'ipop_2.0':
            # Aggressive IPOP with sigma doubling
            new_pop = int(current_pop * 2.0)
            new_sigma = current_sigma * 2.0
            return new_pop, new_sigma

        elif self.restart_strategy == 'inverted_ipop':
            # Current implementation: pop increases, sigma decreases
            new_pop = int(current_pop * 2.0)
            new_sigma = current_sigma * 0.8
            return new_pop, new_sigma

        elif self.restart_strategy == 'bipop':
            # BIPOP: Alternates between large and small populations
            # Even restarts (0, 2, 4...): large population with increased sigma
            # Odd restarts (1, 3, 5...): reset to small population with reduced sigma
            if restart_idx % 2 == 0:
                # Large restart
                new_pop = int(current_pop * 2.0)
                new_sigma = current_sigma * 2.0
            else:
                # Small restart - reset to initial values
                new_pop = self.POPULATION_SIZE
                new_sigma = self.INITIAL_SIGMA * 0.5
            return new_pop, new_sigma

        elif self.restart_strategy == 'fixed_pop':
            # Fixed population, only sigma changes
            new_pop = self.POPULATION_SIZE  # Keep constant
            new_sigma = current_sigma * 0.8  # Decrease sigma
            return new_pop, new_sigma

        elif self.restart_strategy == 'amnesia':
            # Reset mean and covariance (happens automatically on restart)
            # but keep sigma constant to test value of covariance memory
            new_pop = int(current_pop * 2.0)  # Standard IPOP population growth
            new_sigma = current_sigma  # Keep sigma constant - THIS IS THE KEY
            return new_pop, new_sigma

        elif self.restart_strategy == 'no_covariance':
            # Uses SepCMA (diagonal-only covariance) - implemented in _run_restart
            # Pop and sigma handled same as inverted_ipop
            new_pop = int(current_pop * 2.0)
            new_sigma = current_sigma * 0.8
            return new_pop, new_sigma

        else:
            raise ValueError(f"Unknown restart_strategy: '{self.restart_strategy}'. "
                           f"Valid options: 'none', 'ipop_1.2', 'ipop_2.0', 'inverted_ipop', 'bipop', 'fixed_pop', 'amnesia', 'no_covariance'")

    def _should_stop_optimization(self):
        evals_met = self.total_function_evals >= self.MAX_FUNCTION_EVALS
        similarity_met = self.best_fitness <= self.target_fitness
        user_stop = self.user_stop_callback and self.user_stop_callback()

        # Stop when ANY condition is met (whichever comes first)
        return evals_met or similarity_met or user_stop

    def _run_single_optimization(self, population_size, initial_sigma, restart_idx):
        bounds = np.array([[self.PARAM_BOUNDS[0], self.PARAM_BOUNDS[1]]] * self.N_PARAMS)

        if restart_idx == 0:
            initial_mean = self.INITIAL_MEAN.copy()
        else:
            np.random.seed(self.seed + restart_idx if self.seed else None)
            initial_mean = self.INITIAL_MEAN + np.random.uniform(-0.1, 0.1, self.N_PARAMS)
            initial_mean = np.clip(initial_mean, self.PARAM_BOUNDS[0], self.PARAM_BOUNDS[1])

        # Use SepCMA (diagonal-only covariance) for no_covariance strategy
        # Otherwise use standard CMA-ES
        if self.restart_strategy == 'no_covariance':
            optimizer = SepCMA(
                mean=initial_mean,
                sigma=initial_sigma,
                bounds=bounds,
                seed=(self.seed + restart_idx) if self.seed is not None else None,
                population_size=population_size
            )
        else:
            optimizer = CMA(
                mean=initial_mean,
                sigma=initial_sigma,
                bounds=bounds,
                seed=(self.seed + restart_idx) if self.seed is not None else None,
                population_size=population_size
            )

        restart_best_params = None
        restart_best_fitness = float('inf')
        stagnation_counter = 0
        last_best_fitness = float('inf')

        generation = 0
        max_generations = 10000  # Safety limit

        while generation < max_generations:
            # Check stop criteria (budget or target similarity or user stop)
            if self._should_stop_optimization():
                if self.user_stop_callback and self.user_stop_callback():
                    print(f"  [USER REQUESTED STOP]")
                elif self.total_function_evals >= self.MAX_FUNCTION_EVALS:
                    print(f"  [Budget limit reached]")
                else:
                    print(f"  [Target similarity reached]")
                break

            # Ask for candidates
            candidates = [optimizer.ask() for _ in range(optimizer.population_size)]

            # Evaluate batch
            fitnesses = []
            for x in candidates:
                fitness = self.objective_function(x)
                fitnesses.append(fitness)
                self.total_function_evals += 1

            solutions = list(zip(candidates, fitnesses))

            # Tell optimizer
            optimizer.tell(solutions)

            # Track best
            gen_best_params, gen_best_fitness = min(solutions, key=lambda s: s[1])

            if gen_best_fitness < restart_best_fitness:
                restart_best_fitness = gen_best_fitness
                restart_best_params = gen_best_params

            # Update global best for stop criteria checking
            if restart_best_fitness < self.best_fitness:
                self.best_fitness = restart_best_fitness
                self.best_params = restart_best_params

            # Progress callback
            if self.progress_callback:
                self.progress_callback(
                    self.total_function_evals,
                    self.MAX_FUNCTION_EVALS,
                    restart_best_fitness
                )

            # Stagnation check
            if abs(restart_best_fitness - last_best_fitness) < self.TOLERANCE_FUN:
                stagnation_counter += 1
            else:
                stagnation_counter = 0

            last_best_fitness = restart_best_fitness

            # Logging
            if generation % 10 == 0:
                similarity = 1.0 - restart_best_fitness
                print(f"  Gen {generation:3d}: "
                      f"Fitness={restart_best_fitness:.6f} (sim={similarity:.2%}), "
                      f"Sigma={optimizer._sigma:.4f}, "
                      f"Evals={self.total_function_evals}")

                # Save to history if logging enabled
                if self.log_dir is not None:
                    self.optimization_history.append({
                        'generation': generation,
                        'restart_idx': restart_idx,
                        'total_evals': self.total_function_evals,
                        'fitness': float(restart_best_fitness),
                        'similarity': float(similarity),
                        'sigma': float(optimizer._sigma),
                        'population_size': optimizer.population_size
                    })

            # Early stopping
            if restart_best_fitness < 0.01:
                print(f"  [PERFECT MATCH]")
                break

            if stagnation_counter >= self.STAGNATION_GENERATIONS:
                print(f"  [Converged - no improvement for {self.STAGNATION_GENERATIONS} gens]")
                break

            if optimizer.should_stop():
                print(f"  [CMA-ES stopping criteria met]")
                break

            generation += 1

        print(f"  Restart complete: Fitness = {restart_best_fitness:.6f}")
        return restart_best_params, restart_best_fitness

    def _save_logs(self):
        import json
        import os
        from datetime import datetime

        # Create log directory if needed
        os.makedirs(self.log_dir, exist_ok=True)

        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_filename = f"optimization_log_{timestamp}.json"
        log_path = os.path.join(self.log_dir, log_filename)

        # Calculate duration
        duration = self.end_time - self.start_time if (self.start_time and self.end_time) else None

        # Prepare log data
        log_data = {
            'timestamp': timestamp,
            'timing': {
                'start_time': self.start_time,
                'end_time': self.end_time,
                'duration_seconds': duration
            },
            'metadata': self.metadata,  # Target info, etc.
            'hyperparameters': {
                'n_params': self.N_PARAMS,
                'population_size': self.POPULATION_SIZE,
                'initial_sigma': self.INITIAL_SIGMA,
                'population_multiplier': self.POPULATION_MULTIPLIER,
                'sigma_multiplier': self.SIGMA_MULTIPLIER,
                'stagnation_generations': self.STAGNATION_GENERATIONS,
                'n_restarts': self.N_RESTARTS,
                'max_evals': self.MAX_FUNCTION_EVALS,
                'target_similarity': self.target_similarity,
                'restart_strategy': self.restart_strategy,
            },
            'results': {
                'best_fitness': float(self.best_fitness),
                'best_similarity': float(1.0 - self.best_fitness),
                'total_evals': int(self.total_function_evals),
                'num_restarts_completed': len(self.all_restart_solutions),
            },
            'history': self.optimization_history,
            'restart_solutions': [
                {
                    'params': sol['params'].tolist() if hasattr(sol['params'], 'tolist') else list(sol['params']),
                    'fitness': float(sol['fitness']),
                    'similarity': float(sol['similarity'])
                }
                for sol in self.all_restart_solutions
            ]
        }

        # Save to JSON
        with open(log_path, 'w') as f:
            json.dump(log_data, f, indent=2)

        print(f"[LOG] Saved optimization log to: {log_path}")
