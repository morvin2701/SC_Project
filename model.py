# ==============================================================
# model.py
# Hybrid PSO-GA Feature Selection Algorithm (clean version)
# ==============================================================

import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings("ignore")


class HybridPSOGA:
    """
    Hybrid Particle Swarm Optimization + Genetic Algorithm for Feature Selection
    """

    def __init__(self, n_particles=30, n_iterations=50, w=0.7, c1=1.5, c2=1.5,
                 crossover_rate=0.8, mutation_rate=0.1, ga_interval=5,
                 classifier_type='rf', cv_folds=3):
        self.n_particles = n_particles
        self.n_iterations = n_iterations
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.ga_interval = ga_interval
        self.classifier_type = classifier_type
        self.cv_folds = cv_folds

    # ------------------------------------------------------------------
    def evaluate_fitness(self, X, y, particle):
        """Evaluate subset fitness using stratified CV accuracy."""
        selected_features = np.where(particle == 1)[0]
        if len(selected_features) == 0:
            return -1.0

        X_sel = X[:, selected_features]
        clf = RandomForestClassifier(n_estimators=100, random_state=42)

        # Dynamic fold count (fixes small-class error)
        unique, counts = np.unique(y, return_counts=True)
        min_class_size = np.min(counts)
        cv_folds = int(min(3, min_class_size))
        if cv_folds < 2:
            cv_folds = 2

        cv = StratifiedKFold(n_splits=int(cv_folds), shuffle=True, random_state=42)
        try:
            scores = cross_val_score(clf, X_sel, y, cv=cv, scoring="accuracy")
            fitness = scores.mean() - 0.01 * (len(selected_features) / X.shape[1])
        except Exception:
            fitness = -1.0
        return fitness

    # ------------------------------------------------------------------
    def optimize(self, X, y):
        """Main optimization loop."""
        n_features = X.shape[1]
        population = np.random.randint(0, 2, (self.n_particles, n_features))
        velocities = np.random.uniform(-1, 1, (self.n_particles, n_features))

        personal_best = population.copy()
        personal_best_scores = np.array([self.evaluate_fitness(X, y, p) for p in population])
        global_best = personal_best[np.argmax(personal_best_scores)].copy()
        global_best_score = np.max(personal_best_scores)
        fitness_history = [global_best_score]

        for iteration in range(self.n_iterations):
            for i in range(self.n_particles):
                r1, r2 = np.random.rand(n_features), np.random.rand(n_features)
                velocities[i] = (
                    self.w * velocities[i]
                    + self.c1 * r1 * (personal_best[i] - population[i])
                    + self.c2 * r2 * (global_best - population[i])
                )
                sigmoid = 1 / (1 + np.exp(-velocities[i]))
                population[i] = (sigmoid > np.random.rand(n_features)).astype(int)
                if population[i].sum() == 0:
                    population[i, np.random.randint(n_features)] = 1

            # GA operators every few iterations
            if (iteration + 1) % self.ga_interval == 0:
                population = self.apply_ga(population)

            scores = np.array([self.evaluate_fitness(X, y, p) for p in population])
            better_mask = scores > personal_best_scores
            personal_best_scores[better_mask] = scores[better_mask]
            personal_best[better_mask] = population[better_mask]
            best_idx = np.argmax(personal_best_scores)
            if personal_best_scores[best_idx] > global_best_score:
                global_best_score = personal_best_scores[best_idx]
                global_best = personal_best[best_idx].copy()

            fitness_history.append(global_best_score)

        print(f"Optimization complete. Best score: {global_best_score:.4f}")
        return global_best

    # ------------------------------------------------------------------
    def apply_ga(self, population):
        """Simple GA crossover + mutation."""
        new_pop = population.copy()
        for _ in range(len(population) // 2):
            i1, i2 = np.random.choice(len(population), 2, replace=False)
            p1, p2 = population[i1], population[i2]
            if np.random.rand() < self.crossover_rate:
                mask = np.random.rand(len(p1)) > 0.5
                c1, c2 = p1.copy(), p2.copy()
                c1[mask], c2[mask] = p2[mask], p1[mask]
                new_pop[i1], new_pop[i2] = c1, c2

        for i in range(len(new_pop)):
            for j in range(new_pop.shape[1]):
                if np.random.rand() < self.mutation_rate:
                    new_pop[i, j] = 1 - new_pop[i, j]
            if new_pop[i].sum() == 0:
                new_pop[i, np.random.randint(new_pop.shape[1])] = 1
        return new_pop



