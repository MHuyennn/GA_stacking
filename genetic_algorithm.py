from __init__ import *
from chromosome import Chromosome

class GeneticAlgorithm:
    def __init__(self, 
                 params_tuple,
                 population_size=50, 
                 generations=100,
                 crossover_rate=0.8,
                 mutation_rate=0.05, 
                 tournament_size=10,
                 elitism=True,
                 model_performances=None,
                 initial_crossover_rate=0.8, 
                 initial_mutation_rate=0.05, 
                 min_crossover_rate=0.5,
                 min_mutation_rate=0.01):
        """
        Initializes the Genetic Algorithm with dynamic rate adjustment.
        """
        self.params_tuple = params_tuple
        self.population_size = population_size
        self.generations = generations
        self.initial_crossover_rate = initial_crossover_rate
        self.initial_mutation_rate = initial_mutation_rate
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.min_crossover_rate = min_crossover_rate
        self.min_mutation_rate = min_mutation_rate
        self.tournament_size = tournament_size
        self.elitism = elitism
        self.model_performances = model_performances

        self.population = []
        self.best_chromosome = Chromosome()
        self.previous_best_fitness = -np.inf
        self.initialize_population()

    def initialize_population(self):
        print("Initializing population...")
        if self.model_performances is not None:
            performance = np.array(self.model_performances)
            performance = performance / performance.sum()
            for _ in range(self.population_size):
                genes = [np.random.choice([0, 1], p=[1 - p, p]) for p in performance]
                chromosome = Chromosome(genes)
                chromosome.calculate_fitness_score(self.params_tuple)
                self.population.append(chromosome)
        else:
            for _ in range(self.population_size):
                chromosome = Chromosome()
                chromosome.calculate_fitness_score(self.params_tuple)
                self.population.append(chromosome)
        self.update_best_chromosome()

    def update_best_chromosome(self):
        current_best = max(self.population, key=lambda chromo: chromo.fitness_score)
        if (self.best_chromosome is None) or (current_best.fitness_score > self.best_chromosome.fitness_score):
            self.best_chromosome = deepcopy(current_best)

    def tournament_selection(self):
        tournament = random.sample(self.population, self.tournament_size)
        winner = max(tournament, key=lambda chromo: chromo.fitness_score)
        return winner

    def crossover(self, parent1, parent2):
        if random.random() < self.crossover_rate:
            point = random.randint(1, len(MODELS) - 1)
            child1_genes = parent1.genes[:point] + parent2.genes[point:]
            child2_genes = parent2.genes[:point] + parent1.genes[point:]
            return Chromosome(child1_genes), Chromosome(child2_genes)
        return deepcopy(parent1), deepcopy(parent2)

    def mutate(self, chromosome):
        for i in range(len(chromosome.genes)):
            if random.random() < self.mutation_rate:
                chromosome.genes[i] = 1 - chromosome.genes[i]
        return chromosome

    def evolve_population(self):
        new_population = []
        if self.elitism:
            new_population.append(deepcopy(self.best_chromosome))

        while len(new_population) < self.population_size:
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()
            child1, child2 = self.crossover(parent1, parent2)
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)
            child1.calculate_fitness_score(self.params_tuple)
            child2.calculate_fitness_score(self.params_tuple)
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        self.population = new_population
        self.update_best_chromosome()
        if self.best_chromosome.fitness_score > self.previous_best_fitness:
            self.crossover_rate = max(self.crossover_rate * 0.99, self.min_crossover_rate)
            self.mutation_rate = max(self.mutation_rate * 0.99, self.min_mutation_rate)
        else:
            self.crossover_rate = min(self.crossover_rate * 1.01, self.initial_crossover_rate)
            self.mutation_rate = min(self.mutation_rate * 1.01, self.initial_mutation_rate)
        self.previous_best_fitness = self.best_chromosome.fitness_score

    def get_best_chromosome(self):
        return self.best_chromosome
    
    def run(self):
        print("Starting Genetic Algorithm evolution...")
        for generation in range(1, self.generations + 1):
            print(f"\n--- Generation {generation} ---")
            self.evolve_population()
            print(f"Generation {generation}: Best Fitness = {self.best_chromosome.fitness_score}")
        print("\nGenetic Algorithm completed.")
        print(f"Best Chromosome: {self.best_chromosome}")
        print(f"Best Chromosome fitness score: {self.best_chromosome.fitness_score}")
