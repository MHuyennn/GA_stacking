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

        Parameters:
            params_tuple (tuple): A tuple of parameters required for fitness evaluation.
            population_size (int, optional): The number of chromosomes in the population. Default is 50.
            generations (int, optional): The number of generations to evolve. Default is 100.
            crossover_rate (float, optional): The probability of performing crossover between two parents. Default is 0.8.
            mutation_rate (float, optional): The probability of mutating each gene in a chromosome. Default is 0.05.
            tournament_size (int, optional): The number of chromosomes competing in each tournament selection. Default is 10.
            elitism (bool, optional): Whether to retain the best chromosome from each generation. Default is True.
            model_performances (list, optional): Performance scores of models used for weighted initialization. Default is None.
            initial_crossover_rate (float, optional): The starting crossover rate for dynamic adjustment. Default is 0.8.
            initial_mutation_rate (float, optional): The starting mutation rate for dynamic adjustment. Default is 0.05.
            min_crossover_rate (float, optional): The minimum allowable crossover rate during adjustment. Default is 0.5.
            min_mutation_rate (float, optional): The minimum allowable mutation rate during adjustment. Default is 0.01.
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
        self.model_performances = model_performances  # Performance scores for weighted initialization

        self.population = []
        self.best_chromosome = Chromosome()
        self.previous_best_fitness = -np.inf

        # Initialize the population at the start
        self.initialize_population()

    def initialize_population(self):
        """
        Initializes the population of chromosomes.

        If `model_performances` is provided, initializes the population using weighted probabilities
        based on the performance scores. Each gene in a chromosome is set to 1 with a probability
        proportional to its corresponding model's performance. Otherwise, initializes chromosomes
        with genes randomly set to 0 or 1 with equal probability.

        This method also evaluates the fitness of each chromosome and updates the best chromosome found.
        """
        print("Initializing population...")
        if self.model_performances is not None:
            # Normalize performance scores to sum to 1 for probability distribution
            performance = np.array(self.model_performances)
            performance = performance / performance.sum()
            for _ in range(self.population_size):
                genes = []
                for i in range(len(MODELS)):
                    # Set gene to 1 with probability equal to the model's normalized performance
                    gene = np.random.choice([0, 1], p=[1 - performance[i], performance[i]])
                    genes.append(gene)
                chromosome = Chromosome(genes)
                chromosome.calculate_fitness_score(self.params_tuple)
                self.population.append(chromosome)
        else:
            # Uniform random initialization: each gene is 0 or 1 with equal probability
            for _ in range(self.population_size):
                chromosome = Chromosome()
                chromosome.calculate_fitness_score(self.params_tuple)
                self.population.append(chromosome)
        
        # Identify and store the best chromosome in the initial population
        self.update_best_chromosome()

    def update_best_chromosome(self):
        """
        Identifies and updates the best chromosome in the current population.

        Compares the fitness scores of all chromosomes in the population and updates
        `self.best_chromosome` if a chromosome with a higher fitness score is found.
        """
        current_best = max(self.population, key=lambda chromo: chromo.fitness_score)
        if (self.best_chromosome is None) or (current_best.fitness_score > self.best_chromosome.fitness_score):
            self.best_chromosome = deepcopy(current_best)

    def tournament_selection(self):
        """
        Selects a parent chromosome using tournament selection.

        Randomly selects a subset of chromosomes (`tournament_size`) from the population
        and returns the chromosome with the highest fitness score among them.

        Returns:
            Chromosome: The selected parent chromosome with the highest fitness in the tournament.
        """
        tournament = random.sample(self.population, self.tournament_size)
        winner = max(tournament, key=lambda chromo: chromo.fitness_score)
        return winner

    def crossover(self, parent1, parent2):
        """
        Performs single-point crossover between two parent chromosomes to produce offspring.

        With a probability defined by `crossover_rate`, selects a random crossover point and
        exchanges the gene segments between the two parents to create two new offspring.
        If crossover is not performed, returns deep copies of the original parents.

        Parameters:
            parent1 (Chromosome): The first parent chromosome.
            parent2 (Chromosome): The second parent chromosome.

        Returns:
            tuple: Two offspring chromosomes resulting from the crossover operation.
        """
        if random.random() < self.crossover_rate:
            # Choose a random crossover point excluding the first and last positions
            point = random.randint(1, len(MODELS) - 1)
            # Create offspring by swapping gene segments at the crossover point
            child1_genes = parent1.genes[:point] + parent2.genes[point:]
            child2_genes = parent2.genes[:point] + parent1.genes[point:]
            return Chromosome(child1_genes), Chromosome(child2_genes)
        else:
            # No crossover; offspring are exact copies of parents
            return deepcopy(parent1), deepcopy(parent2)

    def mutate(self, chromosome):
        """
        Applies mutation to a chromosome by flipping its genes with a certain probability.

        Iterates through each gene in the chromosome and, with a probability defined by
        `mutation_rate`, flips the gene from 0 to 1 or from 1 to 0.

        Parameters:
            chromosome (Chromosome): The chromosome to mutate.

        Returns:
            Chromosome: The mutated chromosome.
        """
        for i in range(len(chromosome.genes)):
            if random.random() < self.mutation_rate:
                # Flip the gene: 0 becomes 1, and 1 becomes 0
                chromosome.genes[i] = 1 - chromosome.genes[i]
        return chromosome

    def evolve_population(self):
        """
        Evolves the population by creating a new generation through selection, crossover, and mutation.

        The method performs the following steps:
            1. **Elitism**: If enabled, retains the best chromosome from the current population.
            2. **Selection**: Uses tournament selection to choose parent chromosomes.
            3. **Crossover**: Applies single-point crossover to produce offspring.
            4. **Mutation**: Mutates the offspring chromosomes.
            5. **Fitness Evaluation**: Calculates the fitness scores of the new offspring.
            6. **Population Update**: Replaces the old population with the new generation.
            7. **Rate Adjustment**: Dynamically adjusts the crossover and mutation rates based on improvement.

        The dynamic adjustment works as follows:
            - If the best chromosome's fitness has improved, slightly decrease the crossover and mutation rates.
            - If there is no improvement, slightly increase the crossover and mutation rates.
            - Rates are bounded by their specified minimum and initial values.
        """
        new_population = []

        # Step 1: Elitism - retain the best chromosome
        if self.elitism:
            new_population.append(deepcopy(self.best_chromosome))

        # Step 2-5: Generate the rest of the population
        while len(new_population) < self.population_size:
            # Selection: choose two parents via tournament selection
            parent1 = self.tournament_selection()
            parent2 = self.tournament_selection()

            # Crossover: produce two offspring
            child1, child2 = self.crossover(parent1, parent2)

            # Mutation: apply mutation to the offspring
            child1 = self.mutate(child1)
            child2 = self.mutate(child2)

            # Fitness Evaluation: calculate fitness scores for offspring
            child1.calculate_fitness_score(self.params_tuple)
            child2.calculate_fitness_score(self.params_tuple)

            # Add offspring to the new population
            new_population.append(child1)
            if len(new_population) < self.population_size:
                new_population.append(child2)

        # Step 6: Update the population with the new generation
        self.population = new_population
        self.update_best_chromosome()

        # Step 7: Dynamic Rate Adjustment based on improvement
        if self.best_chromosome.fitness_score > self.previous_best_fitness:
            # Improvement detected: decrease rates to fine-tune search
            self.crossover_rate = max(self.crossover_rate * 0.99, self.min_crossover_rate)
            self.mutation_rate = max(self.mutation_rate * 0.99, self.min_mutation_rate)
        else:
            # No improvement: increase rates to encourage exploration
            self.crossover_rate = min(self.crossover_rate * 1.01, self.initial_crossover_rate)
            self.mutation_rate = min(self.mutation_rate * 1.01, self.initial_mutation_rate)
        
        # Update the record of the best fitness score
        self.previous_best_fitness = self.best_chromosome.fitness_score

    def get_best_chromosome(self):
        """
        Retrieves the best chromosome found by the Genetic Algorithm.

        Returns:
            Chromosome: The chromosome with the highest fitness score in the population.
        """
        return self.best_chromosome
    
    def run(self):
        """
        Executes the Genetic Algorithm over the specified number of generations.

        The method performs the following actions:
            1. Prints a starting message.
            2. Iterates through each generation, evolving the population.
            3. After each generation, prints the generation number and the best fitness score.
            4. Upon completion, prints the final best chromosome and its fitness score.

        This method encapsulates the entire evolutionary process from initialization to final solution.
        """
        print("Starting Genetic Algorithm evolution...")
        for generation in range(1, self.generations + 1):
            print(f"\n--- Generation {generation} ---")
            self.evolve_population()
            print(f"Generation {generation}: Best Fitness = {self.best_chromosome.fitness_score}")
        print("\nGenetic Algorithm completed.")
        print(f"Best Chromosome: {self.best_chromosome}")
        print(f"Best Chromosome fitness score: {self.best_chromosome.fitness_score}")
