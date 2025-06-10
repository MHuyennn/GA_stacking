from __init__ import *

class Chromosome:
    def __init__(self, genes=None):
        """
        Initializes a Chromosome with a binary list of genes.
        If no genes are provided, initializes randomly.
        """
        if genes is not None:
            assert len(genes) == len(MODELS), "Genes length must be equal to MODELS length."
            self.genes = genes
        else:
            # Random initialization
            self.genes = np.random.randint(0, 2, size=len(MODELS)).tolist()
        
        self.meta_learner = LogisticRegression(max_iter=1000)
        self.fitness_score = 0

    def __repr__(self):
        return f"Chromosome(genes={self.genes}, fitness={self.fitness_score})"

    def calculate_fitness_score(self, params_tuple, img_size=128):
        """
        Calculates the fitness score (F1 score) of the chromosome.
        """

        chosen_indexes = [i for i, value in enumerate(self.genes) if (value == 1) and (i not in (7, 8, 9, 10))]
        
        if all(x==0 for x in chosen_indexes):
            return 0
        
        data, CACHE_PREDICTIONS, x_train_meta = params_tuple

        # Subset from x_train_meta to train the meta learner of this chromosome
        x_chromosome_meta = x_train_meta[:, chosen_indexes]
        y_train_meta = data.y_train
        
        # Train the chromosome's meta learner
        clf = self.meta_learner.fit(x_chromosome_meta, y_train_meta)
        
        # Select columns from chosen_indexes in CACHE_PREDICTIONS to get x_test_meta and use the above model to predict on it and get y_test_meta
        x_test_meta = CACHE_PREDICTIONS[:, chosen_indexes]
        y_test_meta = clf.predict(x_test_meta)

        # The f1_score of (y_test_meta, data.y_test) will be the fitness score of this chromosome
        self.fitness_score = f1_score(y_test_meta, data.y_test)

        return self.fitness_score



