from __init__ import *

class Chromosome:
    def __init__(self, genes=None):
        if genes is not None:
            assert len(genes) == len(MODELS), "Genes length must be equal to MODELS length."
            self.genes = genes
        else:
            self.genes = np.random.randint(0, 2, size=len(MODELS)).tolist()
        self.meta_learner = LogisticRegression(multi_class='multinomial', max_iter=1000)
        self.fitness_score = 0

    def __repr__(self):
        return f"Chromosome(genes={self.genes}, fitness={self.fitness_score})"

    def calculate_fitness_score(self, params_tuple, img_size=299):
        chosen_indexes = [i for i, value in enumerate(self.genes) if value == 1]
        if not chosen_indexes:
            return 0
        
        data, CACHE_PREDICTIONS, x_train_meta = params_tuple
        x_chromosome_meta = x_train_meta[:, chosen_indexes]
        y_train_meta = data.y_train
        
        clf = self.meta_learner.fit(x_chromosome_meta, y_train_meta)
        x_test_meta = CACHE_PREDICTIONS[:, chosen_indexes]
        y_test_meta = clf.predict(x_test_meta)
        self.fitness_score = f1_score(y_test_meta, data.y_test, average='macro')
        return self.fitness_score
