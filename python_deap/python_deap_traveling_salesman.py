########################################## 1
import csv
import random
from pathlib import Path
import numpy as np
from deap import base, creator, tools
import matplotlib.pyplot as plt
########################################## Globals
INDIVIDUAL_SIZE = NUMBER_OF_CITIES = 81
POPULATION_SIZE = 200
N_ITERATIONS = 1000
N_MATINGS = 50
########################################## 2
# noinspection PyShadowingNames,PyAttributeOutsideInit
class Runner:

    def __init__(self, toolbox):
        self.toolbox = toolbox
        self.set_parameters(10, 5, 2)

    def set_parameters(self, population_size, iterations, n_matings):
        self.iterations = iterations
        self.population_size = population_size
        self.n_matings = n_matings

    def set_fitness(self, population):
        fitnesses = [
            (individual, self.toolbox.evaluate(individual))
            for individual in population
        ]

        for individual, fitness in fitnesses:
            individual.fitness.values = (fitness,)

    def get_offspring(self, population):
        n = len(population)
        for _ in range(self.n_matings):
            i1, i2 = np.random.choice(range(n), size=2, replace=False)

            offspring1, offspring2 = \
                self.toolbox.mate(population[i1], population[i2])

            yield self.toolbox.mutate(offspring1)[0]
            yield self.toolbox.mutate(offspring2)[0]

    @staticmethod
    def pull_stats(population, iteration=1):
        fitnesses = [individual.fitness.values[0] for individual in population]
        return {
            'i': iteration,
            'mu': np.mean(fitnesses),
            'std': np.std(fitnesses),
            'max': np.max(fitnesses),
            'min': np.min(fitnesses)
        }

    @property
    def Run(self):
        population = self.toolbox.population(n=self.population_size)
        self.set_fitness(population)

        stats = []
        for iteration in list(range(1, self.iterations + 1)):
            current_population = list(map(self.toolbox.clone, population))
            offspring = list(self.get_offspring(current_population))
            for child in offspring:
                current_population.append(child)

            ## reset fitness,
            self.set_fitness(current_population)

            population[:] = self.toolbox.select(current_population, len(population))
            stats.append(Runner.pull_stats(population, iteration))
            print('iteration: ' + str(len(stats)))
            # if len(stats) == 100:
            #     break
            # else:
            #     print('iteration: ' + str(len(stats)))
        return stats, population
########################################## 3
creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
# noinspection PyUnresolvedReferences
creator.create("Individual", list, fitness=creator.FitnessMin)
########################################## 4
random.seed(11)
np.random.seed(121)

## City names,
csvDir = str(Path(__file__).parent) + '\\ilmesafe.csv'
with open(csvDir,encoding='utf8') as csvFile:
    data = list(csv.reader(csvFile))
npa = np.asarray(data)  # To Matrix
npa = np.delete(npa,0,1)    # First column
npa = np.delete(npa,0,1)    # Second column
cities = npa[:1].tolist()
cities = cities[0]
npa = np.delete(npa,0,0)    # First row
for i in range(len(npa)):   # Arrange empty values in distance Matrix
    for j in range(len(npa[i])):
        if npa[i,j] == '':
            npa[i, j] = '0'
distances = npa.astype('float64')
########################################## 5
toolbox = base.Toolbox()

## permutation setup for individual,
toolbox.register("indices", random.sample, range(INDIVIDUAL_SIZE), INDIVIDUAL_SIZE)
# noinspection PyUnresolvedReferences
toolbox.register("individual", tools.initIterate, creator.Individual, toolbox.indices)

## population setup,
# noinspection PyUnresolvedReferences
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
########################################## 6
# noinspection PyShadowingNames
def EVALUATE(individual):
    summation = 0
    start = individual[0]
    for i in range(1, len(individual)):
        end = individual[i]
        summation += distances[start][end]
        start = end
    return summation

toolbox.register("evaluate", EVALUATE)
########################################## 7
toolbox.register("mate", tools.cxOrdered)
toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.01)
toolbox.register("select", tools.selTournament, tournsize=10)
########################################## 8
a = Runner(toolbox)
a.set_parameters(POPULATION_SIZE, N_ITERATIONS, N_MATINGS)

stats, population = a.Run
########################################## Dolaşma
print('En kısa yolun uzunluğu')
print(stats[N_ITERATIONS-1])
print('Denizli’den başlayacak şekilde dolaşılacak illerin sıralaması')
arrBefore19 = []
arrAfter19 = []
arrSwitch = False
for plate in population[POPULATION_SIZE-1]:
    if plate == 19:
        arrSwitch = True
    if arrSwitch:
        arrAfter19.append(plate)
    else:
        arrBefore19.append(plate)
concatenatedPlates = arrAfter19 + arrBefore19
for i in range(len(concatenatedPlates)):
    concatenatedPlates[i] = concatenatedPlates[i]+1
cities_to_list = []
for i in range(len(concatenatedPlates)):
    cp_index = concatenatedPlates[i]
    ctl_index = cities[cp_index-1] + '-' + str(cp_index)
    cities_to_list.append(ctl_index)
for i in range(len(cities_to_list)):
    print(str(i) + ': ' + cities_to_list[i])
# print('daa')
########################################## Plot
plt.figure(figsize=(15,5))

plt.subplot(1,2,1)

_ = plt.scatter([ s['min'] for s in stats ], [ s['max'] for s in stats ], marker='.', s=[ (s['std'] + 1) / 20 for s in stats ])

_ = plt.title('min by max')
_ = plt.xlabel('min')
_ = plt.ylabel('max')

_ = plt.plot(stats[0]['min'], stats[0]['max'], marker='.', color='yellow')
_ = plt.plot(stats[-1]['min'], stats[-1]['max'], marker='.', color='red')


plt.subplot(1,2,2)

_ = plt.scatter([ s['i'] for s in stats ], [ s['mu'] for s in stats ], marker='.', s=[ (s['std'] + 1) / 20 for s in stats ])

_ = plt.title('average by iteration')
_ = plt.xlabel('iteration')
_ = plt.ylabel('average')

_ = plt.plot(stats[0]['i'], stats[0]['mu'], marker='.', color='yellow')
_ = plt.plot(stats[-1]['i'], stats[-1]['mu'], marker='.', color='red')

plt.tight_layout()
plt.show()
