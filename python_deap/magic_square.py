import random
import math
from deap import base
from deap import creator
from deap import tools
import matplotlib.pyplot as plt
import numpy as np

def evalMagicSquare(individual):
    size = len(individual)
    n = int(math.sqrt(size))
    magic_n = (n*(n*n +1))/2

    fitness = 0
    diag_aux = 0
    diag2_aux = 0
    for i in range(n):
        row_aux = 0
        column_aux = 0
        for j in range(n):
            row_aux = row_aux + individual[i*n + j]
            column_aux = column_aux + individual[i + j*n]

            if i==j:
                diag_aux = diag_aux + individual[i*n +j]
            elif i == (n-1) -j:
                diag2_aux = diag2_aux + individual[i*n + j]

        fitness = fitness + abs(column_aux - magic_n) + abs(row_aux - magic_n)

    fitness = fitness + abs(diag_aux - magic_n) + abs(diag2_aux - magic_n)
    return fitness,
# noinspection PyTrailingSemicolon,PyRedundantParentheses
def cxCycle(ind1, ind2):
    length = len(ind1)
    index = random.randint(0, length-1)
    cycle = [0]*length

    while(not cycle[index]):
        cycle[index] = 1;
        for i in range(0, length):
            if ind1[index] == ind2[i]:
                index = i
                break

    for j in range(0,length):
      if (cycle[j] == 1):
        temp = ind1[j]
        ind1[j] = ind2[j];
        ind2[j] = temp;

    return ind1,ind2
# noinspection PyTrailingSemicolon,PyShadowingNames
def main():
    BOARD_SIZE = 10

    creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))
    creator.create("Individual", list, fitness=creator.FitnessMin, invfitness=creator.FitnessMax)

    toolbox = base.Toolbox()
    toolbox.register("indices", random.sample, range(1, BOARD_SIZE*BOARD_SIZE+1), BOARD_SIZE*BOARD_SIZE)
    toolbox.register("individual", tools.initIterate, creator.Individual,
                     toolbox.indices)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    toolbox.register("evaluate", evalMagicSquare)
    toolbox.register("mate", cxCycle)
    toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.10)
    toolbox.register("select", tools.selRoulette, fit_attr='invfitness')
    # toolbox.register("select", tools.selTournament, tournsize=3)
    #toolbox.register("select", tools.selBest)

    pop_size = 100;
    cros_over = 0.8;
    n_sons = int(pop_size*cros_over);
    n_ger = 300;
    # n_ger = 50;
    mutation_rate = 0.7;

    pop = toolbox.population(n=pop_size)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, pop))
    for ind, fit in zip(pop, fitnesses):
        ind.fitness.values = fit
        ind.invfitness.values = (100-fit[0],)

    # Variable keeping track of the number of generations
    g = 0
    arrGraphic = [[], [], []]
    while g < n_ger:
        # A new generation
        g = g + 1
        print("-- Generation %i --" % g)

        #Select parents
        parents = toolbox.select(pop, n_sons)
        # Clone the selected individuals
        offspring = list(map(toolbox.clone, parents))
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            toolbox.mate(child1, child2)

        for mutant in offspring:
            if random.random() < mutation_rate:
                toolbox.mutate(mutant)

        fitnesses = list(map(toolbox.evaluate, offspring))
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit
            ind.invfitness.values = (100-fit[0],)

        sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0])
        pop[:(pop_size-n_sons)] = sorted_pop[:(pop_size-n_sons)]
        pop[(pop_size-n_sons):] = offspring

        sorted_pop = sorted(pop, key=lambda ind: ind.fitness.values[0])
        av_min = sorted_pop[0].fitness.values[0]
        arrGraphic[0].append(av_min)
        av_max = sorted_pop[-1].fitness.values[0]
        arrGraphic[1].append(av_max)
        fits = [ind.fitness.values[0] for ind in pop]

        mean = sum(fits) / pop_size
        arrGraphic[2].append(mean)
        sum2 = sum(x*x for x in fits)
        std = abs(sum2 / pop_size - mean**2)**0.5

        print("Best ", sorted_pop[0])
        print("Fitnesses:")
        print("\tMin %0.05s" % av_min)
        print("\tMax %0.05s" % av_max)
        print("\tAvg %0.05s" % mean)
        print("\tStd %0.05s" % std)

        if av_min==0:
            break
    # Draw
    print("--Draw Best--" + str(BOARD_SIZE) + "x" + str(BOARD_SIZE))
    best = []
    insertList = []
    # noinspection PyUnboundLocalVariable
    for i in range(len(list(sorted_pop[0]))):
        insertList.append(list(sorted_pop[0])[i])
        if ((i+1) % BOARD_SIZE == 0) & (i != 0):
            best.append(insertList)
            insertList = []
    for row in best:
        for i in range(len(row)):
            if i == 0:
                print(end='[')
            if ((i + 1) % BOARD_SIZE == 0) & (i != 0):
                print(row[i], end=']\n')
            else:
                print(row[i], end=", ")

    #Graphic # arrGraphic
    # Some example data to display
    arrX = [[], [], []]
    arrY = [[], [], []]
    fig, axs = plt.subplots(3)
    # fig.suptitle('İbrahim Mert Küni - 16253801')
    for i in range(len(arrGraphic[0])):
        arrX[0].append(i)
        arrX[1].append(i)
        arrX[2].append(i)
        arrY[0].append(arrGraphic[0][i])
        arrY[1].append(arrGraphic[1][i])
        arrY[2].append(arrGraphic[2][i])
    axs[0].plot(arrX[0],arrY[0])
    axs[0].set_xlabel('iteration')
    axs[0].set_ylabel('Min.')

    axs[1].plot(arrX[1],arrY[1])
    axs[1].set_xlabel('iteration')
    axs[1].set_ylabel('Max.')

    axs[2].plot(arrX[2],arrY[2])
    axs[2].set_xlabel('iteration')
    axs[2].set_ylabel('Avg.')

    plt.tight_layout()
    plt.show()



    print()
if __name__ == '__main__':
    main()




