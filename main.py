import random
import matplotlib.pyplot as plt
import numpy as np
# константы задачи
   # длина подлежащей оптимизации битовой строки

# константы генетического алгоритма
POPULATION_SIZE = 100   # количество индивидуумов в популяции
P_CROSSOVER = 0.9       # вероятность скрещивания
P_MUTATION = 0.05        # вероятность мутации индивидуума
MAX_GENERATIONS = 25000    # максимальное количество поколений
RANDOM_SEED = 42
random.seed(RANDOM_SEED)

def ToBin(word):
    BinWord=[]
    for i in range(len(word)-1,-1,-1):
        n=word[i]
        while n > 0:
            BinWord.insert(0,n % 2)
            n = n // 2
        while(len(BinWord)%7!=0):
            BinWord.insert(0,0)
    return BinWord

Word=input()
charac=[]
for c in Word:
    if(c!=" "):
        charac.append(ord(c)-1038)
    else:
        charac.append(1)
print(charac)
MAS=ToBin(charac)
print(MAS)

ONE_MAX_LENGTH = len(MAS)


class FitnessMax():
    def __init__(self):
        self.values = [0]


class Individual(list):
    def __init__(self, *args):
        super().__init__(*args)
        self.fitness = FitnessMax()

def Indaclub(indamix):
    indaword=''
    for i in range(len(indamix)//7):
        z = 0
        for j in range(7):
            z+= indamix[i*7 + j]*2 ** (7 - 1 - j)
        if(z!=1):
            indaword+=chr(z+1038)
        else:
            indaword+=' '
    return indaword


def Fitness(individual):
    prodigy=0
    for i in range (len(MAS)):
        if(individual[i]==MAS[i]):
            prodigy+=1
    return prodigy, # кортеж


def individualCreator():
    return Individual([random.randint(0, 1) for i in range(ONE_MAX_LENGTH)])

def populationCreator(n = 0):
    return list([individualCreator() for i in range(n)])


population = populationCreator(n=POPULATION_SIZE)
generationCounter = 0

fitnessValues = list(map(Fitness, population))

for individual, fitnessValue in zip(population, fitnessValues):
    individual.fitness.values = fitnessValue

maxFitnessValues = []
meanFitnessValues = []

def clone(value):
    ind = Individual(value[:])
    ind.fitness.values[0] = value.fitness.values[0]
    return ind

def selTournament(population, p_len):
    offspring = []
    for n in range(p_len):
        i1 = i2 = i3 = 0
        while i1 == i2 or i1 == i3 or i2 == i3:
            i1, i2, i3 = random.randint(0, p_len-1), random.randint(0, p_len-1), random.randint(0, p_len-1)

        offspring.append(max([population[i1], population[i2], population[i3]], key=lambda ind: ind.fitness.values[0]))

    return offspring

def cxOnePoint(child1, child2):
    s = random.randint(2, len(child1)-3)
    child1[s:], child2[s:] = child2[s:], child1[s:]

def mutFlipBit(mutant, indpb=0.01):
    for indx in range(len(mutant)):
        if random.random() < indpb:
            mutant[indx] = 0 if mutant[indx] == 1 else 1


fitnessValues = [individual.fitness.values[0] for individual in population]

while max(fitnessValues) < ONE_MAX_LENGTH and generationCounter < MAX_GENERATIONS:
    generationCounter += 1
    offspring = selTournament(population, len(population))
    offspring = list(map(clone, offspring))

    for child1, child2 in zip(offspring[::2], offspring[1::2]):
        if random.random() < P_CROSSOVER:
            cxOnePoint(child1, child2)

    for mutant in offspring:
        if random.random() < P_MUTATION:
            mutFlipBit(mutant, indpb=1.0/ONE_MAX_LENGTH)

    freshFitnessValues = list(map(Fitness, offspring))
    for individual, fitnessValue in zip(offspring, freshFitnessValues):
        individual.fitness.values = fitnessValue

    population[:] = offspring

    fitnessValues = [ind.fitness.values[0] for ind in population]

    maxFitness = max(fitnessValues)
    meanFitness = sum(fitnessValues) / len(population)
    maxFitnessValues.append(maxFitness)
    meanFitnessValues.append(meanFitness)
    print(f"Поколение {generationCounter}: Макс приспособ. = {maxFitness}, Средняя приспособ.= {meanFitness}")

    best_index = fitnessValues.index(max(fitnessValues))
    print("Лучший индивидуум = ", population[best_index], "\n")
    print("Слово = ", Indaclub(population[best_index]), "\n")
plt.plot(maxFitnessValues, color='red')
plt.plot(meanFitnessValues, color='green')
plt.xlabel('Поколение')
plt.ylabel('Макс/средняя приспособленность')
plt.title('Зависимость максимальной и средней приспособленности от поколения')
plt.show()