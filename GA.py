from Cromossome import Cromossome
from random import random

class GA:
    def __init__(self, args, population, generations, nrounds, mutation, crossover):
        self.nrounds = nrounds
        self.mutation = mutation
        self.crossover = crossover
        self.population = [ Cromossome(*args) for _ in range(population) ]
        self.generations = generations

    def __weighted_random(self, score):
        i = 0
        rnd = random()
        while rnd > 0:
            rnd -= 1 - self.population[i].fitness() / score
            i = i + 1

        return i

    def next_generation(self):
        score = sum(map(lambda x: x.fitness(), self.population))
        size = len(self.population)
        for _ in range(self.nrounds):
            if random() >= self.crossover:
                continue
            a = self.__weighted_random(score)
            b = a
            while b == a:
                b = self.__weighted_random(score)
            a, b = self.population[a], self.population[b]

            c = a.crossover(b, self.mutation)

            if c < self.population[-1]:
                for i in range(len(self.population)):
                    if self.population[i] > c:
                        self.population.insert(i, c)
                        score += c.fitness()
                        break

        self.population = self.population[:size]

    def evolve(self, early_stop=None, min_err=None):
        print("Sorting cromossomes by fitness...")
        self.population.sort()
        print("Starting evolutionary process...")
        best = self.population[0].fitness()
        count = 0
        for gen in range(self.generations + 1):
            print(f"Generation {gen}...")
            self.next_generation()
            print(f"Best: {self.population[0]}")
            if min_err is not None and best <= min_err:
                print(f"Early stop at generation {gen} because the desired error was reached")
                break
            if self.population[0].fitness() == best:
                count += 1
            else:
                count = 0
                best = self.population[0].fitness()
            if early_stop is not None and count >= early_stop:
                print(f"Early stop at generation {gen} because the fitness is not improving")
                break

        return self.population[0]


