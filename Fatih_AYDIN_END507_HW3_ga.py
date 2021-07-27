# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:11:16 2019

@author: fatih
"""
 
from random import randint
import random
import numpy as np
import pandas as pd
import math
import pdb
import operator
import functools

# Number of individuals in each generation
POPULATION_SIZE = 100

maxgen = 600

capacity = 5000
 
w_i = pd.read_csv('w.csv')
v_i = pd.read_csv('v.csv')

W = w_i.to_numpy()
V = v_i.to_numpy()


GENES = [0,1]

TARGET = 10000

weight = 0

trnmt = 10


class Individual(object):
    '''
    Class representing individual in population
    '''
    def __init__(self, chromosome):
        self.chromosome = chromosome
        self.fitness = self.cal_fitness()
        self.weight = weight
        
    @classmethod
    def m1_genes(self):
        
        flip = randint(0,1)
        if flip == 0:
                gene = 1
        else:
                gene = 0
        return gene

    @classmethod
    def m2_genes(self):
    
        flip = randint(0,1)
        if flip == 0:
                gene = 0
        else:
                gene = 1
        return gene    
        
    @classmethod
    def selection(population):
        tournament = (random.choice(population) for i in range(trnmt))
        tournament = sorted(tournament, key = lambda x:x.fitness, reverse = True)
        return tournament
        
    @classmethod
    def mutated_genes(self):
        global GENES
        gene = random.choice(GENES)
        return gene

    @classmethod
    def create_gnome(self):
        global TARGET
        gnome_len = 20
        return [self.mutated_genes() for _ in range(gnome_len)]
    
    @classmethod
    def initial_gnome(self):
        global TARGET
        
        gnome_len = 20
        #return [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0]
        #return [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
        
        #return [0, 0, 0, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 1]
        #return [0, 1, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 0]
        #return [1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 0, 0, 1, 0, 0, 1, 0, 0, 1, 1]
        #return [0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 0, 0, 1]
        #return [0, 0, 1, 0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 1, 0]
        #return [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0]
        #return [0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 1, 0, 0, 1, 0, 0, 1, 1, 0]
        #return [1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1]
    
    
    
    
    
    
    def mate(self, par2):
        '''
        Perform mating and produce new offspring
        '''

        child_chromosome = []
        q = 0
        for gp1, gp2 in zip(self.chromosome, par2.chromosome):

                   prob = random.random()

                   if prob < (0.45):
                           child_chromosome.append(gp1)

                   elif prob < (0.90):
                           child_chromosome.append(gp2)

                   else:
                           child_chromosome.append(self.mutated_genes())

        return Individual(child_chromosome)

    def cal_fitness(self):
        
        fitness = 0
        weight = 0

        nKromozom = list(map(int,self.chromosome))
        nK = np.asarray(nKromozom)
    
        for i in range(20):
            if weight <= capacity - W[i]:
                fitness += V[i]*nK[i]
                weight += W[i]*nK[i]
                
        for i in range(20):
            fitness -= nK[i]
        
        return fitness

def sumproduct(*lists):
    return sum(functools.reduce(operator.mul, data) for data in zip(*lists))

def main():
    global POPULATION_SIZE
    global generation
    global maxfitness
    #current generation 
    generation = 1
    maxfitness = 50000
    found = False
    population = []

    for _ in range(POPULATION_SIZE):
                #belirlenmiş bşl çözümü
                #gnome = Individual.initial_gnome()
                #rassal bşl çözümü
                gnome = Individual.create_gnome()
                
    population.append(Individual(gnome))

    while not found:
       
        population = sorted(population, key = lambda x:x.fitness, reverse = True)

        if population[0].fitness >= 50000:
            found = True
            break
        
        if generation == maxgen:
            found = True
            break
        
        if population[0].fitness >= maxfitness:
                maxfitness = population[0].fitness    
        
        if generation % 5 == 0:
            print("Generation: {}\Kromozom: {}\tFitness: {}".\
                  format(generation,
                  population[0].chromosome,
                  population[0].fitness))
                                 
        new_generation = []
           
        s = int((10*POPULATION_SIZE)/100)
        new_generation.extend(population[:s])
        
        s = int((90*POPULATION_SIZE)/100)
        for _ in range(s):
            
            parent1 = selection(population)[0]
            parent2 = selection(population)[0]
            
            #parent1 = random.choice(population[:50])
            #parent2 = random.choice(population[:50])
            
            child = parent1.mate(parent2)
        
        new_generation.append(child)
        population = new_generation     

        generation += 1

    print("Generation: {}\Kromozom: {}\tFitness: {} weight:{}".\
          format(generation,
          population[0].chromosome,
          population[0].fitness,
          population[0].weight))

if __name__ == '__main__':
    main()