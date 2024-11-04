import streamlit as st
import numpy as np
import pandas as pd

import random, string
import sys, os
import copy
import numpy as np
import time

# Assuming the GeneticAlgorithm class is already implemented
# from your_module import GeneticAlgorithm

# Title of the Streamlit app
st.title("Genetic Algorithm Optimization")

# Input parameters
st.header("Configure Genetic Algorithm Parameters")

mutation_depth = st.slider("Mutation Depth", min_value=1, max_value=10, value=3)
iterations = st.slider("Iterations", min_value=10, max_value=1000, value=479)
mutation_probability = st.slider("Mutation Probability", min_value=0.0, max_value=1.0, value=0.1)
generation_size = st.slider("Generation Size", min_value=1, max_value=100, value=20)

# User input for other parameters
num_facilities = st.number_input("Number of Facilities", min_value=1, max_value=1000, value=134)
p_value = st.number_input("P Value (Cluster Count)", min_value=1, max_value=1000, value=134)

# Upload or generate a cost matrix
st.header("Upload or Generate Cost Matrix")

uploaded_file = st.file_uploader("Upload a CSV file for cost matrix", type="csv")
if uploaded_file:
    dist_data = pd.read_csv(uploaded_file)
    cost_matrix = np.matrix(dist_data.iloc[:, 1:])
else:
    # Generate a random cost matrix if no file is uploaded
    st.write("No file uploaded, generating a random cost matrix...")
    cost_matrix = np.random.randint(1, 100, size=(num_facilities, num_facilities))

# Checkbox options for the optimization process
init_population_with_center_point = st.checkbox("Initialize Population with Center Point", value=False)
apply_hypermutation = st.checkbox("Apply Hypermutation", value=False)



class Chromosome:

    def __init__(self, content, fitness):
        self.content = content
        self.fitness = fitness
    def __str__(self): return "%s f=%d" % (self.content, self.fitness)
    def __repr__(self): return "%s f=%d" % (self.content, self.fitness)

class GeneticAlgorithm:

    def __init__(self,input_list, num_facilities, p, cost_matrix, init_population_with_center_point=False, apply_hypermutation=False):

        self.input_list = input_list
        self.num_facilities = num_facilities
        self.p = p
        self.cost_matrix = cost_matrix

        self.init_population_with_center_point = init_population_with_center_point
        self.apply_hypermutation = apply_hypermutation

        #self.iterations = 50                                      # Maximal number of iterations
        self.current_iteration = 0
        #self.generation_size = 35                                 # Number of individuals in one generation
        #self.reproduction_size = 13                               # Number of individuals for reproduction

        #self.mutation_prob = 0.3                                  # Mutation probability
        self.hypermutation_prob = 0.03                            # Hypermutation probability
        self.hypermutation_population_percent = 10

        self.mutation_depth = input_list[0]
        self.iterations = input_list[1]
        self.mutation_prob = input_list[2]
        self.generation_size = input_list[3]

        self.reproduction_size =  int ( self.generation_size /3  )                         # Number of individuals for reproduction


        self.top_chromosome = None      # Chromosome that represents solution of optimization process
        self.wrost_chromosome = None      # Chromosome that represents solution of optimization process



    def mutation(self, chromosome):
        """
        Applies mutation over chromosome with probability self.mutation_prob
        In this process, a randomly selected median is replaced with a randomly selected demand point.
        """

        mp = random.random()
        if mp < self.mutation_prob:
            # index of ra ndomly selected median:
            for a in range(int ( self.mutation_depth*self.p)  ):
              i = random.randint(0, len(chromosome)-1)
              i = random.randint(0, len(chromosome)-1)
            # demand points without current medians:
              demand_points = [element for element in range(0,len(self.cost_matrix)) if element not in chromosome]
            # replace selected median with randomly selected demand point:
              chromosome[i] = random.choice(demand_points)

        return chromosome

    def crossover(self, parent1, parent2):

        identical_elements = [element for element in parent1 if element in parent2]

        # If the two parents are equal to each other, one of the parents is reproduced unaltered for the next generation
        # and the other parent is deleted, to avoid that duplicate individuals be inserted into the population.
        if len(identical_elements) == len(parent1):
            return parent1, None

        child1 = []
        child2 = []

        exchange_vector_for_parent1 = [element for element in parent1 if element not in identical_elements]
        exchange_vector_for_parent2 = [element for element in parent2 if element not in identical_elements]

        c = random.randint(0, len(exchange_vector_for_parent1)-1)

        for i in range(c):
            exchange_vector_for_parent1[i], exchange_vector_for_parent2[i] = exchange_vector_for_parent2[i], exchange_vector_for_parent1[i]

        child1 = identical_elements + exchange_vector_for_parent1
        child2 = identical_elements + exchange_vector_for_parent2

        return child1, child2


    def cost_to_nearest_median(self, facility, medians):
        """ For given facility, returns cost to its nearest median """
        min_cost = self.cost_matrix[facility, medians[0]]
        for median in medians:
            if min_cost > self.cost_matrix[facility, median]:
                min_cost = self.cost_matrix[facility, median]
        return min_cost

    def fitness(self, chromosome):
        """ Calculates fitness of given chromosome """
        cost_sum = 0
        for i in range(self.num_facilities):
            cost_sum += self.cost_to_nearest_median(i, chromosome)
        return cost_sum


    def initial_random_population(self):
        """
        Creates initial population by generating self.generation_size random individuals.
        Each individual is created by randomly choosing p facilities to be medians.
        """

        init_population = []
        for k in range(self.generation_size):
            rand_medians = []
            facilities = list(range(self.num_facilities))
            for i in range(self.p):
                rand_median = random.choice(facilities)
                rand_medians.append(rand_median)
                facilities.remove(rand_median)
            init_population.append(rand_medians)

        init_population = [Chromosome(content, self.fitness(content)) for content in init_population]
        self.top_chromosome = min(init_population, key=lambda chromo: chromo.fitness)
        self.wrost_chromosome = max(init_population, key=lambda chromo: chromo.fitness)
        print("Current top solution: %s" % self.top_chromosome)
        print("Current top solution: %s" % self.wrost_chromosome)

        return init_population


    def selection(self, chromosomes):
        """Ranking-based selection method"""

        # Chromosomes are sorted ascending by their fitness value
        chromosomes.sort(key=lambda x: x.fitness)
        L = self.reproduction_size
        selected_chromosomes = []

        for i in range(int (self.reproduction_size)):
            j = L - np.floor((-1 + np.sqrt(1 + 4*random.uniform(0, 1)*(L**2 + L))) / 2)
            selected_chromosomes.append(chromosomes[int(j)])
        return selected_chromosomes


    def create_generation(self, for_reproduction):
        """
        Creates new generation from individuals that are chosen for reproduction,
        by applying crossover and mutation operators.
        Size of the new generation is same as the size of previous.
        """
        new_generation = []

        while len(new_generation) < self.generation_size:
            parents = random.sample(for_reproduction, 2)
            child1, child2 = self.crossover(parents[0].content, parents[1].content)

            self.mutation(child1)
            new_generation.append(Chromosome(child1, self.fitness(child1)))

            if child2 != None and len(new_generation) < self.generation_size:
                self.mutation(child2)
                new_generation.append(Chromosome(child2, self.fitness(child2)))

        return new_generation


    def nearest_median(self, facility, medians):
        """ Returns the nearest median for given facility """
        min_cost = self.cost_matrix[facility, medians[0]]
        nearest_med = medians[0]
        for median in medians:
            if min_cost > self.cost_matrix[facility, median]:
                nearest_med = median
        return nearest_med


    def initial_population_with_center_point(self):
        """
        Creates initial population.
        Based on paper: Oksuz, Satoglu, Kayakutlu: 'A Genetic Algorithm for the P-Median Facility Location Problem'
        """

        init_population = []
        for k in range(self.generation_size):

            # Randomly select p-medians
            medians = []
            facilities = list(range(self.num_facilities))
            for i in range(self.p):
                rand_median = random.choice(facilities)
                medians.append(rand_median)
                facilities.remove(rand_median)

            # Assign all demand points to nearest median
            median_nearestpoints_map = dict((el, []) for el in medians)
            for i in range(self.num_facilities):
                median_nearestpoints_map[self.nearest_median(i, medians)].append(i)

            n = len(medians)
            # For each median
            for i in range(n):
                median = medians[i]
                # Determine the center point which has minimum distance to all demand points
                # that assigned this median
                min_dist = float(np.inf)
                center_point = median

                cluster = [median] + median_nearestpoints_map[median]
                for point in cluster:
                    dist = 0
                    for other_point in cluster:
                        dist += self.cost_matrix[point, other_point]
                    if dist < min_dist:
                        min_dist = dist
                        center_point = point

                # Replace the median with center point
                medians[i] = center_point

            init_population.append(medians)

        init_population = [Chromosome(content, self.fitness(content)) for content in init_population]
        self.top_chromosome = min(init_population, key=lambda chromo: chromo.fitness)
        self.wrost_chromosome = min(init_population, key=lambda chromo: chromo.fitness)

        print("Current top solution: %s" % self.top_chromosome)
        print("Current top solution: %s" % self.wrost_chromosome)

        return init_population



    def optimize(self):
        report = []
        start_time = time.time()

        if self.init_population_with_center_point:
            chromosomes = self.initial_population_with_center_point()
        else:
            chromosomes = self.initial_random_population()

        while self.current_iteration < self.iterations:

            if self.current_iteration % 10 == 0:
              print("Iteration: %d" % self.current_iteration)

            # From current population choose individuals for reproduction
            for_reproduction = self.selection(chromosomes)

            # Create new generation from individuals that are chosen for reproduction
            chromosomes = self.create_generation(for_reproduction)

            if self.apply_hypermutation:
                hp = random.random()
                if hp < self.hypermutation_prob:
                    print("Hypermutation...")

                    chromosomes_content = [chromo.content for chromo in chromosomes]

                    # choose individuals on which hypermutation will be applied
                    k = int(self.generation_size * self.hypermutation_population_percent / 100)
                    individuals_subset = random.sample(chromosomes_content, k)

                    for individual in individuals_subset:
                        chromosomes_content.remove(individual)

                    new_individuals_subset = self.hypermutation(individuals_subset)

                    for individual in new_individuals_subset:
                        chromosomes_content.append(individual)

                    chromosomes = [Chromosome(chromo_content, self.fitness(chromo_content)) for chromo_content in chromosomes_content]


            self.current_iteration += 1

            chromosome_with_min_fitness = min(chromosomes, key=lambda chromo: chromo.fitness)
            chromosome_with_max_fitness = max(chromosomes, key=lambda chromo: chromo.fitness)

            fitness_values = [chromosome.fitness for chromosome in chromosomes]
            average_fitness = sum(fitness_values) / len(fitness_values)

            if chromosome_with_min_fitness.fitness < self.top_chromosome.fitness:
                self.top_chromosome = chromosome_with_min_fitness
            #print("\n Current top solution: %s" % self.top_chromosome)

            rep = {"iter": self.current_iteration, "top": self.top_chromosome.fitness, "wrost": chromosome_with_max_fitness.fitness,"time":time.time()-start_time,
                   "avg_fitness":average_fitness,
                   "best_fit":self.top_chromosome.fitness , "best":self.top_chromosome.content}
            report.append(rep)
            if self.current_iteration % 10 == 0:
              print(rep)

        end_time = time.time()
        self.time = end_time - start_time
        hours, rem = divmod(end_time - start_time, 3600)
        minutes, seconds = divmod(rem, 60)


        report_name = "_".join(map(str, self.input_list)) + ".csv"
        report_df = pd.DataFrame(report)
        report_df.to_csv(report_name)

        print("Final top solution: %s" % self.top_chromosome)
        print('Time: {:0>2}:{:0>2}:{:05.2f}'.format(int(hours), int(minutes), seconds))
        return "Final top solution: %s" % self.top_chromosome


    def hypermutation(self, individuals_subset):

        n = len(individuals_subset)

        # FOR EACH individual X from selected individuals DO
        for idx in range(n):
            X = individuals_subset[idx]

            # Let H be the set of facility indexes that are not currently present
            # in the genotype of individual X
            H = [element for element in range(self.num_facilities) if element not in X]

            # FOR EACH facility index “i” included in set H DO
            for i in H:

                best = X

                # FOR EACH facility index “j” that is currently present in the genotype of
                # the individual X DO
                for j in X:

                    # Let Y be a new individual with the set of facilities given by: (X – {j}) ∪ {i}
                    Y = copy.deepcopy(X)
                    Y.remove(j)
                    Y = Y + [i]

                    if self.fitness(Y) < self.fitness(best):
                        best = Y

                if self.fitness(best) < self.fitness(X):
                    # Insert the new X into the population, replacing the old X
                    individuals_subset[idx] = best

        return individuals_subset


    def medians_and_assigned_demand_points(self):
      median_nearestpoints_map = dict((median, []) for median in self.top_chromosome.content)
      for i in range(self.num_facilities):
        median_nearestpoints_map[self.nearest_median(i, self.top_chromosome.content)].append(i)
      return median_nearestpoints_map


# Run the Genetic Algorithm
if st.button("Run Optimization"):
    # Initialize the Genetic Algorithm
    genetic = GeneticAlgorithm(
        (mutation_depth, iterations, mutation_probability, generation_size),
        num_facilities,
        p_value,
        cost_matrix,
        init_population_with_center_point=init_population_with_center_point,
        apply_hypermutation=apply_hypermutation
    )
    
    # Optimize and display results
    result = genetic.optimize()
    st.write("Optimization Result:", genetic.top_chromosome)
    st.write("Result Fitness :", genetic.top_chromosome.fitness)

    hours, rem = divmod(genetic.time, 3600)
    minutes, seconds = divmod(rem, 60)

    st.write(f"Time: {hours}h {minutes}m {int(seconds)}s ")
else:
    st.write("Click 'Run Optimization' to start the Genetic Algorithm.")

# Footer information
st.write("Adjust the parameters and rerun to see different results.")
