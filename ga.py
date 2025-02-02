import numpy as np
import matplotlib.pyplot as plt

# Define the objective function to minimize
def objective_function(x):
    return x**2 + 3*x + 4

# Genetic Algorithm
def genetic_algorithm(population_size, chromosome_length, mutation_rate, num_generations):
    # Generate initial population
    population = np.random.randint(0, 2, size=(population_size, chromosome_length))
    
    # Initialize lists to store best fitness values
    best_fitness_values = []

    # Main loop for evolution
    for generation in range(num_generations):
        # Evaluate the fitness of each individual in the population
        fitness_scores = [1 / (objective_function(int(''.join(map(str, chromosome)), 2)) + 1) for chromosome in population]
        best_fitness = max(fitness_scores)
        best_fitness_values.append(best_fitness)

        # Select parents for crossover based on their fitness
        selected_indices = np.random.choice(range(population_size), size=population_size, replace=True, p=fitness_scores/np.sum(fitness_scores))
        parents = population[selected_indices]

        # Perform crossover (single-point crossover)
        crossover_point = np.random.randint(1, chromosome_length)
        offspring = np.empty_like(population)
        for i in range(0, population_size, 2):
            parent1, parent2 = parents[i], parents[i+1]
            offspring[i][:crossover_point] = parent1[:crossover_point]
            offspring[i][crossover_point:] = parent2[crossover_point:]
            offspring[i+1][:crossover_point] = parent2[:crossover_point]
            offspring[i+1][crossover_point:] = parent1[crossover_point:]

        # Perform mutation
        for i in range(population_size):
            if np.random.rand() < mutation_rate:
                mutation_point = np.random.randint(chromosome_length)
                offspring[i][mutation_point] = 1 - offspring[i][mutation_point]

        # Replace the old population with the new offspring
        population = offspring

    return best_fitness_values

# Parameters
population_size = 20
chromosome_length = 10
mutation_rate = 0.1
num_generations = 100

# Run Genetic Algorithm
best_fitness_values = genetic_algorithm(population_size, chromosome_length, mutation_rate, num_generations)

# Plot the progression of the best fitness value over generations
plt.plot(range(num_generations), best_fitness_values)
plt.xlabel('Generation')
plt.ylabel('Best Fitness Value')
plt.title('Progression of Best Fitness Value over Generations')
plt.show()
