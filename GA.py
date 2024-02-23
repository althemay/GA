import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits import mplot3d
from matplotlib.ticker import MultipleLocator, AutoMinorLocator
plt.rcParams.update({'font.size': 12})

def error(a, b):
   # defines the function from here https://www.youtube.com/watch?v=1i8muvzZkPw
    func = 3*(1 - a)**2*np.exp(-a**2-(b+1)**2) - 10*(a/5-a**3-b**5)*np.exp(-a**2-b**2) - (1/3)*np.exp(-b**2-(a+1)**2)
    return -func + 7.8

def get_xyz(f, a_limits=(-3, 3), b_limits=(-3, 3)):
    # sets up an xy grid with associated z values for 3d plotting
    a = np.linspace(a_limits[0], a_limits[1], num=100)
    b = np.linspace(b_limits[0], b_limits[1], num=100)

    mesh = np.meshgrid(a, b)
    z = np.vectorize(f)(mesh[0], mesh[1])

    return mesh[0], mesh[1], z

def fmt(x):
    return '{0:.{1}f}'.format(x, 1)

def cmap_map(function, cmap):
    """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
    This routine will break any discontinuous points in a colormap.
    """
    cdict = cmap._segmentdata
    step_dict = {}
    # Firt get the list of points where the segments start or end
    for key in ('red', 'green', 'blue'):
        step_dict[key] = list(map(lambda x: x[0], cdict[key]))
    step_list = sum(step_dict.values(), [])
    step_list = np.array(list(set(step_list)))
    # Then compute the LUT, and apply the function to the LUT
    reduced_cmap = lambda step : np.array(cmap(step)[0:3])
    old_LUT = np.array(list(map(reduced_cmap, step_list)))
    new_LUT = np.array(list(map(function, old_LUT)))
    # Now try to make a minimal segment definition of the new LUT
    cdict = {}
    for i, key in enumerate(['red','green','blue']):
        this_cdict = {}
        for j, step in enumerate(step_list):
            if step in step_dict[key]:
                this_cdict[step] = new_LUT[j, i]
            elif new_LUT[j,i] != old_LUT[j, i]:
                this_cdict[step] = new_LUT[j, i]
        colorvector = list(map(lambda x: x + (x[1], ), this_cdict.items()))
        colorvector.sort()
        cdict[key] = colorvector

    return matplotlib.colors.LinearSegmentedColormap('colormap',cdict,1024)

dark_jet = cmap_map(lambda x: x*0.75, matplotlib.cm.jet)

def _plot2d(f, a_limits=(-3, 3), b_limits=(-3, 3), max=(0, 1.6)):
    # creates a 2d contour plot
    a, b, z = get_xyz(f, a_limits, b_limits)
    
    fig = plt.figure(figsize=(8,8))
    ax = plt.axes()

    z = ax.contour(a, b, z, levels=22, cmap=dark_jet)
    ax.clabel(z, inline=True, fontsize=12,  fmt=fmt)

    ax.grid(True)
    ax.grid(True, which="minor", linestyle="dotted")

    ax.xaxis.set_major_locator(MultipleLocator(1))
    ax.xaxis.set_minor_locator(AutoMinorLocator(5))

    ax.yaxis.set_major_locator(MultipleLocator(1))
    ax.yaxis.set_minor_locator(AutoMinorLocator(5))

    ax.plot(*max, marker="*", markersize=18)

    return fig, ax

def plot2d(f, a_limits=(-3, 3), b_limits=(-3, 3), max=(0, 1.6)):
    # plots the optimization landscape
    _, ax = _plot2d(f, a_limits, b_limits, max)
    ax.set_xlabel('a',fontweight='bold')
    ax.set_ylabel('b',fontweight='bold')
    plt.show()

def plot3d(f, a_limits=(-3, 3), b_limits=(-3, 3)):
   # same, but plots in 3d with the same colorscheme
    a, b, z = get_xyz(f, a_limits, b_limits)

    fig = plt.figure(figsize=(14, 7))
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    #ax = plt.axes(projection='3d')
    ax.view_init(12,-135)
    ax.plot_surface(a, b, z, cmap=dark_jet, edgecolor='none')
    ax.set_xlabel('a',fontweight='bold')
    ax.set_ylabel('b',fontweight='bold')
    ax.set_zlabel('Error',fontweight='bold')
    
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    ax.view_init(-5,40)
    ax.plot_surface(a, b, z, cmap=dark_jet, edgecolor='none')
    ax.set_xlabel('a',fontweight='bold')
    ax.set_ylabel('b',fontweight='bold')
    ax.set_zlabel('Error',fontweight='bold')
    
    plt.show()

# Initialize the population with random values within the given range of the terrain
def initialize_population(population_size, gene_range=(-3, 3)):
    return np.random.uniform(gene_range[0], gene_range[1], (population_size, 2))

# Calculate the fitness of each individual in the population
def calculate_fitness(population):
    # The fitness is the negative error because we want to maximize fitness (minimize error)
    return np.array([-error(ind[0], ind[1]) for ind in population])

# Select the top individuals based on survival rate to be parents for the next generation
def selection(population, fitness, survival_rate):
    # Sort individuals by their fitness in descending order
    sorted_idx = np.argsort(fitness)[::-1]
    num_survivors = int(len(population) * survival_rate)
    # Return the top individuals based on survival rate
    return population[sorted_idx][:num_survivors]

# Generate offspring through crossover between parents
def crossover(parents, offspring_size):
    offspring = np.empty(offspring_size)
    crossover_point = np.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        # Select parents for crossover
        parent1_idx = k % parents.shape[0]
        parent2_idx = (k+1) % parents.shape[0]
        # Perform crossover and produce offspring
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring

# Mutate the offspring to introduce genetic variation
def mutate(offspring_crossover, mutation_rate):
    for idx in range(offspring_crossover.shape[0]):
        if np.random.rand() < mutation_rate:
            # Introduce a small random change to a gene
            random_value = np.random.uniform(-0.5, 0.5, 1)
            offspring_crossover[idx, np.random.randint(2)] += random_value
    # Ensure that the mutated gene values are within bounds
    return np.clip(offspring_crossover, -3, 3)

# The main genetic algorithm function
def gen_algo(population_size, n_generations, mutation_rate, survival_rate):
    population = initialize_population(population_size)
    fitness = calculate_fitness(population)
    best_idx = np.argmax(fitness)
    print(f'Generation 0 | Best Coordinates: {population[best_idx]} | Best Error: {-fitness[best_idx]}')

    for gen in range(1, n_generations + 1):
        # Select survivors based on fitness
        survivors = selection(population, fitness, survival_rate)
        # Generate offspring through crossover
        offspring = crossover(survivors, (population_size - survivors.shape[0], 2))
        # Introduce mutations to the offspring
        offspring = mutate(offspring, mutation_rate)
        # Form the new population for the next generation
        population = np.vstack((survivors, offspring))
        
        fitness = calculate_fitness(population)
        best_idx = np.argmax(fitness)
        print(f'Generation {gen} | Best Coordinates: {population[best_idx]} | Best Error: {-fitness[best_idx]}')

pop_size = 100  # Size of the initial population
n_gen = 50  # Number of generations
mutation = 0.02  # Mutation rate (0 to 1)
survival = 0.6  # Survival rate (0 to 1)

gen_algo(pop_size, n_gen, mutation, survival)

#TESTING
# the modified error function to create a new terrain 
def error(a, b):
    # Adjusted error function with slight modifications to the original coefficients and exponents
    func = 3*(1 - a)**3*np.exp(-a**2 - (b+1)**2) - 30*(a/5 - a**3 - b**5)*np.exp(-a**2 - b**2) - (1/3)*np.exp(-(b+1)**2 - (a+1)**2)
    return -func + 7.8

# visualizations of the new terrain
plot3d(error)
plot2d(error)

pop_size = 100  # Size of the initial population
n_gen = 50  # Number of generations
mutation = 0.02  # Mutation rate (0 to 1)
survival = 0.6  # Survival rate (0 to 1)

gen_algo(pop_size, n_gen, mutation, survival)