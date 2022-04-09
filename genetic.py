from math import log2, ceil
from random import choices, randint, shuffle, uniform

"""
A bit of explanation about formulas used down below:

(b-a)*10^p is the nr of numbers in the interval [a,b) with precision of p digits after the period
for example:
    if we need to know how many integers (numbers with p = 0) are in the the interval [-1,2)
    the answer is (2 - (-1))*10^0 = 3*1 = 3
    the same for p = 1 (numbers with 1 digit) but the result is (2-(-1))*10^1 = 30


length of the chromosome is on how many bits in total we can represent the amount of numbers from above.
so that if there are 10 numbers ceil(log2(10)) = 4, which means that all 10 numbers can be represented on minimum 4 bits. 
"""

#             THE GENETIC ALGORITHM   

# Globals

# population size
population_size = 0

# a and b of the domain
a = 0
b = 0

# Coefficients of the polinomial fitness function
a1 = 0
a2 = 0
a3 = 0

# Precision
p = 0

# Probabilities of crossover and mutation
crossover_probability = 0
mutation_probability = 0

# Number of steps in the algorithm
number_steps = 0

# Evolution of the best value
evolution_of_best_value = []

# Input and output file
fin = open("input.txt","r")
fout = open("output.txt","w")

# Read the input data
def read_input():
    global population_size, a, b, a1, a2, a3, p, crossover_probability, mutation_probability, number_steps, evolution_of_best_value
    population_size = int(fin.readline())
    a = float(fin.readline())
    b = float(fin.readline())
    a1 = float(fin.readline())
    a2 = float(fin.readline())
    a3 = float(fin.readline())
    p = float(fin.readline())
    crossover_probability = float(fin.readline())
    mutation_probability = float(fin.readline())
    number_steps = int(fin.readline())

# Write the initial data in the output file
def write_data_fancy():
    fout.write("-----------------------------------------------\n\n")
    fout.write("INPUT DATA:\n\n")
    fout.write("Population size = {}\n".format(population_size))
    fout.write("Function domain D[a,b] = [{},{}]\n".format(a,b))
    fout.write("Fitness function f(x) = ({})*x^2 + ({})*x + ({})\n".format(a1,a2,a3))
    fout.write("Precision = {}\n".format(p))
    fout.write("Crossover probability = {}\n".format(crossover_probability))
    fout.write("Mutation probability = {}\n".format(mutation_probability))
    fout.write("Number of steps in the algorithm = {}\n\n".format(number_steps))


def generate_population():
    # Define maximum length of the chromosome
    global l
    l = ceil(log2((b - a)*(10**p)))

    # Generate each individual of the population
    population = []
    for i in range(population_size):
        # Randomly generate chromosomes of the individuals of size l
        individual = choices([0,1],k=l)
        population.append(individual)
    return population

# Value corresponding to the chromosome in the domain
def translated_value(chromosome):
    number = (b-a)/(pow(2,l)-1)*int(chromosome,2)+a
    return number

# Fitness function
def fitness(translated):
    x = translated
    number = a1 * pow(x,2) + a2 * x + a3
    return number


# Write the individuals as
# index of individual, chromosome as binary number, translated chromosome value, fitness value 
def write_population(population):
    for i in range(len(population)):
        individual = population[i]
        chromosome = "".join(map(str, individual))
        translated = translated_value(chromosome)
        fit = fitness(translated)
        fout.write("{}: chromosome = {}   translated = {}   fitness = {}\n".format(i+1,chromosome, translated, fit))
    fout.write("\n")

def write_probabilities_selected(probability_selected):
    fout.write("-----------------------------------------------\n\n")
    fout.write("SELECTION PROBABILITIES:\n\n")
    for i in range(len(probability_selected)):
        fout.write("chromosome {} = {}\n".format(i+1,probability_selected[i]))
    

def write_intervals(intervals):
    fout.write("-----------------------------------------------\n\n")
    fout.write("INTERVALS FOR SELECTION PROBABILITIES:\n\n")
    for i in range(len(intervals)):
        fout.write("chromosome {} = [{},{})\n".format(i+1,intervals[i][0],intervals[i][1]))

# Binary search to find the index of the interval
def binary_search_intervals(number,intervals):
    start = 0
    end = len(intervals) - 1
    while start <= end:
        middle = (start + end) // 2
        left = intervals[middle][0]
        right = intervals[middle][1]
        
        if left <= number and number < right:
            return middle
        
        if number >= right:
            start = middle + 1
        else:
            end = middle - 1


def selection(population):
    global step
    # Calculate fitness for each individual and total sum 
    fit_all = []
    for i in range(len(population)):
        individual = population[i]
        chromosome = "".join(map(str, individual))
        translated = translated_value(chromosome)
        fit = fitness(translated)
        fit_all.append(fit)
    total_sum_F = sum(fit_all)

    # Find the individual with the best fitness
    best_fitness = fit_all[0]
    index_best_fitness = 0
    for i in range(1,len(fit_all)):
        if fit_all[i] > best_fitness:
            best_fitness = fit_all[i]
            index_best_fitness = i
    evolution_of_best_value.append((best_fitness,total_sum_F/population_size))

    # List of all selected indexes
    selected = []
    

    # Calculate the probability of each individual to be selected
    # using formula 
    # prob_selected(i) = f(x(i))/ F
    # where f(x(i)) is fitness of individual and F is a total sum of all "fitnesses"
    probability_selected = list(map(lambda prob: prob / total_sum_F, fit_all))
    
    

    # Calculate the intervals for the probabilities
    intervals = []
    intervals.append((0,probability_selected[0]))
    for i in range(1,len(probability_selected)):
        previous_prob = intervals[i-1][1]
        following_prob = previous_prob + probability_selected[i]
        intervals.append((previous_prob,following_prob))
    if step == 1:
        write_probabilities_selected(probability_selected)
        write_intervals(intervals)
    
    # For (population_size - 1) times generate uniform number in [0,1) and choose individuals 
    # where this number falls into the interval
    if step == 1:
        fout.write("-----------------------------------------------\n\n")
        fout.write("SELECTING CHROMOSOMES OUT OF ALL:\n\n")
    for i in range(population_size - 1):
        uniform_number = uniform(0,1)
        
        # Binary search to find the index of the interval
        index_element = binary_search_intervals(uniform_number, intervals)
        if step == 1:
            fout.write("u = {} --> chromosome {} was selected\n".format(uniform_number,index_element+1))
        selected.append(index_element)
                
    
    # Remodel selected list so that instead of indexes it contains chromosomes
    selected = [population[x] for x in selected]
    best_individual = population[index_best_fitness]
    return (best_individual,selected)

def shuffle_suffixes(pop):
    
    # Generate random int between 1 and l to "cut" the numbers off
    cut = randint(1,l-1)
    prefix = [ x[:cut] for x in pop ]
    suffix = [ x[cut:] for x in pop ]

    shuffle(prefix)
    shuffle(suffix)

    # Combine the numbers back
    result = [ prefix[i] + suffix[i] for i in range(len(prefix))]
    if step == 1:
        fout.write("-----------------------------------------------\n\n")
        fout.write("SHUFFLING THE PREFIXES AND SUFFIXES CUT AFTER {}TH INDEX:\n\n".format(cut))
    return result

    

def crossover(population):
    if step == 1:
        fout.write("-----------------------------------------------\n\n")
        fout.write("SELECTING CHROMOSOMES FOR CROSSOVER:\n\n")
    # For crossover participation random individuals will be selected
    # based on probability of crossover
    ready_for_crossover = []
    ready_for_crossover_indexes = []
    for i in range(len(population)):
        u = uniform(0,1)
        if step == 1:
            fout.write("u = {} -->  ".format(u))
        if u < crossover_probability:
            ready_for_crossover.append(population[i])
            ready_for_crossover_indexes.append(i)
            if step == 1:
                fout.write("chromosome {} was selected\n".format(i+1))
        else:
            if step == 1:
                fout.write("chromosome {} was NOT selected\n".format(i+1))

    done_with_crossover = shuffle_suffixes(ready_for_crossover)
    for i in range(len(ready_for_crossover_indexes)):
        index = ready_for_crossover_indexes[i]
        population[index] = done_with_crossover[i]
    return population


def mutate_one_bit(list_of_bits):
    random_mutation_index = randint(0,l-1)
    list_of_bits[random_mutation_index] = 0 if list_of_bits[random_mutation_index] == 1 else 1
    return list_of_bits


def mutation(population):
    if step == 1:
        fout.write("-----------------------------------------------\n\n")
        fout.write("SELECTING CHROMOSOMES FOR MUTATION:\n\n")
    # For mutation participation random individuals will be selected
    # based on probability of mutation
    ready_for_mutation = []
    ready_for_mutation_indexes = []
    for i in range(len(population)):
        u = uniform(0,1)
        if step == 1:
            fout.write("u = {} -->  ".format(u))
        if u < mutation_probability:
            ready_for_mutation.append(population[i])
            ready_for_mutation_indexes.append(i)
            if step == 1:
                fout.write("chromosome {} was selected\n".format(i+1))
        else:
            if step == 1:
                fout.write("chromosome {} was NOT selected\n".format(i+1))
    done_with_mutations = list(map(mutate_one_bit,ready_for_mutation))
    for i in range(len(ready_for_mutation_indexes)):
        index = ready_for_mutation_indexes[i]
        population[index] = done_with_mutations[i]
    return population

def write_evolution_of_maxim():
    fout.write("-----------------------------------------------\n\n")
    fout.write("EVOLUTION ITSELF\n\n")
    fout.write("fittest individual --- average fitness\n\n")
    for i in range(len(evolution_of_best_value)):
        fout.write("{} --- {}\n".format(evolution_of_best_value[i][0],evolution_of_best_value[i][1]))
  

# Main actions here
read_input()
write_data_fancy()
population = generate_population()
fout.write("-----------------------------------------------\n\n")
fout.write("INITIAL POPULATION:\n\n")
write_population(population)


# Start the algorithm
step = 1
while step <= number_steps:
    best_and_rest = selection(population)
    if step == 1:
        fout.write("-----------------------------------------------\n\n")
        fout.write("POPULATION AFTER SELECTION:\n\n")
        all_individuals = best_and_rest[1]
        all_individuals.append(best_and_rest[0])
        write_population(all_individuals)
    the_fittest = best_and_rest[0]
    population_after_crossover = crossover(best_and_rest[1])
    if step == 1:
        fout.write("-----------------------------------------------\n\n")
        fout.write("POPULATION AFTER CROSSOVER:\n\n")
        all_individuals = population_after_crossover
        all_individuals.append(best_and_rest[0])
        write_population(all_individuals)
    population_after_mutation = mutation(population_after_crossover)
    population_after_mutation.append(the_fittest)
    population = population_after_mutation
    step += 1
    
write_evolution_of_maxim()


fin.close()
fout.close()
