# Genetic Programming

In artificial intelligence, genetic programming (GP) is a technique of evolving programs, starting from a population of unfit (usually random) programs, fit for a particular task by applying operations analogous to natural genetic processes to the population of programs. It is essentially a heuristic search technique often described as 'hill climbing', i.e. searching for an optimal or at least suitable program among the space of all programs.

The operations are: selection of the fittest programs for reproduction (crossover) and mutation according to a predefined fitness measure, usually proficiency at the desired task. The crossover operation involves swapping random parts of selected pairs (parents) to produce new and different offspring that become part of the new generation of programs. Mutation involves substitution of some random part of a program with some other random part of a program. Some programs not selected for reproduction are copied from the current generation to the new generation. Then the selection and other operations are recursively applied to the new generation of programs.

Typically, members of each new generation are on average more fit than the members of the previous generation, and the best-of-generation program is often better than the best-of-generation programs from previous generations. Termination of the recursion is when some individual program reaches a predefined proficiency or fitness level.

It may and often does happen that a particular run of the algorithm results in premature convergence to some local maximum which is not a globally optimal or even good solution. Multiple runs (dozens to hundreds) are usually necessary to produce a very good result. It may also be necessary to increase the starting population size and variability of the individuals to avoid pathologies.

Source : https://en.wikipedia.org/wiki/Genetic_programming
