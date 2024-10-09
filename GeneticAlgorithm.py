import numpy as np
import random
import matplotlib.pyplot as plt
import time


class GeneticAlgorithm:

    def __init__(self, interval=(-20, -3.1), population_size=100, mutation_rate=0.07, crossover_rate=0.7,
                 num_generations=101):
        self.interval = interval
        self.population_size = population_size
        self.num_generations = num_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.population = []
        self.all_population_points = []
        self.generations_points = {}

    def fitness_function(self, x):
        """Фитнес-функция для оптимизации"""
        if self.interval[0] <= x <= self.interval[1]:
            return np.sin(2 * x) / (x ** 2)
        else:
            return float('-inf')

    def generate_population(self):
        """Генерация начальной популяции"""
        self.population = [random.uniform(self.interval[0], self.interval[1]) for _ in range(self.population_size)]

    def evaluate_population(self):
        """Оценка приспособленности (fitness) популяции"""
        return [self.fitness_function(ind) for ind in self.population]

    def select_parents(self, fitnesses):
        """Отбор родителей с помощью рулетки"""
        min_fitness = min(fitnesses)
        if min_fitness < 0:
            fitnesses = [f - min_fitness for f in fitnesses]

        total_fitness = sum(fitnesses)
        probabilities = [f / total_fitness for f in fitnesses]

        parents = [np.random.choice(self.population, p=probabilities) for _ in range(self.population_size)]
        return parents

    def float_to_binary(self, value):
        """Преобразование float в бинарную строку"""
        if value < 0:
            return '-' + self.float_to_binary(-value)

        if value == 0.0:
            return '0.0'

        integer_part = int(value)
        fractional_part = value - integer_part

        binary_integer_part = bin(integer_part)[2:]

        binary_fractional_part = []
        while fractional_part > 0 and len(binary_fractional_part) < 52:
            fractional_part *= 2
            bit = int(fractional_part)
            binary_fractional_part.append(str(bit))
            fractional_part -= bit

        if binary_fractional_part:
            return f'{binary_integer_part}.' + ''.join(binary_fractional_part)
        else:
            return binary_integer_part

    def binary_to_float(self, binary_str):
        """Преобразование бинарной строки обратно в float"""
        is_negative = binary_str.startswith('-')
        if is_negative:
            binary_str = binary_str[1:]

        if binary_str.startswith('+'):
            binary_str = binary_str[1:]

        if '.' in binary_str:
            int_part_str, frac_part_str = binary_str.split('.')
        else:
            int_part_str, frac_part_str = binary_str, ''

        int_part = int(int_part_str, 2) if int_part_str else 0

        frac_part = 0.0
        for i, bit in enumerate(frac_part_str):
            frac_part += int(bit) * (2 ** -(i + 1))

        result = int_part + frac_part
        if is_negative:
            result = -result

        return result

    def align_fractional_parts(self, bin_str1, bin_str2):
        """Выравнивание дробных частей для кроссовера"""

        def split_sign(s):
            if s.startswith('-') or s.startswith('+'):
                return s[0], s[1:]
            else:
                return '', s

        sign1, num_str1 = split_sign(bin_str1)
        sign2, num_str2 = split_sign(bin_str2)

        int_part1, frac_part1 = num_str1.split('.') if '.' in num_str1 else (num_str1, '')
        int_part2, frac_part2 = num_str2.split('.') if '.' in num_str2 else (num_str2, '')

        max_int_len = max(len(int_part1), len(int_part2))
        int_part1 = int_part1.rjust(max_int_len, '0')
        int_part2 = int_part2.rjust(max_int_len, '0')

        max_frac_len = max(len(frac_part1), len(frac_part2))
        frac_part1 = frac_part1.ljust(max_frac_len, '0')
        frac_part2 = frac_part2.ljust(max_frac_len, '0')

        aligned_bin_str1 = f"{sign1}{int_part1}.{frac_part1}"
        aligned_bin_str2 = f"{sign2}{int_part2}.{frac_part2}"

        return aligned_bin_str1, aligned_bin_str2

    def crossover(self, parents):
        """Одноточечный кроссовер"""
        offspring = []
        while len(offspring) < len(parents):
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)

            if random.random() < self.crossover_rate:
                parent1_bin = self.float_to_binary(parent1)
                parent2_bin = self.float_to_binary(parent2)

                parent1_bin, parent2_bin = self.align_fractional_parts(parent1_bin, parent2_bin)

                possible_crossover_points = [i for i in range(len(parent1_bin)) if parent1_bin[i] != '.']
                crossover_point = random.choice(possible_crossover_points)

                child1_bin = parent1_bin[:crossover_point] + parent2_bin[crossover_point:]
                child2_bin = parent2_bin[:crossover_point] + parent1_bin[crossover_point:]

                child1 = self.binary_to_float(child1_bin)
                child2 = self.binary_to_float(child2_bin)

                child1 = max(self.interval[0], min(child1, self.interval[1]))
                child2 = max(self.interval[0], min(child2, self.interval[1]))

                offspring.append(child1)
                if len(offspring) < len(parents):
                    offspring.append(child2)
            else:
                offspring.append(parent1)
                if len(offspring) < len(parents):
                    offspring.append(parent2)

        return offspring

    def mutate(self, offspring):
        """Мутация потомков"""
        mutated_offspring = []
        for individual in offspring:
            if random.random() <= self.mutation_rate:
                individual_bin = self.float_to_binary(individual)
                dot_position = individual_bin.find('.')

                mutation_point = dot_position
                while mutation_point == dot_position:
                    mutation_point = random.randint(1, len(individual_bin) - 1)

                mutated_bin = (
                        individual_bin[:mutation_point] +
                        ('1' if individual_bin[mutation_point] == '0' else '0') +
                        individual_bin[mutation_point + 1:]
                )

                mutated_individual = self.binary_to_float(mutated_bin)
                mutated_individual = max(self.interval[0], min(mutated_individual, self.interval[1]))
            else:
                mutated_individual = individual

            mutated_offspring.append(mutated_individual)

        return mutated_offspring

    def plot_population(self, generation, plot_interval=20, show_plot=True):
        """Построение графиков"""
        if not show_plot or generation % plot_interval != 0:
            return

        x = np.linspace(self.interval[0], self.interval[1] + 1.5, 500)
        y = [self.fitness_function(xi) for xi in x]

        plt.figure(figsize=(12, 6))
        plt.plot(x, y, label="Функция")
        plt.scatter(self.population, [self.fitness_function(pt) for pt in self.population], color='red',
                    label=f"Популяция на итерации {generation}", zorder=5)
        plt.axvline(x=self.interval[0], color='k', linestyle='--',
                    label='Границы интервала' if generation == 0 else "")
        plt.axvline(x=self.interval[1], color='k', linestyle='--')
        plt.title(f"Популяция на итерации {generation}")
        plt.xlabel("x")
        plt.ylabel("fitness(x)")
        plt.legend()
        plt.show()

    def run(self, plot_interval=20, show_plot=True):
        """Основной цикл генетического алгоритма"""
        start_time = time.time()
        self.generate_population()

        for generation in range(self.num_generations):
            fitnesses = self.evaluate_population()

            best_fitness = max(fitnesses)
            best_individual = self.population[fitnesses.index(best_fitness)]
            print(
                f"Поколение {generation}: Лучшая приспособленность = {best_fitness}, Лучшая особь = {best_individual}")

            self.all_population_points.extend(self.population)

            if generation % 10 == 0:
                self.generations_points[generation] = self.population.copy()

            parents = self.select_parents(fitnesses)
            offspring = self.crossover(parents)
            self.population = self.mutate(offspring)

            # Построение графиков на каждом plot_interval поколении
            self.plot_population(generation, plot_interval, show_plot)

        end_time = time.time()
        print(f"Время работы: {end_time - start_time:.2f} секунд")