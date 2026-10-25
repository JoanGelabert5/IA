import gymnasium as gym
import numpy as np

# ---------------------------
# 1. AVALUAR UNA POLÍTICA (El "Test de qualitat")
# ---------------------------

def evaluate_policy(policy, episodes=200, render_mode=None):
    
    if render_mode:
        env = gym.make("FrozenLake-v1", is_slippery=True, render_mode=render_mode)
    else:
        env = gym.make("FrozenLake-v1", is_slippery=True)

    successes = 0

    for _ in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = int(policy[state])
            state, reward, terminated, truncated, info = env.step(action)

        # Acumulam punts només si arribem a l'objectiu
        successes += reward 

    env.close()
    # Retornam el % d'èxit. Això serà el "fitness" de l'individu.
    return successes / episodes


# ---------------------------
# 2. ALGORISME GENÈTIC
# ---------------------------

def genetic_algorithm(
    pop_size=50,
    generations=200,
    episodes_per_individual=50,
    mutation_rate=0.1,
    crossover_rate=0.7,
    tournament_size=3,
):
    #obrim l'entorn per saber quantes caselles i accions tenim
    env = gym.make("FrozenLake-v1", is_slippery=True)
    n_states = env.observation_space.n
    n_actions = env.action_space.n
    env.close()

    # Creem la primera generació totalment a l'atzar. 
    # Cada fila és un "individu" (una solució possible).
    population = np.random.randint(0, n_actions, size=(pop_size, n_states))

    # Avaluem primers intents aleatoris
    fitness = np.array([
        evaluate_policy(individual, episodes=episodes_per_individual)
        for individual in population
    ])

    for gen in range(generations):
        new_population = []

        # ELITISME
        # Agafem el millor i el passem directament a la següent ronda.
        # Així ens assegurem que si trobem una bona solució, no la perdem per culpa d'una mala mutació.
        best_idx = np.argmax(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        new_population.append(best_individual)

        # Ara omplim la resta de la nova generació tenint fills
        while len(new_population) < pop_size:
            # Selecció: Triem dos pares que siguin bons (guanyadors del torneig)
            parent1 = tournament_selection(population, fitness, tournament_size)
            parent2 = tournament_selection(population, fitness, tournament_size)

            # Creuament
            if np.random.rand() < crossover_rate:
                child1, child2 = one_point_crossover(parent1, parent2)
            else:
                # Si no hi ha creuament, els fills són clons dels pares
                child1, child2 = parent1.copy(), parent2.copy()

            # Mutació
            mutate(child1, mutation_rate, n_actions)
            mutate(child2, mutation_rate, n_actions)

            new_population.extend([child1, child2])

        # Tallam per si mos hem passat de mida i actualitzam la població
        population = np.array(new_population[:pop_size])

        # Reevaluació: Tornam a mirar quina nota treuen els nous.
        fitness = np.array([
            evaluate_policy(individual, episodes=episodes_per_individual)
            for individual in population
        ])

        print(f"Generació {gen+1}/{generations} - Millor fitness: {best_fitness*100:.2f}%")

    # Al final de tot, retornem el millor que hem trobat en tota la història
    best_idx = np.argmax(fitness)
    best_policy = population[best_idx].copy()
    best_fitness = fitness[best_idx]

    return best_policy, best_fitness


# ---------------------------
# 3. OPERADORS GENÈTICS
# ---------------------------

def tournament_selection(population, fitness, tournament_size=3):
    """
    Agafam 3 individus a l'atzar de la població i ens quedam amb el millor.
    """
    indices = np.random.choice(len(population), size=tournament_size, replace=False)
    best_idx = indices[np.argmax(fitness[indices])]
    return population[best_idx]


def one_point_crossover(parent1, parent2):
    n_genes = len(parent1)
    point = np.random.randint(1, n_genes) # Triem punt de tall
    child1 = np.concatenate([parent1[:point], parent2[point:]])
    child2 = np.concatenate([parent2[:point], parent1[point:]])
    return child1, child2


def mutate(individual, mutation_rate, n_actions):
    """
    Recorrem cada gen i el canviam per una altre acció.
    Això evita que ens quedem encallats en un òptim local.
    """
    for i in range(len(individual)):
        if np.random.rand() < mutation_rate:
            individual[i] = np.random.randint(0, n_actions)


# ---------------------------
# 4. RENDERITZAR JUGADA
# ---------------------------

def play_render(policy):
    
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    state, info = env.reset()
    done = False

    while not done:
        action = int(policy[state])
        state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

    env.close()


# ---------------------------
# 5. MAIN ESTANDAR
# ---------------------------

if __name__ == "__main__":
    print("\n--- ENTRENANT ALGORISME GENÈTIC ---")
    
    # Llançam l'entrenament.
    # mutation_rate=0.01 vol dir que hi ha un 1% de possibilitats de canviar un gen.
    best_policy, best_fitness = genetic_algorithm(
        pop_size=50,
        generations=200,
        episodes_per_individual=50,
        mutation_rate=0.01,
        crossover_rate=0.7,
    )

    print(f"\nMillor fitness (entrenament): {best_fitness*100:.2f}%")

    # Fem una prova final més rigorosa (200 episodis) per estar-ne segurs
    acc = evaluate_policy(best_policy, episodes=200)
    print(f"Taxa d'encerts final sobre 200 episodis: {acc*100:.2f}%")

    print("\n--- RENDER DEL JOC ---")
    play_render(best_policy)