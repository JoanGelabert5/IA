import time
import gymnasium as gym
import numpy as np
from collections import defaultdict

# Política epsilon-greedy
def epsilon_greedy(Q, state, env, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return int(np.argmax(Q[state]))

# -------------------------------------------------------------------
#               MONTE CARLO CONTROL (FIRST VISIT)
# -------------------------------------------------------------------
def montecarlo(episodes=250000, gamma=0.9, epsilon=0.1):
    env = gym.make("FrozenLake-v1", is_slippery=True)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = defaultdict(lambda: np.zeros(n_actions))
    returns_sum = defaultdict(lambda: np.zeros(n_actions))
    returns_count = defaultdict(lambda: np.zeros(n_actions))

    print("Monte Carlo entrenant...")

    for ep in range(episodes):
        state, info = env.reset()
        episode = []
        terminated = False
        truncated = False

        # Generar episodi complet
        while not (terminated or truncated):
            action = epsilon_greedy(Q, state, env, epsilon)
            next_state, reward, terminated, truncated, info = env.step(action)
            episode.append((state, action, reward))
            state = next_state

        # First-visit MonteCarlo
        G = 0
        visited = set()

        for t in reversed(range(len(episode))):
            s, a, r = episode[t]
            G = gamma * G + r

            if (s, a) not in visited:
                visited.add((s, a))
                returns_sum[s][a] += G
                returns_count[s][a] += 1
                Q[s][a] = returns_sum[s][a] / returns_count[s][a]

    env.close()
    print("Entrenament Monte Carlo completat!")
    return Q

# -------------------------------------------------------------------
#              AVALUACIÓ GREEDY PURA
# -------------------------------------------------------------------
def evaluate(Q, episodes=200):
    env = gym.make("FrozenLake-v1", is_slippery=True)

    successes = 0
    for _ in range(episodes):
        state, info = env.reset()
        done = False

        while not done:
            action = int(np.argmax(Q[state]))
            state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        
        successes += reward

    env.close()
    return successes / episodes

# -------------------------------------------------------------------
#              TEST 1 → Política final com a mapa
# -------------------------------------------------------------------
def print_policy(Q):
    print("\n Política final (greedy):\n")
    arrows = {0:"←", 1:"↓", 2:"→", 3:"↑"}

    policy = []
    for s in range(16):
        policy.append(arrows[np.argmax(Q[s])])

    for i in range(4):
        print(policy[i*4:(i+1)*4])
    print()

# -------------------------------------------------------------------
#              TEST 4 → Q-values importants
# -------------------------------------------------------------------
def print_q_values(Q):
    print("\n Q-values (resum):\n")
    for s in range(16):
        print(f"s={s} → {np.round(Q[s],3)}")

# -------------------------------------------------------------------
#              RENDER
# -------------------------------------------------------------------
# --------------------------------------------------------
#   Funció auxiliar → moviment ideal sense lliscament
# --------------------------------------------------------
def deterministic_next(s, a):
    row = s // 4
    col = s % 4

    if a == 0:   # esquerra
        col = max(0, col - 1)
    elif a == 1: # avall
        row = min(3, row + 1)
    elif a == 2: # dreta
        col = min(3, col + 1)
    elif a == 3: # amunt
        row = max(0, row - 1)

    return row * 4 + col

# --------------------------------------------------------
#   Funció auxiliar → detectar lliscament
# --------------------------------------------------------
def slipped(s, a, ns):
    ideal = deterministic_next(s, a)

    if ns == ideal:
        return False  # no ha lliscat

    # Accions laterals que provoca el lliscament
    lateral_moves = {
        0: [3, 1],   # esquerra → lliscar amunt o avall
        1: [0, 2],   # avall → lliscar esquerra o dreta
        2: [3, 1],   # dreta → lliscar amunt o avall
        3: [0, 2],   # amunt → lliscar esquerra o dreta
    }

    for lat in lateral_moves[a]:
        if deterministic_next(s, lat) == ns:
            return True  # lliscament legítim

    return "strange"  # moviment inesperat

# -------------------------------------------------------------------
#              RENDER (AMB HISTORIAL I DETECCIÓ DE LLISCAMENT)
# -------------------------------------------------------------------
def play_render(Q):
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    state, info = env.reset()
    done = False

    history = []  

    while not done:
        action = int(np.argmax(Q[state]))
        next_state, reward, terminated, truncated, info = env.step(action)

        history.append((state, action, next_state))

        state = next_state
        done = terminated or truncated
        time.sleep(2)

    env.close()
    time.sleep(1)

    # -------------------------------
    # IMPRIMIR HISTORIAL AMB LLISCAMENT
    # -------------------------------
    print("\n HISTORIAL REAL DEL RENDER:")
    for (s, a, ns) in history:
        ideal = deterministic_next(s, a)
        status = slipped(s, a, ns)

        if status is False:
            print(f" Estat {s} → acció {a} → nou estat {ns} (sense lliscament")
        elif status is True:
            print(f" Estat {s} → acció {a} → nou estat {ns} (LLISCAMENT")
        else:
            print(f" Estat {s} → acció {a} → nou estat {ns}")

    print("\n RESULTAT FINAL:", "GOAL!" if reward == 1 else "CAIGUDA AL FORAT")




# -------------------------------------------------------------------
#              MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- ENTRENANT MONTE CARLO ---")
    Q = montecarlo()

    print("\nEntrenament acabat.\n")

    accuracy = evaluate(Q, 200)
    print(f"Monte Carlo ha aconseguit l'objectiu {accuracy*200:.0f} vegades")
    print(f"Monte Carlo té un encert del {accuracy*100:.2f}%\n")

    # TESTS
    print_policy(Q)
    print_q_values(Q)

    print("\n--- RENDER DEL JOC ---")
    play_render(Q)
