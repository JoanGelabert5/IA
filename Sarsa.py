import time
import gymnasium as gym
import numpy as np

# Política epsilon-greedy
def epsilon_greedy(Q, state, env, epsilon):
    if np.random.rand() < epsilon:
        return env.action_space.sample()
    return int(np.argmax(Q[state]))

# -------------------------------------------------------------------
#               SARSA ALGORITME
# -------------------------------------------------------------------
def sarsa(episodes=250000, alpha=0.1, gamma=0.9, epsilon=0.1):
    env = gym.make("FrozenLake-v1", is_slippery=True)

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    for ep in range(episodes):
        state, info = env.reset()
        action = epsilon_greedy(Q, state, env, epsilon)

        terminated = False
        truncated = False

        while not (terminated or truncated):
            next_state, reward, terminated, truncated, info = env.step(action)
            next_action = epsilon_greedy(Q, next_state, env, epsilon)

            Q[state, action] += alpha * (
                reward + gamma * Q[next_state, next_action] - Q[state, action]
            )

            state = next_state
            action = next_action

    env.close()
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
#              TEST 1 → Mostrar política final com a mapa
# -------------------------------------------------------------------
def print_policy(Q):
    print("\n Política final (greedy):\n")
    arrows = {0:"←", 1:"↓", 2:"→", 3:"↑"}  # left down right up

    policy = []
    for s in range(16):
        policy.append(arrows[np.argmax(Q[s])])

    for i in range(4):
        print(policy[i*4:(i+1)*4])
    print()

# -------------------------------------------------------------------
#              TEST 4 → Mostrar Q-values importants
# -------------------------------------------------------------------
def print_q_values(Q):
    print("\n Q-values (resum):\n")
    for s in range(16):
        print(f"s={s} → {np.round(Q[s],3)}")


# -------------------------------------------------------------------
#              RENDER
# -------------------------------------------------------------------
def play_render(Q):
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    state, info = env.reset()
    done = False

    history = []   # llista per guardar el camí real del render

    while not done:
        action = int(np.argmax(Q[state]))
        next_state, reward, terminated, truncated, info = env.step(action)

        # guardar pas real
        history.append((state, action, next_state))

        state = next_state
        done = terminated or truncated

        time.sleep(0.7)  # la teva velocitat original

    time.sleep(2)
    env.close()

    # ----------------------------------------------
    # IMPRIMIR HISTORIAL REAL DEL RENDER
    # ----------------------------------------------
    print("\n HISTORIAL REAL DEL RENDER:")
    for (s, a, ns) in history:
        print(f"Estat {s} → acció {a} → nou estat {ns}")

    print("\n RESULTAT FINAL:", "GOAL!" if reward == 1 else "CAIGUDA AL FORAT")


# -------------------------------------------------------------------
#              MAIN
# -------------------------------------------------------------------
if __name__ == "__main__":
    print("\n--- ENTRENANT SARSA ---")
    Q = sarsa()

    print("Entrenament acabat.\n")

    accuracy = evaluate(Q, 200)
    print(f"SARSA ha aconseguit l'objectiu {accuracy*200:.0f} vegades")
    print(f"SARSA té un encert del {accuracy*100:.2f}%\n")

    # TESTS AFEGITS
    print_policy(Q)
    print_q_values(Q)

    print("\n--- RENDER DEL JOC ---")
    play_render(Q)
