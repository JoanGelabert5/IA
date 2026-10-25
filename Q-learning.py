import time
import gymnasium as gym
import numpy as np

def q_learning(episodes=250000, alpha=0.1, gamma=0.9, epsilon=0.1):

    env = gym.make("FrozenLake-v1", is_slippery=True) #Provar això a fals i veure amem que hi ha

    n_states = env.observation_space.n
    n_actions = env.action_space.n

    Q = np.zeros((n_states, n_actions))

    for ep in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            # Triar A amb política epsilon-greedy
            if np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = int(np.argmax(Q[state]))

            next_state, reward, terminated, truncated, info = env.step(action)

            Q[state, action] += alpha * (reward + gamma * np.max(Q[next_state]) - Q[state, action])

            state = next_state

    env.close()
    return Q


def evaluate(Q, episodes=200):
    env = gym.make("FrozenLake-v1", is_slippery=True)

    successes = 0

    for _ in range(episodes):
        state, info = env.reset()
        terminated = False
        truncated = False

        while not (terminated or truncated):
            action = int(np.argmax(Q[state]))    # GREEDY PUR
            state, reward, terminated, truncated, info = env.step(action)
        successes += reward

    env.close()
    return successes / episodes

# -------------------------------------------------------------------
#              TEST 4 → Q-values importants
# -------------------------------------------------------------------
def print_q_values(Q):
    print("\n Q-values (resum):\n")
    for s in range(16):
        print(f"s={s} → {np.round(Q[s],3)}")

def print_policy(Q):
    print("\n Política final (greedy):\n")
    arrows = {0:"←", 1:"↓", 2:"→", 3:"↑"}  # left, down, right, up

    policy = []
    for s in range(16):
        best_a = np.argmax(Q[s])
        policy.append(arrows[best_a])

    # imprimir en format 4x4
    for i in range(4):
        print(policy[i*4:(i+1)*4])
    print()



#NOVA FUNCIÓ PER VEURE EL JOC AMB RENDER
def play_render(Q):
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode="human")
    state, info = env.reset()
    done = False

    history = []  # guardarem el camí real del render

    while not done:
        action = int(np.argmax(Q[state]))  # política greedy amb la Q aprenduda
        next_state, reward, terminated, truncated, info = env.step(action)

        # Guardar la transició real
        history.append((state, action, next_state))

        state = next_state
        done = terminated or truncated
        time.sleep(0.7)

    env.close()

    # -------------------------------
    # IMPRIMIR CAMÍ REAL FET AL RENDER
    # -------------------------------
    print("\n HISTORIAL REAL DEL RENDER:")
    for s, a, ns in history:
        print(f"Estat {s} → acció {a} → nou estat {ns}")

    print("\n RESULTAT FINAL:", "GOAL!" if reward == 1 else "CAIGUDA AL FORAT")



if __name__ == "__main__":
    print("\n--- ENTRENANT Q-LEARNING ---")
    Q = q_learning()

    print("Entrenament acabat.\n")

    accuracy = evaluate(Q, 200)
    print(f"El número d'encerts de Q-Learning ha estat {accuracy*200:.0f}")
    print(f"Q-Learning té un encert del {accuracy*100:.2f}%")
    print_q_values(Q)
    print("\n--- POLÍTICA APRESA ---")
    print_policy(Q)
    print("\n--- RENDER DEL JOC ---")
    play_render(Q)
