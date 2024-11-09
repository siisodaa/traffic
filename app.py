import numpy as np
import random

# Define the environment for a single intersection
class TrafficEnvironment:
    def __init__(self):
        # Assume 2 directions: North-South (NS) and East-West (EW)
        self.traffic_density = {'NS': random.randint(1, 10), 'EW': random.randint(1, 10)}
        self.state = (self.traffic_density['NS'], self.traffic_density['EW'])
        self.light_state = 'NS'  # Initial light is green for NS
        self.steps = 0

    def reset(self):
        self.traffic_density = {'NS': random.randint(1, 10), 'EW': random.randint(1, 10)}
        self.state = (self.traffic_density['NS'], self.traffic_density['EW'])
        self.light_state = 'NS'
        self.steps = 0
        return self.state

    def step(self, action):
        # Action 0 = Stay on current light, 1 = Switch light
        if action == 1:
            self.light_state = 'EW' if self.light_state == 'NS' else 'NS'

        # Simulate traffic flow: the green light direction reduces traffic
        if self.light_state == 'NS':
            self.traffic_density['NS'] = max(0, self.traffic_density['NS'] - 3)
            self.traffic_density['EW'] += random.randint(1, 5)
        else:
            self.traffic_density['EW'] = max(0, self.traffic_density['EW'] - 3)
            self.traffic_density['NS'] += random.randint(1, 5)

        # Update state
        self.state = (self.traffic_density['NS'], self.traffic_density['EW'])
        self.steps += 1

        # Reward function: reward less traffic density
        reward = -sum(self.traffic_density.values())
        
        # Check if done (limit the steps)
        done = self.steps >= 100  # end episode after 100 steps
        return self.state, reward, done

# Q-learning agent
class TrafficAgent:
    def __init__(self, actions, alpha=0.1, gamma=0.9, epsilon=0.1):
        self.q_table = {}  # Dictionary for state-action values
        self.actions = actions  # [0, 1] for staying or switching lights
        self.alpha = alpha  # Learning rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate

    def choose_action(self, state):
        if np.random.uniform(0, 1) < self.epsilon:
            return np.random.choice(self.actions)
        else:
            return self.q_table.get(state, [0, 0]).index(max(self.q_table.get(state, [0, 0])))

    def learn(self, state, action, reward, next_state):
        current_q = self.q_table.get(state, [0, 0])[action]
        max_next_q = max(self.q_table.get(next_state, [0, 0]))
        new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)

        if state not in self.q_table:
            self.q_table[state] = [0, 0]
        self.q_table[state][action] = new_q

# Training the agent in the environment
def train_agent(episodes=1000):
    env = TrafficEnvironment()
    agent = TrafficAgent(actions=[0, 1])

    for episode in range(episodes):
        state = env.reset()
        total_reward = 0

        while True:
            action = agent.choose_action(state)
            next_state, reward, done = env.step(action)
            agent.learn(state, action, reward, next_state)
            state = next_state
            total_reward += reward

            if done:
                break

        print(f"Episode {episode + 1}, Total Reward: {total_reward}")

# Run training
if __name__ == "__main__":
    train_agent()
