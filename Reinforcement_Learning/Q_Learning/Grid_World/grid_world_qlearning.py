"""
Q-Learning Implementation for Grid World
=======================================

This project implements Q-learning algorithm for a grid world environment
where an agent learns to navigate from start to goal while avoiding obstacles.

Features:
- Custom grid world environment
- Q-learning algorithm implementation
- Visualization of learning process
- Performance metrics and analysis
- Policy extraction and evaluation
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.animation import FuncAnimation
import time

class GridWorld:
    def __init__(self, width=8, height=8, start=(0, 0), goal=(7, 7)):
        """
        Initialize the grid world environment.
        
        Args:
            width (int): Width of the grid
            height (int): Height of the grid
            start (tuple): Starting position (x, y)
            goal (tuple): Goal position (x, y)
        """
        self.width = width
        self.height = height
        self.start = start
        self.goal = goal
        self.current_pos = start
        self.obstacles = set()
        
        # Actions: 0=up, 1=right, 2=down, 3=left
        self.actions = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        self.action_names = ['Up', 'Right', 'Down', 'Left']
        
        # Rewards
        self.rewards = {
            'goal': 100,
            'obstacle': -100,
            'step': -1,
            'out_of_bounds': -50
        }
        
    def add_obstacles(self, obstacles):
        """Add obstacles to the grid."""
        self.obstacles.update(obstacles)
    
    def reset(self):
        """Reset the environment to starting position."""
        self.current_pos = self.start
        return self.current_pos
    
    def step(self, action):
        """
        Take a step in the environment.
        
        Args:
            action (int): Action to take (0-3)
        
        Returns:
            tuple: (next_state, reward, done, info)
        """
        # Calculate new position
        dx, dy = self.actions[action]
        new_x = self.current_pos[0] + dx
        new_y = self.current_pos[1] + dy
        
        # Check bounds
        if (new_x < 0 or new_x >= self.height or 
            new_y < 0 or new_y >= self.width):
            return self.current_pos, self.rewards['out_of_bounds'], False, {}
        
        new_pos = (new_x, new_y)
        
        # Check if hit obstacle
        if new_pos in self.obstacles:
            return self.current_pos, self.rewards['obstacle'], False, {}
        
        # Update position
        self.current_pos = new_pos
        
        # Check if reached goal
        if new_pos == self.goal:
            return new_pos, self.rewards['goal'], True, {}
        
        # Regular step
        return new_pos, self.rewards['step'], False, {}
    
    def get_valid_actions(self, state):
        """Get valid actions for a given state."""
        valid_actions = []
        for action in range(4):
            dx, dy = self.actions[action]
            new_x = state[0] + dx
            new_y = state[1] + dy
            
            # Check bounds
            if (new_x < 0 or new_x >= self.height or 
                new_y < 0 or new_y >= self.width):
                continue
            
            new_pos = (new_x, new_y)
            
            # Check obstacle
            if new_pos in self.obstacles:
                continue
            
            valid_actions.append(action)
        
        return valid_actions
    
    def render(self, q_table=None, policy=None):
        """Render the grid world."""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create grid
        grid = np.zeros((self.height, self.width))
        
        # Mark obstacles
        for obs in self.obstacles:
            grid[obs[0], obs[1]] = -1
        
        # Mark start and goal
        grid[self.start[0], self.start[1]] = 2
        grid[self.goal[0], self.goal[1]] = 3
        
        # Mark current position
        grid[self.current_pos[0], self.current_pos[1]] = 4
        
        # Create heatmap
        sns.heatmap(grid, annot=True, fmt='.0f', cmap='RdYlBu_r', 
                   cbar=False, square=True, ax=ax)
        
        # Add text annotations
        for i in range(self.height):
            for j in range(self.width):
                if (i, j) == self.start:
                    ax.text(j + 0.5, i + 0.5, 'START', ha='center', va='center', 
                           fontweight='bold', color='white')
                elif (i, j) == self.goal:
                    ax.text(j + 0.5, i + 0.5, 'GOAL', ha='center', va='center', 
                           fontweight='bold', color='white')
                elif (i, j) in self.obstacles:
                    ax.text(j + 0.5, i + 0.5, 'X', ha='center', va='center', 
                           fontweight='bold', color='white')
                elif (i, j) == self.current_pos:
                    ax.text(j + 0.5, i + 0.5, 'AGENT', ha='center', va='center', 
                           fontweight='bold', color='white')
        
        # Add Q-values or policy arrows if provided
        if q_table is not None or policy is not None:
            for i in range(self.height):
                for j in range(self.width):
                    if (i, j) not in self.obstacles and (i, j) != self.goal:
                        if policy is not None and (i, j) in policy:
                            action = policy[(i, j)]
                            dx, dy = self.actions[action]
                            ax.arrow(j + 0.5, i + 0.5, dy * 0.3, dx * 0.3, 
                                   head_width=0.1, head_length=0.1, fc='black', ec='black')
        
        ax.set_title('Grid World Environment')
        plt.tight_layout()
        plt.show()

class QLearningAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.95, epsilon=0.1):
        """
        Initialize Q-learning agent.
        
        Args:
            env: GridWorld environment
            learning_rate (float): Learning rate (alpha)
            discount_factor (float): Discount factor (gamma)
            epsilon (float): Exploration rate
        """
        self.env = env
        self.lr = learning_rate
        self.gamma = discount_factor
        self.epsilon = epsilon
        
        # Initialize Q-table
        self.q_table = {}
        self.initialize_q_table()
        
        # Training history
        self.episode_rewards = []
        self.episode_lengths = []
        self.epsilon_history = []
        
    def initialize_q_table(self):
        """Initialize Q-table with zeros."""
        for i in range(self.env.height):
            for j in range(self.env.width):
                state = (i, j)
                if state not in self.env.obstacles:
                    self.q_table[state] = {}
                    valid_actions = self.env.get_valid_actions(state)
                    for action in valid_actions:
                        self.q_table[state][action] = 0.0
    
    def get_action(self, state, training=True):
        """
        Choose action using epsilon-greedy policy.
        
        Args:
            state: Current state
            training (bool): Whether in training mode
        """
        valid_actions = self.env.get_valid_actions(state)
        
        if not valid_actions:
            return None
        
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return np.random.choice(valid_actions)
        else:
            # Exploitation: best action
            q_values = [self.q_table[state].get(action, 0) for action in valid_actions]
            max_q = max(q_values)
            best_actions = [action for action, q in zip(valid_actions, q_values) if q == max_q]
            return np.random.choice(best_actions)
    
    def update_q_value(self, state, action, reward, next_state):
        """
        Update Q-value using Q-learning update rule.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
        """
        if state not in self.q_table or action not in self.q_table[state]:
            return
        
        # Get current Q-value
        current_q = self.q_table[state][action]
        
        # Get max Q-value for next state
        if next_state in self.q_table:
            next_q_values = list(self.q_table[next_state].values())
            max_next_q = max(next_q_values) if next_q_values else 0
        else:
            max_next_q = 0
        
        # Q-learning update rule
        new_q = current_q + self.lr * (reward + self.gamma * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
    def train(self, episodes=1000, max_steps=100):
        """
        Train the agent using Q-learning.
        
        Args:
            episodes (int): Number of training episodes
            max_steps (int): Maximum steps per episode
        """
        print(f"Training for {episodes} episodes...")
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while steps < max_steps:
                # Choose action
                action = self.get_action(state, training=True)
                if action is None:
                    break
                
                # Take action
                next_state, reward, done, _ = self.env.step(action)
                
                # Update Q-value
                self.update_q_value(state, action, reward, next_state)
                
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    break
            
            # Record episode statistics
            self.episode_rewards.append(total_reward)
            self.episode_lengths.append(steps)
            self.epsilon_history.append(self.epsilon)
            
            # Decay epsilon
            if episode % 100 == 0 and episode > 0:
                self.epsilon = max(0.01, self.epsilon * 0.95)
            
            # Print progress
            if episode % 100 == 0:
                avg_reward = np.mean(self.episode_rewards[-100:])
                print(f"Episode {episode}: Avg Reward = {avg_reward:.2f}, Epsilon = {self.epsilon:.3f}")
    
    def get_policy(self):
        """Extract policy from Q-table."""
        policy = {}
        for state in self.q_table:
            if self.q_table[state]:
                best_action = max(self.q_table[state], key=self.q_table[state].get)
                policy[state] = best_action
        return policy
    
    def evaluate_policy(self, episodes=100, max_steps=100):
        """
        Evaluate the learned policy.
        
        Args:
            episodes (int): Number of evaluation episodes
            max_steps (int): Maximum steps per episode
        """
        success_count = 0
        total_rewards = []
        episode_lengths = []
        
        for episode in range(episodes):
            state = self.env.reset()
            total_reward = 0
            steps = 0
            
            while steps < max_steps:
                action = self.get_action(state, training=False)
                if action is None:
                    break
                
                next_state, reward, done, _ = self.env.step(action)
                total_reward += reward
                state = next_state
                steps += 1
                
                if done:
                    if state == self.env.goal:
                        success_count += 1
                    break
            
            total_rewards.append(total_reward)
            episode_lengths.append(steps)
        
        success_rate = success_count / episodes
        avg_reward = np.mean(total_rewards)
        avg_length = np.mean(episode_lengths)
        
        print(f"\nPolicy Evaluation Results:")
        print(f"Success Rate: {success_rate:.2%}")
        print(f"Average Reward: {avg_reward:.2f}")
        print(f"Average Episode Length: {avg_length:.2f}")
        
        return success_rate, avg_reward, avg_length
    
    def plot_training_history(self):
        """Plot training history."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Plot 1: Episode rewards
        axes[0, 0].plot(self.episode_rewards)
        axes[0, 0].set_title('Episode Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True)
        
        # Plot 2: Moving average rewards
        window = 100
        moving_avg = np.convolve(self.episode_rewards, np.ones(window)/window, mode='valid')
        axes[0, 1].plot(moving_avg)
        axes[0, 1].set_title(f'Moving Average Rewards (window={window})')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Average Reward')
        axes[0, 1].grid(True)
        
        # Plot 3: Episode lengths
        axes[1, 0].plot(self.episode_lengths)
        axes[1, 0].set_title('Episode Lengths')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Steps')
        axes[1, 0].grid(True)
        
        # Plot 4: Epsilon decay
        axes[1, 1].plot(self.epsilon_history)
        axes[1, 1].set_title('Epsilon Decay')
        axes[1, 1].set_xlabel('Episode')
        axes[1, 1].set_ylabel('Epsilon')
        axes[1, 1].grid(True)
        
        plt.tight_layout()
        plt.show()

def main():
    """Main function to run Q-learning on grid world."""
    # Create environment
    env = GridWorld(width=8, height=8, start=(0, 0), goal=(7, 7))
    
    # Add obstacles
    obstacles = [
        (1, 1), (1, 2), (1, 3), (1, 4),
        (3, 3), (3, 4), (3, 5),
        (5, 1), (5, 2), (5, 3), (5, 4), (5, 5),
        (6, 6)
    ]
    env.add_obstacles(obstacles)
    
    # Render initial environment
    print("Grid World Environment:")
    env.render()
    
    # Create and train agent
    agent = QLearningAgent(env, learning_rate=0.1, discount_factor=0.95, epsilon=0.1)
    
    # Train the agent
    agent.train(episodes=1000, max_steps=100)
    
    # Plot training history
    agent.plot_training_history()
    
    # Get learned policy
    policy = agent.get_policy()
    
    # Render environment with policy
    print("\nLearned Policy:")
    env.render(policy=policy)
    
    # Evaluate policy
    success_rate, avg_reward, avg_length = agent.evaluate_policy(episodes=100)
    
    # Demonstrate optimal path
    print("\nDemonstrating optimal path:")
    state = env.reset()
    path = [state]
    
    for step in range(50):
        action = agent.get_action(state, training=False)
        if action is None:
            break
        
        next_state, reward, done, _ = env.step(action)
        path.append(next_state)
        state = next_state
        
        if done:
            break
    
    print(f"Path: {path}")
    print(f"Final position: {state}")
    print(f"Reached goal: {state == env.goal}")

if __name__ == "__main__":
    main() 