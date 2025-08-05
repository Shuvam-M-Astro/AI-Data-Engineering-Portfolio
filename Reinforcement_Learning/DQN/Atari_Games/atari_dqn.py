"""
DQN (Deep Q-Network) for Atari Games
====================================

This module implements a comprehensive DQN (Deep Q-Network) system for Atari games that:
- Implements DQN with experience replay and target networks
- Supports multiple Atari environments (Pong, Breakout, Space Invaders, etc.)
- Provides comprehensive training and evaluation tools
- Implements epsilon-greedy exploration strategy
- Offers visualization and analysis of training progress

Author: AI Data Engineering Portfolio
Date: 2024
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import random
import matplotlib.pyplot as plt
import time
import os
from typing import List, Tuple, Dict, Optional
import warnings
warnings.filterwarnings('ignore')

# For Atari environments (if gym is available)
try:
    import gym
    from gym.wrappers import FrameStack, GrayScaleObservation, ResizeObservation
    ATARI_AVAILABLE = True
except ImportError:
    ATARI_AVAILABLE = False
    print("Warning: gym not available. Using simulated environment.")

# Experience replay buffer
Experience = namedtuple('Experience', ('state', 'action', 'reward', 'next_state', 'done'))

class DQNNetwork(nn.Module):
    """
    Deep Q-Network for Atari games
    """
    
    def __init__(self, input_shape, n_actions):
        """
        Initialize DQN network
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input state (channels, height, width)
        n_actions : int
            Number of possible actions
        """
        super(DQNNetwork, self).__init__()
        
        self.conv1 = nn.Conv2d(input_shape[0], 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        
        # Calculate the size of flattened features
        conv_out_size = self._get_conv_out_size(input_shape)
        
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, n_actions)
        
    def _get_conv_out_size(self, shape):
        """
        Calculate the size of flattened conv features
        
        Parameters:
        -----------
        shape : tuple
            Input shape
            
        Returns:
        --------
        int
            Size of flattened features
        """
        o = torch.zeros(1, *shape)
        o = F.relu(self.conv1(o))
        o = F.relu(self.conv2(o))
        o = F.relu(self.conv3(o))
        return int(np.prod(o.size()))
    
    def forward(self, x):
        """
        Forward pass
        
        Parameters:
        -----------
        x : torch.Tensor
            Input state
            
        Returns:
        --------
        torch.Tensor
            Q-values for all actions
        """
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)

class ReplayBuffer:
    """
    Experience replay buffer for DQN
    """
    
    def __init__(self, capacity):
        """
        Initialize replay buffer
        
        Parameters:
        -----------
        capacity : int
            Maximum number of experiences to store
        """
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        """
        Add experience to buffer
        
        Parameters:
        -----------
        *args : tuple
            Experience components (state, action, reward, next_state, done)
        """
        self.buffer.append(Experience(*args))
    
    def sample(self, batch_size):
        """
        Sample a batch of experiences
        
        Parameters:
        -----------
        batch_size : int
            Number of experiences to sample
            
        Returns:
        --------
        tuple
            Batch of experiences
        """
        batch = random.sample(self.buffer, batch_size)
        batch = Experience(*zip(*batch))
        return batch
    
    def __len__(self):
        return len(self.buffer)

class SimulatedAtariEnvironment:
    """
    Simulated Atari environment for demonstration when gym is not available
    """
    
    def __init__(self, game_name='Pong'):
        """
        Initialize simulated environment
        
        Parameters:
        -----------
        game_name : str
            Name of the game to simulate
        """
        self.game_name = game_name
        self.reset()
        
        # Define action space
        self.action_space = 6  # Standard Atari action space
        self.observation_space = (4, 84, 84)  # 4 stacked frames, 84x84
        
        # Game state
        self.score = 0
        self.lives = 3
        self.frame_count = 0
        
    def reset(self):
        """
        Reset environment
        
        Returns:
        --------
        numpy.ndarray
            Initial state
        """
        self.score = 0
        self.lives = 3
        self.frame_count = 0
        
        # Generate initial state (4 stacked frames)
        state = np.random.rand(4, 84, 84)
        return state
    
    def step(self, action):
        """
        Take action in environment
        
        Parameters:
        -----------
        action : int
            Action to take
            
        Returns:
        --------
        tuple
            (next_state, reward, done, info)
        """
        self.frame_count += 1
        
        # Simulate game logic
        reward = 0
        done = False
        
        # Random reward based on action
        if random.random() < 0.1:  # 10% chance of positive reward
            reward = 1
            self.score += 1
        
        # Random negative reward
        if random.random() < 0.05:  # 5% chance of negative reward
            reward = -1
            self.lives -= 1
        
        # Game over conditions
        if self.lives <= 0 or self.frame_count > 1000:
            done = True
        
        # Generate next state
        next_state = np.random.rand(4, 84, 84)
        
        info = {
            'score': self.score,
            'lives': self.lives,
            'frame_count': self.frame_count
        }
        
        return next_state, reward, done, info

class DQNAgent:
    """
    DQN Agent for Atari games
    """
    
    def __init__(self, state_shape, n_actions, device='cpu'):
        """
        Initialize DQN agent
        
        Parameters:
        -----------
        state_shape : tuple
            Shape of state
        n_actions : int
            Number of actions
        device : str
            Device to use ('cpu' or 'cuda')
        """
        self.device = device
        self.state_shape = state_shape
        self.n_actions = n_actions
        
        # Networks
        self.policy_net = DQNNetwork(state_shape, n_actions).to(device)
        self.target_net = DQNNetwork(state_shape, n_actions).to(device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        
        # Optimizer
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=0.00025)
        
        # Hyperparameters
        self.gamma = 0.99  # Discount factor
        self.epsilon = 1.0  # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.target_update = 1000  # Update target network every N steps
        self.batch_size = 32
        self.replay_buffer = ReplayBuffer(100000)
        
        # Training variables
        self.steps_done = 0
        self.episode_rewards = []
        self.episode_lengths = []
        
    def select_action(self, state, training=True):
        """
        Select action using epsilon-greedy policy
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        training : bool
            Whether in training mode
            
        Returns:
        --------
        int
            Selected action
        """
        if training and random.random() < self.epsilon:
            return random.randrange(self.n_actions)
        
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.argmax().item()
    
    def store_experience(self, state, action, reward, next_state, done):
        """
        Store experience in replay buffer
        
        Parameters:
        -----------
        state : numpy.ndarray
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : numpy.ndarray
            Next state
        done : bool
            Whether episode is done
        """
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train_step(self):
        """
        Perform one training step
        
        Returns:
        --------
        float
            Loss value
        """
        if len(self.replay_buffer) < self.batch_size:
            return 0.0
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        
        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch.state)).to(self.device)
        action_batch = torch.LongTensor(batch.action).to(self.device)
        reward_batch = torch.FloatTensor(batch.reward).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch.next_state)).to(self.device)
        done_batch = torch.BoolTensor(batch.done).to(self.device)
        
        # Compute current Q values
        current_q_values = self.policy_net(state_batch).gather(1, action_batch.unsqueeze(1))
        
        # Compute next Q values
        with torch.no_grad():
            next_q_values = self.target_net(next_state_batch).max(1)[0]
            target_q_values = reward_batch + (self.gamma * next_q_values * ~done_batch)
        
        # Compute loss
        loss = F.mse_loss(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()
        
        return loss.item()
    
    def update_target_network(self):
        """
        Update target network
        """
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def update_epsilon(self):
        """
        Update exploration rate
        """
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def train_episode(self, env, max_steps=1000):
        """
        Train for one episode
        
        Parameters:
        -----------
        env : object
            Environment
        max_steps : int
            Maximum steps per episode
            
        Returns:
        --------
        tuple
            (episode_reward, episode_length, loss)
        """
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        total_loss = 0
        
        for step in range(max_steps):
            # Select action
            action = self.select_action(state, training=True)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Store experience
            self.store_experience(state, action, reward, next_state, done)
            
            # Train
            loss = self.train_step()
            total_loss += loss
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            # Update target network
            if self.steps_done % self.target_update == 0:
                self.update_target_network()
            
            self.steps_done += 1
            
            if done:
                break
        
        # Update epsilon
        self.update_epsilon()
        
        return episode_reward, episode_length, total_loss / episode_length if episode_length > 0 else 0
    
    def evaluate_episode(self, env, max_steps=1000):
        """
        Evaluate for one episode (no exploration)
        
        Parameters:
        -----------
        env : object
            Environment
        max_steps : int
            Maximum steps per episode
            
        Returns:
        --------
        tuple
            (episode_reward, episode_length)
        """
        state = env.reset()
        episode_reward = 0
        episode_length = 0
        
        for step in range(max_steps):
            # Select action (no exploration)
            action = self.select_action(state, training=False)
            
            # Take action
            next_state, reward, done, info = env.step(action)
            
            # Update state
            state = next_state
            episode_reward += reward
            episode_length += 1
            
            if done:
                break
        
        return episode_reward, episode_length
    
    def save_model(self, path):
        """
        Save model
        
        Parameters:
        -----------
        path : str
            Path to save model
        """
        torch.save({
            'policy_net_state_dict': self.policy_net.state_dict(),
            'target_net_state_dict': self.target_net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epsilon': self.epsilon,
            'steps_done': self.steps_done,
            'episode_rewards': self.episode_rewards,
            'episode_lengths': self.episode_lengths
        }, path)
    
    def load_model(self, path):
        """
        Load model
        
        Parameters:
        -----------
        path : str
            Path to load model from
        """
        checkpoint = torch.load(path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint['policy_net_state_dict'])
        self.target_net.load_state_dict(checkpoint['target_net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epsilon = checkpoint['epsilon']
        self.steps_done = checkpoint['steps_done']
        self.episode_rewards = checkpoint['episode_rewards']
        self.episode_lengths = checkpoint['episode_lengths']

class AtariDQNTrainer:
    """
    Trainer for DQN on Atari games
    """
    
    def __init__(self, game_name='Pong', device='cpu'):
        """
        Initialize trainer
        
        Parameters:
        -----------
        game_name : str
            Name of the Atari game
        device : str
            Device to use
        """
        self.game_name = game_name
        self.device = device
        
        # Create environment
        if ATARI_AVAILABLE:
            self.env = self._create_atari_env(game_name)
        else:
            self.env = SimulatedAtariEnvironment(game_name)
        
        # Create agent
        state_shape = self.env.observation_space
        n_actions = self.env.action_space
        self.agent = DQNAgent(state_shape, n_actions, device)
        
        # Training history
        self.training_rewards = []
        self.training_lengths = []
        self.training_losses = []
        self.eval_rewards = []
        
    def _create_atari_env(self, game_name):
        """
        Create Atari environment
        
        Parameters:
        -----------
        game_name : str
            Name of the game
            
        Returns:
        --------
        object
            Environment
        """
        env = gym.make(f'{game_name}-v4')
        env = GrayScaleObservation(env)
        env = ResizeObservation(env, 84)
        env = FrameStack(env, 4)
        return env
    
    def train(self, n_episodes=1000, eval_interval=100, save_interval=500):
        """
        Train the agent
        
        Parameters:
        -----------
        n_episodes : int
            Number of training episodes
        eval_interval : int
            Evaluate every N episodes
        save_interval : int
            Save model every N episodes
        """
        print(f"Training DQN on {self.game_name}")
        print("=" * 50)
        
        for episode in range(n_episodes):
            # Train episode
            reward, length, loss = self.agent.train_episode(self.env)
            
            # Store results
            self.training_rewards.append(reward)
            self.training_lengths.append(length)
            self.training_losses.append(loss)
            
            # Print progress
            if (episode + 1) % 10 == 0:
                avg_reward = np.mean(self.training_rewards[-10:])
                avg_length = np.mean(self.training_lengths[-10:])
                avg_loss = np.mean(self.training_losses[-10:])
                
                print(f"Episode {episode + 1}/{n_episodes}")
                print(f"  Avg Reward: {avg_reward:.2f}")
                print(f"  Avg Length: {avg_length:.1f}")
                print(f"  Avg Loss: {avg_loss:.4f}")
                print(f"  Epsilon: {self.agent.epsilon:.3f}")
                print()
            
            # Evaluate
            if (episode + 1) % eval_interval == 0:
                eval_reward, eval_length = self.agent.evaluate_episode(self.env)
                self.eval_rewards.append(eval_reward)
                print(f"Evaluation: Reward = {eval_reward:.2f}, Length = {eval_length}")
                print()
            
            # Save model
            if (episode + 1) % save_interval == 0:
                self.agent.save_model(f'dqn_{self.game_name}_episode_{episode + 1}.pth')
        
        # Save final model
        self.agent.save_model(f'dqn_{self.game_name}_final.pth')
        print("Training completed!")
    
    def visualize_training(self):
        """
        Visualize training progress
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'DQN Training Progress - {self.game_name}', fontsize=16)
        
        # Training rewards
        axes[0, 0].plot(self.training_rewards)
        axes[0, 0].set_title('Training Rewards')
        axes[0, 0].set_xlabel('Episode')
        axes[0, 0].set_ylabel('Reward')
        axes[0, 0].grid(True, alpha=0.3)
        
        # Training lengths
        axes[0, 1].plot(self.training_lengths)
        axes[0, 1].set_title('Episode Lengths')
        axes[0, 1].set_xlabel('Episode')
        axes[0, 1].set_ylabel('Length')
        axes[0, 1].grid(True, alpha=0.3)
        
        # Training losses
        axes[1, 0].plot(self.training_losses)
        axes[1, 0].set_title('Training Losses')
        axes[1, 0].set_xlabel('Episode')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Evaluation rewards
        if self.eval_rewards:
            eval_episodes = np.arange(len(self.eval_rewards)) * 100
            axes[1, 1].plot(eval_episodes, self.eval_rewards)
            axes[1, 1].set_title('Evaluation Rewards')
            axes[1, 1].set_xlabel('Episode')
            axes[1, 1].set_ylabel('Reward')
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def evaluate_model(self, n_episodes=10):
        """
        Evaluate the trained model
        
        Parameters:
        -----------
        n_episodes : int
            Number of evaluation episodes
        """
        print(f"Evaluating DQN on {self.game_name}")
        print("=" * 50)
        
        eval_rewards = []
        eval_lengths = []
        
        for episode in range(n_episodes):
            reward, length = self.agent.evaluate_episode(self.env)
            eval_rewards.append(reward)
            eval_lengths.append(length)
            
            print(f"Episode {episode + 1}: Reward = {reward:.2f}, Length = {length}")
        
        avg_reward = np.mean(eval_rewards)
        avg_length = np.mean(eval_lengths)
        std_reward = np.std(eval_rewards)
        
        print(f"\nEvaluation Results:")
        print(f"  Average Reward: {avg_reward:.2f} ± {std_reward:.2f}")
        print(f"  Average Length: {avg_length:.1f}")
        print(f"  Best Reward: {max(eval_rewards):.2f}")
        print(f"  Worst Reward: {min(eval_rewards):.2f}")


def main():
    """
    Main function to demonstrate DQN training
    """
    print("DQN for Atari Games")
    print("=" * 50)
    
    # Create trainer
    trainer = AtariDQNTrainer(game_name='Pong', device='cpu')
    
    # Train the agent
    print("Starting training...")
    trainer.train(n_episodes=100, eval_interval=20, save_interval=50)
    
    # Visualize results
    print("Generating visualizations...")
    trainer.visualize_training()
    
    # Evaluate final model
    print("Evaluating final model...")
    trainer.evaluate_model(n_episodes=5)
    
    print("\nDQN training demonstration completed!")


if __name__ == "__main__":
    main() 