# Set device "mps" (mac silicon), "cpu" (cpu) or "cuda" (gpu)
device = "mps"

#############################################################

import torch
import numpy as np
from torch import nn
from src.model import Model
from src.utils import save_checkpoint, write_to_csv

device = torch.device(device) 

class Agent:
    def __init__(self, env, save_directory, gamma, lamda, epilson, epochs, divisor, interval, batch_size, a_lr, c_lr, show, verbose):
        self.env = env
        self.show = show
        # Set up arrays for storing data
        self.rewards = []
        self.total_rewards = []
        self.episode = 0
        # Set hyper parameters
        self.gamma, self.lamda, self.epilson, self.epochs, self.interval, self.batch_size, self.save_directory, self.verbose = gamma, lamda, epilson, epochs, interval, batch_size, save_directory, verbose
        self.mini_batch_size = self.batch_size // divisor
        self.obs = np.array(self.env.reset()[0])

        # Create current policy and clone it to old policy
        self.model = Model(env).to(device)
        self.optimizer = torch.optim.Adam([
            {'params': self.model.actor.parameters(), 'lr': a_lr},
            {'params': self.model.critic.parameters(), 'lr': c_lr}
        ], eps=1e-4)
        self.model_old = Model(env).to(device)
        self.model_old.load_state_dict(self.model.state_dict())
        self.mse_loss = nn.MSELoss()

    def sample(self):
        # Initialize empty numpy arrays for sampling
        rewards = np.zeros(self.batch_size, dtype=np.float32)
        actions = np.zeros(self.batch_size, dtype=np.float32)
        log_probs = np.zeros(self.batch_size, dtype=np.float32)
        values = np.zeros(self.batch_size, dtype=np.float32)
        done = np.zeros(self.batch_size, dtype=np.bool_)
        obs = np.zeros((self.batch_size, 4, 84, 84), dtype=np.float32)
        for time in range(self.batch_size):
            # Sample actions
            with torch.no_grad():
                obs[time] = self.obs
                # Forward pass through networks
                policy, value = self.model_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
                # Get value at time
                values[time] = value.cpu().numpy()
                # Get action probability distribution
                action = policy.sample()
                # Get action from policy
                actions[time] = action.cpu().numpy()
                # Get log probability
                log_probs[time] = policy.log_prob(action).cpu().numpy()
            self.obs, rewards[time], done[time], _, _ = self.env.step(actions[time])
            self.obs = self.obs.__array__()

            if self.show:
                self.env.render()
            
            if self.verbose:
                print("Episode: {}, Step: {}, reward: {}".format(self.episode, time, rewards[time]))
            
            self.rewards.append(rewards[time])
            if done[time]:
                # If done increment episodes and reset environment
                self.episode += 1
                self.total_rewards.append(np.sum(self.rewards))
                self.rewards = []
                self.env.reset()
                if self.episode % self.interval == 0:
                    print('Episode: {}, average reward: {}'.format(self.episode, np.mean(self.total_rewards[-10:])))
                    write_to_csv(self.save_directory, 'training_data.csv', self.episode, np.mean(self.total_rewards[-10:]))
                    save_checkpoint(self.model_old, self.episode, self.save_directory)
        # Get advantages
        gae, advantages = self.calculate_advantages(done, rewards, values)
        return {
            'obs': torch.tensor(obs.reshape(obs.shape[0], *obs.shape[1:]), dtype=torch.float32, device=device),
            'actions': torch.tensor(actions, device=device),
            'values': torch.tensor(values, device=device),
            'log_probs': torch.tensor(log_probs, device=device),
            'advantages': torch.tensor(advantages, device=device, dtype=torch.float32),
            'gae': torch.tensor(gae, device=device, dtype=torch.float32)
        }

    def calculate_advantages(self, done, rewards, values):
        _, value = self.model_old(torch.tensor(self.obs, dtype=torch.float32, device=device).unsqueeze(0))
        # Add current prediction from model to value predictions
        value = value.cpu().data.numpy()
        values = np.append(values, value)
        returns = []
        # Build up GAE
        gae = 0
        # Loop in reverse to more efficently build up value
        for i in reversed(range(len(rewards))):
            # Ensures that steps that are out of episode are not counted
            mask = 1.0 - done[i]
            # Calculate delta for TD Error
            delta = rewards[i] + self.gamma * values[i + 1] * mask - values[i]
            # Calculate GAE for step using delta, gamma, lambda, mask and current gae as per formula
            gae = delta + self.gamma * self.lamda * mask * gae
            returns.insert(0, gae + values[i])
        # Calculate advantages using GAE and V
        adv = np.array(returns) - values[:-1]
        # Return GAEs for each step and normalised advantage
        return returns, (adv - np.mean(adv)) / (np.std(adv) + 1e-8)

    def train(self, samples):
        # Randomise the indexes using a random permutation
        indexes = torch.randperm(self.batch_size)
        for start in range(0, self.batch_size, self.mini_batch_size):
            # Get indexes for minibatch
            mini_batch_indexes = indexes[start: start + self.mini_batch_size]
            mini_batch = {}
            # Add episodes to mini batch
            for i, v in samples.items():
                mini_batch[i] = v[mini_batch_indexes]
            for _ in range(self.epochs):
                # Calculate loss and train the model
                loss = self.calculate_loss(self.epilson, mini_batch)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            # Load new model into old model
            self.model_old.load_state_dict(self.model.state_dict())

    def calculate_loss(self, epilson, samples):
        gae = samples['gae']
        sampled_advantages = samples['advantages']
        policy, value = self.model(samples['obs'])
        ratio = torch.exp(policy.log_prob(samples['actions']) - samples['log_probs'])
        clipped_ratio = ratio.clamp(min=1.0 - epilson, max=1.0 + epilson)
        # Get the clipped 
        clipped_objective = torch.min(ratio * sampled_advantages, clipped_ratio * sampled_advantages)
        # Calculate entropy
        s = policy.entropy()
        # Value function loss
        vf_loss = self.mse_loss(value, gae)
        loss = -clipped_objective + 0.5 * vf_loss - 0.01 * s
        return loss.mean()