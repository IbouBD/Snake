"""
First, we will implement Deep Neural Network to solve a snake game with a 9x9 grid game.
So to create our neural network we will use tensorflow and create a first neural network then a target network similar to the first.
the two neural networks we will use to approximate the Q values
And we will create a replay buffer to store past transitions (state, action, reward, next state) and sample mini-batches from this buffer during training.
"""

import tensorflow as tf
import keras
import numpy as np
from collections import deque
import random

class DQN(tf.keras.Model):
    def __init__(self):
        super(DQN, self).__init__()

        self.conv1 = keras.layers.Conv2D(32, 2, activation='relu')
        #self.conv2 = tf.keras.layers.Conv()
        self.fc1 = tf.keras.layers.Dense(32, activation='relu')
        self.fc2 = tf.keras.layers.Dense(64, activation='relu')
        self.fc3 = tf.keras.layers.Dense(128, activation='relu')
        self.out = tf.keras.layers.Dense(4, activation='linear')
        self.flat = tf.keras.layers.Flatten()

    def call(self, inputs):
        """Forward pass."""
        x = self.flat(inputs)
        
        x = self.fc1(x)
        
        x = self.fc2(x)
        
        x = self.fc3(x)
        
        output = self.out(x)
        return output
    
class ReplayBuffer(object):
  """Experience replay buffer that samples uniformly."""
  def __init__(self, size):
    self.buffer = deque(maxlen=size)

  def add(self, state, action, reward, next_state, done):
    self.buffer.append((state, action, reward, next_state, done))

  def __len__(self):
    return len(self.buffer)

  def sample(self, num_samples):
    states, actions, rewards, next_states, dones = [], [], [], [], []
    idx = np.random.choice(len(self.buffer), num_samples)
    for i in idx:
      elem = self.buffer[i]
      state, action, reward, next_state, done = elem
      states.append(np.array(state, copy=False))
      actions.append(np.array(action, copy=False))
      rewards.append(reward)
      next_states.append(np.array(next_state, copy=False))
      dones.append(done)
    states = np.array(states)
    actions = np.array(actions)
    rewards = np.array(rewards, dtype=np.float32)
    next_states = np.array(next_states)
    dones = np.array(dones, dtype=np.float32)
    return states, actions, rewards, next_states, dones
  
action_space = [0, 1, 2, 3]
  
def select_epsilon_greedy_action(state, epsilon):
  """Take random action with probability epsilon, else take best action."""

  #result = tf.random.uniform((1,))
  if epsilon > 0.5:

    return random.randint(0, len(action_space) - 1)
  
  else:
    return tf.argmax(main_net(state)[0]).numpy() # Greedy action for state.
  

gamma = 0.99
optimizer = tf.keras.optimizers.Adam(1e-4)
mse = tf.keras.losses.MeanSquaredError()

@tf.function
def train_step(states, actions, rewards, next_states, dones, action_space):
  """Perform a training iteration on a batch of data sampled from the experience
  replay buffer."""
  # Calculate targets.
  next_qs = target_net(next_states)
  max_next_qs = tf.reduce_max(next_qs, axis=-1)
  target = rewards + (1. - dones) * gamma * max_next_qs
  with tf.GradientTape() as tape:
    qs = main_net(states)
    action_masks = tf.one_hot(actions, len(action_space))
    masked_qs = tf.reduce_sum(action_masks * qs, axis=-1)
    loss = mse(target, masked_qs)
  grads = tape.gradient(loss, main_net.trainable_variables)
  optimizer.apply_gradients(zip(grads, main_net.trainable_variables))
  return loss
    
    
    

import os
import time


def reward(snake, snake_body, apple_pos, grid_size, snake_size):
    head = snake[0]
    rewards = 0

    # Récompense pour avoir mangé la pomme
    if head == apple_pos:
        rewards += 10

    # Pénalité pour collision avec les murs ou le corps du serpent
    if  head in snake_body:
        rewards -= 10

    # Récompense pour s'approcher de la pomme
    distance_to_apple = abs(head[0] - apple_pos[0]) + abs(head[1] - apple_pos[1])
    rewards -= distance_to_apple / (grid_size * 2)  # Normalisation de la distance

    rewards = 0.4 * snake_size

    return rewards



# Define the environment
def env(max_episode, grid_size, snake_initial_size):
    apple = -1
    game_size = np.zeros((grid_size, grid_size))
    snake = [(3, 3)]
    snake_size = snake_initial_size
    best_size = snake_initial_size

    for i in range(snake_size - 1):
        snake.append((snake[-1][0], snake[-1][1] - 1))

    apple_pos = None
    while apple_pos is None:
        random_indice_x = random.randint(0, grid_size - 1)
        random_indice_y = random.randint(0, grid_size - 1)
        if (random_indice_x, random_indice_y) not in snake:
            apple_pos = (random_indice_x, random_indice_y)
            game_size[apple_pos[0]][apple_pos[1]] = apple

    colision_handler = 0
    print("Game started. Press 'ctrl c' to quit.")
    time.sleep(1)

    for e in range(max_episode):
        collision = False
        epsilon = 1
        done = False
        while True:
            try:
                epsilon = epsilon
                os.system('clear')
                state = game_size
                done = False
                state = np.reshape(state, (-1, 81))
                action = select_epsilon_greedy_action(state, epsilon)

                # Calculate the new position of the head
                if action == 0:  # Right
                    new_head = ((snake[0][0] + 1) % grid_size, snake[0][1])
                elif action == 1:  # Left
                    new_head = ((snake[0][0] - 1) % grid_size, snake[0][1])
                elif action == 2:  # Up
                    new_head = (snake[0][0], (snake[0][1] - 1) % grid_size)
                elif action == 3:  # Down
                    new_head = (snake[0][0], (snake[0][1] + 1) % grid_size)

                # Get all positions of snake except head and tail
                snake_body = snake[1:-1]

                # Check for collision with the apple
                if new_head == apple_pos:
                    colision_handler += 1
                    # Generate a new position for the apple
                    while True:
                        random_indice_x = random.randint(0, grid_size - 1)
                        random_indice_y = random.randint(0, grid_size - 1)
                        if (random_indice_x, random_indice_y) not in snake:
                            apple_pos = (random_indice_x, random_indice_y)
                            game_size[apple_pos[0]][apple_pos[1]] = apple
                            break

                    # Add the new head without removing the tail
                    snake.insert(0, new_head)
                    game_size[new_head[0]][new_head[1]] = 1
                    snake_size += 1

                # Check if the head is in the snake body
                elif snake[0] in snake_body:
                    # Collision detected
                    collision = True
                    print("beginning of the new episode")
                    done = True

                else:
                    # Normal case, update the position of the snake
                    game_size[snake[-1][0]][snake[-1][1]] = 0
                    snake.pop()
                    snake.insert(0, new_head)
                    game_size[new_head[0]][new_head[1]] = 1

                if collision == True:
                    snake_size = snake_initial_size
                    colision_handler = 0

                    # Normal case, update the position of the snake
                    game_size = np.zeros((grid_size, grid_size))
                    snake = [(3, 3)]
                    game_size[snake[0][0]][snake[0][1]] = 1

                    # Generate a new position for the apple
                    while True:
                        random_indice_x = random.randint(0, grid_size - 1)
                        random_indice_y = random.randint(0, grid_size - 1)
                        if (random_indice_x, random_indice_y) not in snake:
                            apple_pos = (random_indice_x, random_indice_y)
                            game_size[apple_pos[0]][apple_pos[1]] = apple
                            break
                    e += 1
                    
                    collision = False

                next_state = game_size
                rewards = reward(snake, snake_body, apple_pos, grid_size, snake_size)

                if done == True:
                    done = 1
                    epsilon *= 0.9999
                else:
                    done = 0

                transition = state, rewards, action, next_state, done, e
                yield transition

                current_snake_size = snake_size

                if current_snake_size > best_size:
                    best_size = current_snake_size
                


                

                print(f"Episode : {e}, Reward : {rewards}, Epsilon :{epsilon}, Best snake size : {best_size}")
                print()
                print(game_size, " ", f"Grid size : {game_size.shape}, Snake initial size : {snake_initial_size}, Snake current size : {snake_size}, Snake : {snake}, Apple : {apple_pos}")

                if e >= max_episode:
                    os.system('clear')
                    print(f"max episode reached at episode {e}")
                    print()
                    print(f"transition : {transition}")
                    break

                t = 0

                if e % 10 == 0:
                    t = 0.0
                else:
                    t = 0
                time.sleep(t)
            except KeyboardInterrupt:
                os.system('clear')
                break

# Create the main network and target network
main_net = DQN()
target_net = DQN()

# Create the replay buffer
replay_buffer = ReplayBuffer(100000)

# Train the network
for transition in env(10000, 9, 3):
    state, rewards, action, next_state, done, e = transition
    replay_buffer.add(state, action, rewards, next_state, done, )

    if len(replay_buffer) > 10000:
        states, actions, rewards, next_states, dones = replay_buffer.sample(64)
        print("training")
        train_step(states, actions, rewards, next_states, dones, action_space)

    if e % 10 == 0:
        target_net.set_weights(main_net.get_weights()) 