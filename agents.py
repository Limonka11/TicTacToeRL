import numpy as np
import random
import pickle
import torch
import torch.optim as optim
import torch.nn.functional as F

from TicTacToeRL.network import SimpleNN

class DeepQAgent:
    def __init__(self, state_dim, action_dim, epsilon_initial=1.0, epsilon_min=0.01, epsilon_decay=0.95, discount_factor=0.99, learning_rate=0.001):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.epsilon = epsilon_initial
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.discount_factor = discount_factor
        self.learning_rate = learning_rate

        # Initialize Q-networks
        self.q_network = SimpleNN(state_dim, action_dim)
        self.target_q_network = SimpleNN(state_dim, action_dim)
        self.target_q_network.load_state_dict(self.q_network.state_dict())  # Initialize target network with the same weights
        self.q_optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)

    def select_action(self, state, valid_moves):
        if np.random.rand() <= self.epsilon:
            return random.choice(valid_moves)  # Explore by taking a random action
        else:
            with torch.no_grad():
                state = torch.Tensor(state)
                q_values = self.q_network(state)
                valid_q_values = [q_values[i] for i in valid_moves]
                return int(valid_moves[np.argmax(valid_q_values)])  # Exploit by selecting the valid action with the highest Q-value

    def update_epsilon(self):
        self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)

    def train(self, state, action, reward, next_state, done):
        state = torch.Tensor(state)
        next_state = torch.Tensor(next_state)
        action = torch.tensor(action)
        reward = torch.tensor(reward)
        done = torch.tensor(int(done))

        q_values = self.q_network(state)
        target_q_values = self.target_q_network(next_state)

        q_value = q_values[action]
        target_q_value = reward + (1 - done) * self.discount_factor * torch.max(target_q_values)

        loss = F.smooth_l1_loss(q_value, target_q_value)

        self.q_optimizer.zero_grad()
        loss.backward()
        self.q_optimizer.step()

    def update_target_network(self):
        self.target_q_network.load_state_dict(self.q_network.state_dict())


class QLearningAgent:
    def __init__(self,
                 mark,
                 lr=0.3,
                 epsilon=0.2,
                 discount_factor=0.9,
                 epsilon_decay_freq=10000):
        self.mark=mark
        self.Q = {}
        self.lr = lr
        self.epsilon = epsilon
        self.discount_factor = discount_factor
        self.epsilon_decay_freq = epsilon_decay_freq
        self.last_move=None
        self.last_state=None
        self.game_count=0

        # Statistics
        self.game_history = []  # Store state-action pairs for the current game
        self.wins = {}
        self.losses = {}
        self.draws = {}

    def act(self, env):
        self.last_state=env.clone()
        available_moves=env.get_available_moves()

        if random.random() < (self.epsilon / (self.game_count // self.epsilon_decay_freq+1)):
            i=random.randrange(len(available_moves))
            return available_moves[i]
        else:
            Q_values = [self.get_Q_value(tuple(self.last_state.board), action) for action in available_moves]
            max_Q = max(Q_values)

            # Pick randomly if more than one best option
            if Q_values.count(max_Q) > 1:
                best_moves = [i for i in range(len(available_moves)) if Q_values[i] == max_Q]
                i = random.choice(best_moves)
            else:
                i = Q_values.index(max_Q)

            self.last_move = available_moves[i]
            return available_moves[i]

    def get_Q_value(self, state, action):
        if (state, action) not in self.Q:
            self.Q[(state, action)] = 1
        return self.Q[(state, action)]

    def update_Q_value(self, state, action, reward, next_state):
        state_value = self.get_Q_value(tuple(state.board), action)
        if next_state.winner is not None:
            max_next_Q = 0
        else:
            max_next_Q = max([self.get_Q_value(tuple(next_state.board), next_action) for next_action in next_state.get_available_moves()])

        self.Q[(tuple(state.board), action)] = state_value + self.lr * (reward + self.discount_factor * max_next_Q - state_value)

    def learn_from_result(self, env):
        if self.last_move is not None:
            self.game_history.append((tuple(self.last_state.board), self.last_move))

            if env.winner is None:
                self.update_Q_value(self.last_state, self.last_move, 0, env)
                pass
            else:
                if env.winner == self.mark:
                    self.update_Q_value(self.last_state, self.last_move, 1, env)
                    self.update_wins_losses_draws(self.game_history, 'win')
                elif env.winner != "D":
                    self.update_Q_value(self.last_state, self.last_move, -1, env)
                    self.update_wins_losses_draws(self.game_history, 'loss')
                else:
                    self.update_Q_value(self.last_state, self.last_move, 0, env)
                    self.update_wins_losses_draws(self.game_history, 'draw')

                self.game_history = []  # Reset game history
                self.game_count += 1
                self.last_move=None
                self.last_state=None

    def update_wins_losses_draws(self, game_history, result):
        # Update wins, losses, and draws counters for a state-action pair
        for state_key, action_key in game_history:
            if result == 'win':
                self.wins[(state_key, action_key)] = self.wins.get((state_key, action_key), 0) + 1
            elif result == 'loss':
                self.losses[(state_key, action_key)] = self.losses.get((state_key, action_key), 0) + 1
            elif result == 'draw':
                self.draws[(state_key, action_key)] = self.draws.get((state_key, action_key), 0) + 1

    def get_win_percentage(self, state, action):
        # Calculate win percentage for a state-action pair
        state_key = tuple(state.board)
        action_key = action

        total_wins = self.wins.get((state_key, action_key), 0)
        total_losses = self.losses.get((state_key, action_key), 0)
        total_draws = self.draws.get((state_key, action_key), 0)
        total_games = total_wins + total_losses + total_draws

        if total_games > 0:
            win_percentage = (total_wins / total_games) * 100
            return win_percentage
        else:
            return 0.0  # No games played, so win percentage is 0%

    def save_q_table(self, filename):
        with open(filename, 'wb') as file:
            pickle.dump(self.Q, file)

    def load_q_table(self, filename):
        with open(filename, 'rb') as file:
            self.Q = pickle.load(file)

class PlayerHuman:
    def __init__(self, turn):
        self.mark=turn
        self.last_state = None
        
    def act(self, env):
        self.last_state = env.clone()
        valid = False
        while not valid:
            try:
                act = input("Where would you like to place " + self.mark + "? Possible inputs: (0-8)?")
                act = int(act)
                if act >= 0 and act < 9 and self.last_state.board[act].isdigit():
                    valid=True
                    return act
                else:
                    print("That is not a valid move! Please try again.")
            except Exception as e:
                    print(str(act) + "is not a valid move! Please try again.")
        return act
    
    def learn_from_result(self, env):
        if env.winner is not None and \
            env.winner != self.mark and \
            env.winner != "D":
            print("Woah, I lost...")
        
    def get_win_percentage(self, state, action):
        pass

class PlayerRandom:
    def __init__(self, mark):
        self.mark = mark
        
    def act(self, env):
        acts = env.get_available_moves()
        return random.choice(acts)
    
    def learn_from_result(self, env):
        pass

    def get_win_percentage(self, state, action):
        pass