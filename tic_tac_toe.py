import numpy as np
import random
import yaml

from agents import HumanAgent, QLearningAgent, RandomAgent

class TicTacToe:
    def __init__(self, board=None):
        if isinstance(board, np.ndarray):
            self.board = board
        else:
            self.board = np.array([str(i) for i in range(9)])

        self.winner = None
        self.game_over = False

    def get_available_moves(self):
        moves = []
        for i, cell in enumerate(self.board):
            if cell.isdigit():
                moves.append(i)

        return moves

    def print_board(self):
        print("-------------")
        for i in range(0, 9, 3):
            print("|", self.board[i], "|", self.board[i + 1], "|", self.board[i + 2], "|")
            print("-------------")

    def is_board_full(self):
        for el in self.board:
            if el.isdigit():
                return False
        self.winner = "D"
        return True

    def check_win(self, cur_player):
        # Check rows
        for i in range(0, 9, 3):
            if all(self.board[i:i + 3] == [cur_player] * 3):
                self.winner = cur_player
                return True

        # Check columns
        for i in range(3):
            if all(self.board[i::3] == [cur_player] * 3):
                self.winner = cur_player
                return True

        # Check diagonals
        if all(self.board[::4] == [cur_player] * 3) or all(self.board[2:7:2] == [cur_player] * 3):
            self.winner = cur_player
            return True

        return False

    def move(self, move, cur_player):
        self.board[move] = cur_player

        if self.check_win(cur_player):
            self.game_over = True

        if self.is_board_full():
            self.game_over = True
    
    def clone(self):
        return TicTacToe(self.board.copy())

class Coordinator:
    act_turn=0
    winner=None
    
    def __init__(self,
                 player_x,
                 player_o,
                 num_epi=100,
                 print_board=True,
                 print_result=True,
                 log_freq=100):
        self.player_x=player_x
        self.player_o=player_o
        self.win_board={"X":0, "O":0, "D":0}
        self.num_epi=num_epi
        self.players={"X": self.player_x, "O": self.player_o}
        self.env=None
        self.print_board=print_board
        self.print_result=print_result
        self.cur_epi=0
        self.log_freq=log_freq
    
    def start(self):
        while self.cur_epi < self.num_epi:
            self.cur_player = self.players[random.choice(["O", "X"])]
            self.env = TicTacToe()

            while self.env.winner == None:
                if self.print_board:
                    print("It is ", self.cur_player.mark, "'s turn!")

                action = self.cur_player.act(self.env)
                self.env.move(action, self.cur_player.mark)

                if self.print_board:
                    self.env.print_board()
                    win_rate, draw_rate, loss_rate = self.cur_player.get_performance(self.cur_player.last_state, action)
                    print("{0} has {1}% win rate in this situation, {2}% draw rate in this situation! , {3}% loss rate in this situation".format(self.cur_player.mark, win_rate, draw_rate, loss_rate))
               
                # Signal to both players that the game ends
                if self.env.winner != None:
                    for player in self.players.values():
                        player.learn_from_result(self.env)

                    if self.env.winner == "D":
                        if self.print_result:
                            print("Draw Game")
                    elif self.env.winner == self.cur_player.mark:
                        if self.print_result:
                            print("Winner : ", self.cur_player.mark)
                    else:
                        print("Invalid Move!")

                    self.win_board[self.env.winner] += 1
                else:
                    self.switch_player()
                    self.cur_player.learn_from_result(self.env)

            self.cur_epi += 1

            if self.cur_epi % self.log_freq == 0 or \
                self.cur_epi == self.num_epi:
                print(self.player_x.mark + ": " + str(self.win_board[self.player_x.mark]) + \
                        ", "+ self.player_o.mark + ": " + str(self.win_board[self.player_o.mark]) + \
                        ", DRAW: " + str(self.win_board["D"]))

            
    def switch_player(self):
        if self.cur_player == self.player_x:
            self.cur_player=self.player_o
        else:
            self.cur_player=self.player_x

if __name__ == "__main__":
    # Open and load the YAML file
    with open("config.yaml", 'r') as yaml_file:
        config = yaml.safe_load(yaml_file)

    # Train agents vs themselves
    p1 = QLearningAgent(mark="X",
                      lr=config["q_table_model"]["lr"],
                      epsilon=config["q_table_model"]["epsilon"],
                      epsilon_decay_freq=config["q_table_model"]["epsilon_decay_freq"],
                      discount_factor=config["q_table_model"]["discount_factor"])
    p2 = QLearningAgent(mark="O",
                      lr=config["q_table_model"]["lr"],
                      epsilon=config["q_table_model"]["epsilon"],
                      epsilon_decay_freq=config["q_table_model"]["epsilon_decay_freq"],
                      discount_factor=config["q_table_model"]["discount_factor"])

    game = Coordinator(player_x=p1,
                    player_o=p2,
                    num_epi=config["q_table_model"]["num_episodes"],
                    print_board=False,
                    print_result=False,
                    log_freq=config["q_table_model"]["log_freq"])
    game.start()

    # Save Q-Table
    #p2.save_q_table("rl_q_table.txt")

    # Test agent vs random agent
    p1.epsilon=0
    p3 = RandomAgent("O")

    game = Coordinator(player_x=p1,
                    player_o=p3,
                    num_epi=100,
                    print_board=False,
                    print_result=False,
                    log_freq=10)
    game.start()

    # Test agent vs human
    p4 = HumanAgent("O")

    game = Coordinator(player_x=p1,
                    player_o=p4,
                    num_epi=10,
                    print_board=True,
                    print_result=False,
                    log_freq=10)
    game.start()