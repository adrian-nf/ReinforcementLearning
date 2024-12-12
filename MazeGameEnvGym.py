import numpy as np
import matplotlib.pyplot as plt
import copy
from matplotlib.animation import FuncAnimation
from matplotlib import rc
import gymnasium as gym
from gymnasium import spaces

rc('animation', html='jshtml')
class MazeGameEnv(gym.Env):
    def __init__(self, board=[['😊', ' ', '😺'],[' ', ' ', ' '],['😺', ' ', '😍']], actions=["up", "down", "left", "right"], actions_moves=[(0,-1),(0,1),(-1,0),(1,0)], values={'': -1, ' ': -1, '😍': 100, '😺': 20, '😊':0}, player="😊", goal="😍", render_mode=None):
        super(MazeGameEnv, self).__init__()
        self.board = board
        self.actions = actions
        self.actions_moves = actions_moves
        self.values = values
        self.player = player
        self.goal = goal
        self.goal_position = self.get_pos(self.goal)
        self.board_history = []
        self.board_history.append(copy.deepcopy(self.board))
        self.original_board = [row.copy() for row in self.board]
        self.board_size = (len(self.board), len(self.board[0]))

        self.action_space = spaces.Discrete(len(self.actions))
        self.observation_space = spaces.Box(low=-10, high=100, shape=self.board_size, dtype=np.int32)

    def get_numeric_board(self):
        return np.array(
            [[self.values[cell] for cell in row] for row in self.board],
            dtype=np.float32
        )
    
    def get_pos(self, value):
        for i in range(len(self.board)):
            for j in range(len(self.board[0])):
                if value in self.board[i][j]:
                    return (i, j)

    def reset(self, seed=None, options=None):
        self.board = [row.copy() for row in self.original_board]
        self.goal_position = self.get_pos(self.goal)
        self.board_history = []
        self.board_history.append(copy.deepcopy(self.board))
        return self.get_numeric_board(), {}

    def is_finish(self):
        pass

    def is_valid_move(self, state, action):
        x, y = state
        move_x, move_y = self.actions_moves[action]
        new_x = x + move_x
        new_y = y + move_y
        
        if new_x < 0 or new_x >= len(self.board) or new_y < 0 or new_y >= len(self.board):
            return False
        return True
    
    def move(self, state, new_state):
        board = [row.copy() for row in self.original_board]
        x, y = state
        new_x, new_y = new_state
        self.board[x][y] = ' '
        self.board[new_x][new_y] = self.player
    
        return board

    def calculate_reward(self, state):
        return self.values[self.board[state[0]][state[1]]]
    
    def step(self, action):
        truncated = False # Truncated not implemented always False
        state = self.get_pos(self.player)
        if not self.is_valid_move(state, action):
            #print("Not valid move")
            return self.get_numeric_board(), -10, truncated, truncated,{}
        
        x, y = state
        move_x, move_y = self.actions_moves[action]
        new_x = x + move_x
        new_y = y + move_y
        reward = self.calculate_reward((new_x, new_y))
        self.board[x][y] = ' '
        self.board[new_x][new_y] = self.player
        
        done = False
        if self.get_pos(self.player)==self.goal_position:
            done = True

        self.board_history.append(copy.deepcopy(self.board))
        return self.get_numeric_board(), reward, done, truncated,{}

    def render(self):
        fig, ax = plt.subplots()
        ax.set_xticks(np.arange(0, len(self.board) + 1, 1))
        ax.set_yticks(np.arange(0, len(self.board) + 1, 1))
        ax.grid(True, color='black')

        # Set limits and reverse y-axis to have (0,0) in top-left
        ax.set_xlim(0, len(self.board))
        ax.set_ylim(0, len(self.board))
        ax.invert_yaxis()

        # Initialize a list of text objects for each cell
        text_objects = []
        for i in range(len(self.board)):
            row = []
            for j in range(len(self.board)):
                text = ax.text(j + 0.5, i + 0.5, '', ha='center', va='center', fontsize=50)
                row.append(text)
            text_objects.append(row)

        # Function to update the board for each frame of the animation
        def update(frame):
            board = self.board_history[frame]
            for i in range(len(self.board)):
                for j in range(len(self.board)):
                    text_objects[i][j].set_text(board[i][j])
            return [item for sublist in text_objects for item in sublist]

        # Create the animation
        ani = FuncAnimation(fig, update, frames=len(self.board_history), interval=500, blit=True)
        plt.close(fig)
        return ani

    def close(self):
        plt.close()