import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
import math
import pickle
from pathlib import Path
from collections import defaultdict


def load_qtable(path):
    if not path.exists():
        return defaultdict(float)
    try:
        with open(path, 'rb') as f:
            data = pickle.load(f)
            return defaultdict(float, data.get('Q', {}))
    except:
        return defaultdict(float)


def encode_state_action(board, player, action):
    flat = board.flatten()
    state = f"{player}:" + ",".join(map(str, flat))
    return f"{state}|{action}"


class MCTS:
    def __init__(self, state, parent=None, action=None, depth=15, 
                 use_heuristics=True, qtable=None, q_weight=0.3):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.wins = 0
        self.available_actions = state.get_free_cols()
        self.depth = depth
        self.use_heuristics = use_heuristics
        self.qtable = qtable if qtable else defaultdict(float)
        self.q_weight = q_weight
    
    def is_fully_expanded(self):
        return len(self.children) == len(self.available_actions)
    
    def is_terminal(self):
        return self.state.is_final()
    
    def best_child(self, exploration):
        best_score = -999999
        best = None
        
        for child in self.children.values():
            if child.visits == 0:
                return child
            
            win_rate = child.wins / child.visits
            exploration_term = math.sqrt(math.log(self.visits) / child.visits)
            ucb = win_rate + exploration * exploration_term
            
            q_val = 0.0
            if len(self.qtable) > 0 and child.action is not None:
                key = encode_state_action(self.state.board, self.state.player, child.action)
                q_val = self.qtable.get(key, 0.0)
                q_val = (q_val + 1.0) / 2.0
            
            score = (1.0 - self.q_weight) * ucb + self.q_weight * q_val
            
            if score > best_score:
                best_score = score
                best = child
                
        return best
    
    def expand(self):
        untried = [a for a in self.available_actions if a not in self.children]
        
        if not untried:
            return None
        
        action = untried[0]
        new_state = self.state.transition(action)
        child = MCTS(new_state, parent=self, action=action,
                    depth=self.depth, use_heuristics=self.use_heuristics,
                    qtable=self.qtable, q_weight=self.q_weight)
        self.children[action] = child
        return child
    

    def _simulate(self, my_player):
        state = ConnectState(board=self.state.board.copy(), player=self.state.player)
        moves = 0
        
        while moves < self.depth:
            if state.is_final():
                break
            
            columns = state.get_free_cols()
            if not columns:
                break
            
            if self.use_heuristics:
                action = self.choose_with_heuristic(state, columns)
            else:
                action = self.choose_fast(state, columns)
            
            try:
                state = state.transition(action)
                moves += 1
            except:
                break
        
        winner = state.get_winner()
        if winner == my_player:
            return 1.0
        elif winner == 0:
            return 0.5
        return 0.0
    
    def choose_with_heuristic(self, state, columns):
        for col in columns:
            next_state = state.transition(col)
            if next_state.get_winner() == state.player:
                return col
        
        opponent = -state.player
        for col in columns:
            test = ConnectState(board=state.board.copy(), player=opponent)
            next_state = test.transition(col)
            if next_state.get_winner() == opponent:
                return col
        
        if 3 in columns:
            return 3
        return min(columns, key=lambda x: abs(x - 3))
    
    def choose_fast(self, state, columns):
        if 3 in columns:
            return 3
        central = [c for c in columns if 2 <= c <= 4]
        if central:
            return central[0]
        return columns[0]

class LaMejorPoliticaConQvalues(Policy):
    
    def __init__(self, simulations=180, exploration=1.0, depth=20, 
                 heuristics=True, use_qtable=True, q_weight=0.3):
        super().__init__()
        self.simulations = simulations
        self.exploration = exploration
        self.depth = depth
        self.heuristics = heuristics
        self.use_qtable = use_qtable
        self.q_weight = q_weight
        self.qtable = defaultdict(float)

    def mount(self):
        if self.use_qtable:
            path = Path(__file__).parent.parent.parent / "train" / "q_table.pkl"
            self.qtable = load_qtable(path)

    def act(self, s):
        red_pieces = np.sum(s == -1)
        yellow_pieces = np.sum(s == 1)
        player = -1 if red_pieces == yellow_pieces else 1
        state = ConnectState(board=s, player=player)
        
        quick_move = self.check_immediate(state)
        if quick_move is not None:
            return int(quick_move)
        
        if self.use_qtable and self.q_weight >= 0.99 and len(self.qtable) > 0:
            return self.choose_with_qtable(state)
        
        root = MCTS(state, depth=self.depth, use_heuristics=self.heuristics,
                    qtable=self.qtable, q_weight=self.q_weight)
        
        for _ in range(self.simulations):
            node = root
            
            while True:
                if node.is_terminal():
                    break
                if not node.is_fully_expanded():
                    break
                node = node.best_child(self.exploration)
            
            if not node.is_terminal():
                if not node.is_fully_expanded():
                    node = node.expand()
            
            reward = node._simulate(player)
            
            while node is not None:
                node.visits += 1
                if node.parent is None:
                    node.wins += reward
                elif node.parent.state.player == player:
                    node.wins += reward
                else:
                    node.wins += 1 - reward
                node = node.parent
        
        best_action = None
        max_visits = -1
        for action, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_action = action
        
        if best_action is None:
            best_action = state.get_free_cols()[0]
        
        return int(best_action)
    
    def choose_with_qtable(self, state):
        columns = state.get_free_cols()
        best = columns[0]
        best_q = -float('inf')
        
        for col in columns:
            key = encode_state_action(state.board, state.player, col)
            q = self.qtable.get(key, 0.0)
            if q > best_q:
                best_q = q
                best = col
        return int(best)
    
    def check_immediate(self, state):
        columns = state.get_free_cols()
        
        for col in columns:
            next_state = state.transition(col)
            if next_state.get_winner() == state.player:
                return col
        
        opponent = -state.player
        for col in columns:
            test = ConnectState(board=state.board.copy(), player=opponent)
            next_state = test.transition(col)
            if next_state.get_winner() == opponent:
                return col
        
        return None

