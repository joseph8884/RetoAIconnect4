import numpy as np
from connect4.connect_state import ConnectState
from connect4.policy import Policy
import math


class MCTS:
    def __init__(self, state: ConnectState, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.wins = 0
        self.available_actions = state.get_free_cols()
    
    def is_fully_expanded(self):
        total_children = len(self.children)
        total_actions = len(self.available_actions)
        if total_children == total_actions:
            return True
        else:
            return False
    
    def is_terminal(self):
        final = self.state.is_final()
        return final
    
    def best_child(self, exploration_weight):
        best_score = -999999
        best_child = None
        
        for child in self.children.values():
            if child.visits == 0:
                return child
            
            # UCB1
            win_rate = child.wins / child.visits
            parent_visits = self.visits
            child_visits = child.visits
            log_parent = math.log(parent_visits)
            exploration_term = math.sqrt(log_parent / child_visits)
            exploration_bonus = exploration_weight * exploration_term
            score = win_rate + exploration_bonus
            
            if score > best_score:
                best_score = score
                best_child = child
                
        return best_child
    
    def expand(self):
        untried_actions = []
        for a in self.available_actions:
            if a not in self.children:
                untried_actions.append(a)
        
        if len(untried_actions) == 0:
            return None
        
        action = untried_actions[0]
        next_state = self.state.transition(action)
        child_node = MCTS(next_state, parent=self, action=action)
        self.children[action] = child_node
        
        return child_node

    def _simulate(self, my_player):
        # Rollout
        sim_board = self.state.board.copy()
        sim_player = self.state.player
        current_state = ConnectState(board=sim_board, player=sim_player)
        
        max_moves = 42
        moves = 0
        
        while moves < max_moves:
            if current_state.is_final():
                break
            
            available = current_state.get_free_cols()
            if len(available) == 0:
                break
            
            action = np.random.choice(available)
            try:
                current_state = current_state.transition(action)
                moves = moves + 1
            except ValueError:
                break
        
        winner = current_state.get_winner()
        
        if winner == my_player:
            return 1.0
        elif winner == 0:
            return 0.5
        else:
            return 0.0

    
class Hello(Policy):

    def __init__(self):
        super().__init__()
        self.num_simulations = 2000
        self.exploration_weight = 1.414

    def mount(self):
        pass

    def act(self, s):
        # quien juega
        red_pieces = np.sum(s == -1)
        yellow_pieces = np.sum(s == 1)
        
        if red_pieces == yellow_pieces:
            current_player = -1
        else:
            current_player = 1
        
        initial_state = ConnectState(board=s, player=current_player)
        root = MCTS(initial_state)
        
        for _ in range(self.num_simulations):
            node = root
            
            # Selection
            while True:
                if node.is_terminal():
                    break
                if not node.is_fully_expanded():
                    break
                node = node.best_child(self.exploration_weight)
            
            # Expansion
            if not node.is_terminal():
                if not node.is_fully_expanded():
                    node = node.expand()
            
            # Simulation
            reward = node._simulate(current_player)
            
            # Backpropagation
            while node is not None:
                node.visits = node.visits + 1
                
                if node.parent is None:
                    node.wins = node.wins + reward
                elif node.parent.state.player == current_player:
                    node.wins = node.wins + reward
                else:
                    inverse_reward = 1 - reward
                    node.wins = node.wins + inverse_reward
                
                node = node.parent
        
        best_action = None
        max_visits = -1
        for action, child in root.children.items():
            if child.visits > max_visits:
                max_visits = child.visits
                best_action = action
        
        if best_action is None:
            available = initial_state.get_free_cols()
            best_action = available[0]
        
        return int(best_action)
