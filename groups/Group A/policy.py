import numpy as np
from connect4.policy import Policy
from typing import override
from connect4.connect_state import ConnectState
import math

class MCTS:
    def __init__(self, state: ConnectState, parent=None, action=None, rollout_depth=15, heuristics_enabled=True):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.wins = 0
        self.available_actions = state.get_free_cols()
        self.rollout_depth = rollout_depth
        self.heuristics_enabled = heuristics_enabled
    
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
        # Propagar parámetros configurables 
        child_node = MCTS(next_state, parent=self, action=action,
                          rollout_depth=self.rollout_depth,
                          heuristics_enabled=self.heuristics_enabled)
        self.children[action] = child_node
        
        return child_node
    

    def _simulate(self, my_player):
        current_state = ConnectState(board=self.state.board.copy(), player=self.state.player)
        
        # Usar profundidad configurable 
        max_moves = self.rollout_depth
        moves = 0
        
        while moves < max_moves:
            if current_state.is_final():
                break
            
            available = current_state.get_free_cols()
            if len(available) == 0:
                break
            
            # Toggle heurísticas vs selección rápida
            if self.heuristics_enabled:
                action = self._select_heuristic_action(current_state, available)
            else:
                action = self._select_fast_action(current_state, available)
            
            try:
                current_state = current_state.transition(action)
                moves += 1
            except ValueError:
                break
        
        winner = current_state.get_winner()
        
        if winner == my_player:
            return 1.0
        elif winner == 0:
            return 0.5
        else:
            return 0.0
    
    def _select_heuristic_action(self, state, available_actions):
        # Prioridad 1: Ganar inmediatamente
        for action in available_actions:
            next_state = state.transition(action)
            if next_state.get_winner() == state.player:
                return action
        
        # Prioridad 2: Bloquear victoria del oponente
        opponent = -state.player
        for action in available_actions:
            test_state = ConnectState(board=state.board.copy(), player=opponent)
            next_state = test_state.transition(action)
            if next_state.get_winner() == opponent:
                return action
        
        # Prioridad 3: Columnas centrales (mejor estrategia en Connect-4)
        center_col = 3
        if center_col in available_actions:
            return center_col
        
        # Sino, elegir la más cercana al centro
        return min(available_actions, key=lambda x: abs(x - center_col))
    
    def _select_fast_action(self, state, available_actions):
        """Versión rápida de selección para simulaciones"""
        # Solo usar heurística básica de columna central
        center_col = 3
        if center_col in available_actions:
            return center_col
        
        # Si no está disponible el centro, elegir aleatoriamente
        # pero priorizando columnas centrales
        central_actions = [col for col in available_actions if 2 <= col <= 4]
        if central_actions:
            return central_actions[0]
        
        return available_actions[0]

class Aha(Policy):
    
    def __init__(self, num_simulations=200, exploration_weight=1.414,
                 rollout_depth=15, heuristics_enabled=True):
        super().__init__()
        self.num_simulations = num_simulations
        self.exploration_weight = exploration_weight
        self.rollout_depth = rollout_depth
        self.heuristics_enabled = heuristics_enabled

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
        
        quick_move = self._check_immediate_moves(initial_state)
        if quick_move is not None:
            return int(quick_move)
        
        root = MCTS(initial_state, rollout_depth=self.rollout_depth, heuristics_enabled=self.heuristics_enabled)
        
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
                node.visits += 1
                
                if node.parent is None:
                    node.wins += reward
                elif node.parent.state.player == current_player:
                    node.wins += reward
                else:
                    inverse_reward = 1 - reward
                    node.wins += inverse_reward
                
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
    
    def _check_immediate_moves(self, state):
        """Verifica movimientos inmediatos (ganar o bloquear) para evitar MCTS innecesario"""
        available = state.get_free_cols()
        
        # 1. Verificar si podemos ganar
        for action in available:
            next_state = state.transition(action)
            if next_state.get_winner() == state.player:
                return action
        
        # 2. Verificar si necesitamos bloquear
        opponent = -state.player
        for action in available:
            test_state = ConnectState(board=state.board.copy(), player=opponent)
            next_state = test_state.transition(action)
            if next_state.get_winner() == opponent:
                return action
        
        # No hay movimientos obvios, usar MCTS
        return None

