import numpy as np
from connect4.policy import Policy
from connect4.connect_state import ConnectState
import math
import pickle
from pathlib import Path
from collections import defaultdict


def cargar_qtable(ruta):
    if not ruta.exists():
        return defaultdict(float)
    try:
        with open(ruta, 'rb') as f:
            data = pickle.load(f)
            return defaultdict(float, data.get('Q', {}))
    except:
        return defaultdict(float)


def codificar_estado_accion(tablero, jugador, accion):
    flat = tablero.flatten()
    estado = f"{jugador}:" + ",".join(map(str, flat))
    return f"{estado}|{accion}"


class MCTS:
    def __init__(self, state, parent=None, action=None, profundidad=15, 
                 usar_heuristicas=True, qtable=None, peso_q=0.3):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = {}
        self.visits = 0
        self.wins = 0
        self.acciones_disponibles = state.get_free_cols()
        self.profundidad = profundidad
        self.usar_heuristicas = usar_heuristicas
        self.qtable = qtable if qtable else defaultdict(float)
        self.peso_q = peso_q
    
    def esta_expandido(self):
        return len(self.children) == len(self.acciones_disponibles)
    
    def es_terminal(self):
        return self.state.is_final()
    
    def mejor_hijo(self, exploracion):
        mejor_puntaje = -999999
        mejor = None
        
        for hijo in self.children.values():
            if hijo.visits == 0:
                return hijo
            
            tasa_victoria = hijo.wins / hijo.visits
            termino_exploracion = math.sqrt(math.log(self.visits) / hijo.visits)
            ucb = tasa_victoria + exploracion * termino_exploracion
            
            q_val = 0.0
            if len(self.qtable) > 0 and hijo.action is not None:
                clave = codificar_estado_accion(self.state.board, self.state.player, hijo.action)
                q_val = self.qtable.get(clave, 0.0)
                q_val = (q_val + 1.0) / 2.0
            
            puntaje = (1.0 - self.peso_q) * ucb + self.peso_q * q_val
            
            if puntaje > mejor_puntaje:
                mejor_puntaje = puntaje
                mejor = hijo
                
        return mejor
    
    def expandir(self):
        no_probadas = [a for a in self.acciones_disponibles if a not in self.children]
        
        if not no_probadas:
            return None
        
        accion = no_probadas[0]
        nuevo_estado = self.state.transition(accion)
        hijo = MCTS(nuevo_estado, parent=self, action=accion,
                    profundidad=self.profundidad, usar_heuristicas=self.usar_heuristicas,
                    qtable=self.qtable, peso_q=self.peso_q)
        self.children[accion] = hijo
        return hijo
    

    def simular(self, mi_jugador):
        estado = ConnectState(board=self.state.board.copy(), player=self.state.player)
        movimientos = 0
        
        while movimientos < self.profundidad:
            if estado.is_final():
                break
            
            columnas = estado.get_free_cols()
            if not columnas:
                break
            
            if self.usar_heuristicas:
                accion = self.elegir_con_heuristica(estado, columnas)
            else:
                accion = self.elegir_rapido(estado, columnas)
            
            try:
                estado = estado.transition(accion)
                movimientos += 1
            except:
                break
        
        ganador = estado.get_winner()
        if ganador == mi_jugador:
            return 1.0
        elif ganador == 0:
            return 0.5
        return 0.0
    
    def elegir_con_heuristica(self, estado, columnas):
        for col in columnas:
            sig = estado.transition(col)
            if sig.get_winner() == estado.player:
                return col
        
        oponente = -estado.player
        for col in columnas:
            test = ConnectState(board=estado.board.copy(), player=oponente)
            sig = test.transition(col)
            if sig.get_winner() == oponente:
                return col
        
        if 3 in columnas:
            return 3
        return min(columnas, key=lambda x: abs(x - 3))
    
    def elegir_rapido(self, estado, columnas):
        if 3 in columnas:
            return 3
        centrales = [c for c in columnas if 2 <= c <= 4]
        if centrales:
            return centrales[0]
        return columnas[0]

class LaMejorPoliticaConQvalues(Policy):
    
    def __init__(self, simulaciones=180, exploracion=1.0, profundidad=20, 
                 heuristicas=True, usar_qtable=True, peso_q=0.3):
        super().__init__()
        self.simulaciones = simulaciones
        self.exploracion = exploracion
        self.profundidad = profundidad
        self.heuristicas = heuristicas
        self.usar_qtable = usar_qtable
        self.peso_q = peso_q
        self.qtable = defaultdict(float)

    def mount(self):
        if self.usar_qtable:
            ruta = Path(__file__).parent.parent.parent / "train" / "q_table.pkl"
            self.qtable = cargar_qtable(ruta)

    def act(self, s):
        rojas = np.sum(s == -1)
        amarillas = np.sum(s == 1)
        jugador = -1 if rojas == amarillas else 1
        estado = ConnectState(board=s, player=jugador)
        
        movimiento_rapido = self.verificar_inmediato(estado)
        if movimiento_rapido is not None:
            return int(movimiento_rapido)
        
        if self.usar_qtable and self.peso_q >= 0.99 and len(self.qtable) > 0:
            return self.elegir_con_qtable(estado)
        
        raiz = MCTS(estado, profundidad=self.profundidad, usar_heuristicas=self.heuristicas,
                    qtable=self.qtable, peso_q=self.peso_q)
        
        for _ in range(self.simulaciones):
            nodo = raiz
            
            while True:
                if nodo.es_terminal():
                    break
                if not nodo.esta_expandido():
                    break
                nodo = nodo.mejor_hijo(self.exploracion)
            
            if not nodo.es_terminal():
                if not nodo.esta_expandido():
                    nodo = nodo.expandir()
            
            recompensa = nodo.simular(jugador)
            
            while nodo is not None:
                nodo.visits += 1
                if nodo.parent is None:
                    nodo.wins += recompensa
                elif nodo.parent.state.player == jugador:
                    nodo.wins += recompensa
                else:
                    nodo.wins += 1 - recompensa
                nodo = nodo.parent
        
        mejor_accion = None
        max_visitas = -1
        for accion, hijo in raiz.children.items():
            if hijo.visits > max_visitas:
                max_visitas = hijo.visits
                mejor_accion = accion
        
        if mejor_accion is None:
            mejor_accion = estado.get_free_cols()[0]
        
        return int(mejor_accion)
    
    def elegir_con_qtable(self, estado):
        columnas = estado.get_free_cols()
        mejor = columnas[0]
        mejor_q = -float('inf')
        
        for col in columnas:
            clave = codificar_estado_accion(estado.board, estado.player, col)
            q = self.qtable.get(clave, 0.0)
            if q > mejor_q:
                mejor_q = q
                mejor = col
        return int(mejor)
    
    def verificar_inmediato(self, estado):
        columnas = estado.get_free_cols()
        
        for col in columnas:
            sig = estado.transition(col)
            if sig.get_winner() == estado.player:
                return col
        
        oponente = -estado.player
        for col in columnas:
            test = ConnectState(board=estado.board.copy(), player=oponente)
            sig = test.transition(col)
            if sig.get_winner() == oponente:
                return col
        
        return None

