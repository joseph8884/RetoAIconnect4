import numpy as np
import pickle
import json
from pathlib import Path
from collections import defaultdict
from datetime import datetime
import random
import math
import sys

sys.path.append(str(Path(__file__).parent))
from connect4.connect_state import ConnectState


def codificar_estado(tablero, jugador):
    flat = tablero.flatten()
    return f"{jugador}:" + ",".join(map(str, flat))


def crear_clave_estado_accion(tablero, jugador, accion):
    estado = codificar_estado(tablero, jugador)
    return f"{estado}|{accion}"


class AgenteQ:
    def __init__(self, episodios, epsilon_inicio, epsilon_fin, epsilon_decay, 
                 gamma, alpha, ucb_c):
        self.Q = defaultdict(float)
        self.N = defaultdict(int)
        self.visitas_estado = defaultdict(int)
        
        self.epsilon = epsilon_inicio
        self.epsilon_fin = epsilon_fin
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.alpha = alpha
        self.ucb_c = ucb_c
    
    def obtener_q(self, estado, accion):
        clave = crear_clave_estado_accion(estado.board, estado.player, accion)
        return self.Q[clave]
    
    def establecer_q(self, estado, accion, valor):
        clave = crear_clave_estado_accion(estado.board, estado.player, accion)
        self.Q[clave] = valor
    
    def elegir_con_ucb(self, estado, columnas):
        clave_estado = codificar_estado(estado.board, estado.player)
        visitas_totales = self.visitas_estado[clave_estado]
        
        if visitas_totales == 0:
            return random.choice(columnas)
        
        mejor = None
        mejor_ucb = -float('inf')
        
        for col in columnas:
            clave = crear_clave_estado_accion(estado.board, estado.player, col)
            visitas_accion = self.N[clave]
            
            if visitas_accion == 0:
                return col
            
            q = self.Q[clave]
            exploracion = self.ucb_c * math.sqrt(math.log(visitas_totales) / visitas_accion)
            ucb = q + exploracion
            
            if ucb > mejor_ucb:
                mejor_ucb = ucb
                mejor = col
        
        return mejor if mejor else random.choice(columnas)
    
    def elegir_con_epsilon(self, estado, columnas):
        if random.random() < self.epsilon:
            return random.choice(columnas)
        
        mejor = None
        mejor_q = -float('inf')
        
        for col in columnas:
            q = self.obtener_q(estado, col)
            if q > mejor_q:
                mejor_q = q
                mejor = col
        
        return mejor if mejor else random.choice(columnas)
    
    def seleccionar_accion(self, estado, usar_ucb=False):
        columnas = estado.get_free_cols()
        
        if usar_ucb:
            return self.elegir_con_ucb(estado, columnas)
        else:
            return self.elegir_con_epsilon(estado, columnas)
    
    def actualizar_epsilon(self):
        self.epsilon = max(self.epsilon_fin, self.epsilon * self.epsilon_decay)


class Aleatorio:
    def actuar(self, estado):
        return random.choice(estado.get_free_cols())


class Entrenador:
    def __init__(self, episodios=1000, epsilon_inicio=1.0, epsilon_fin=0.1, 
                 epsilon_decay=0.995, gamma=0.95, alpha=0.1, ucb_c=2.0,
                 prob_aleatorio=0.3):
        
        self.episodios = episodios
        self.prob_aleatorio = prob_aleatorio
        self.guardar_cada = 100
        self.carpeta_salida = Path(__file__).parent / "train"
        
        self.agente = AgenteQ(episodios, epsilon_inicio, epsilon_fin, 
                             epsilon_decay, gamma, alpha, ucb_c)
        self.oponente_aleatorio = Aleatorio()
        
        self.victorias = 0
        self.derrotas = 0
        self.empates = 0
        self.historial_q = []
    
    def jugar_partida(self, numero_episodio):
        estado = ConnectState()
        trayectoria = []
        
        contra_aleatorio = random.random() < self.prob_aleatorio
        usar_ucb = numero_episodio % 2 == 0
        
        movimientos = 0
        while not estado.is_final() and movimientos < 42:
            jugador_actual = estado.player
            
            if jugador_actual == -1:
                accion = self.agente.seleccionar_accion(estado, usar_ucb)
            else:
                if contra_aleatorio:
                    accion = self.oponente_aleatorio.actuar(estado)
                else:
                    accion = self.agente.seleccionar_accion(estado, not usar_ucb)
            
            siguiente = estado.transition(accion)
            
            if jugador_actual == -1:
                trayectoria.append((estado, accion))
            
            estado = siguiente
            movimientos += 1
        
        ganador = estado.get_winner()
        if ganador == -1:
            recompensa = 1.0
            self.victorias += 1
        elif ganador == 1:
            recompensa = -1.0
            self.derrotas += 1
        else:
            recompensa = 0.0
            self.empates += 1
        
        return trayectoria, recompensa
    
    def actualizar_q_values(self, trayectoria, recompensa_final):
        retorno = recompensa_final
        visitados = set()
        
        for i in range(len(trayectoria) - 1, -1, -1):
            estado, accion = trayectoria[i]
            clave = crear_clave_estado_accion(estado.board, estado.player, accion)
            
            if clave not in visitados:
                visitados.add(clave)
                
                q_viejo = self.agente.obtener_q(estado, accion)
                q_nuevo = q_viejo + self.agente.alpha * (retorno - q_viejo)
                self.agente.establecer_q(estado, accion, q_nuevo)
                
                self.agente.N[clave] += 1
                clave_estado = codificar_estado(estado.board, estado.player)
                self.agente.visitas_estado[clave_estado] += 1
            
            retorno = self.agente.gamma * retorno
    
    def entrenar(self):
        print(f"\n{'='*50}")
        print(f"Entrenando {self.episodios} episodios...")
        print(f"{'='*50}\n")
        
        for ep in range(1, self.episodios + 1):
            trayectoria, recompensa = self.jugar_partida(ep)
            self.actualizar_q_values(trayectoria, recompensa)
            self.agente.actualizar_epsilon()
            
            if ep % 100 == 0:
                q_promedio = np.mean(list(self.agente.Q.values())) if self.agente.Q else 0.0
                self.historial_q.append(q_promedio)
                tasa_victoria = self.victorias / ep * 100
                
                print(f"Episodio {ep}/{self.episodios}")
                print(f"  Victorias: {self.victorias} | Derrotas: {self.derrotas}")
                print(f"  Tasa victoria: {tasa_victoria:.1f}%")
                print(f"  Q-values: {len(self.agente.Q)}\n")
                
                self.guardar_checkpoint(ep)
        
        self.guardar_modelo_final()
        print(f"\n{'='*50}")
        print("Entrenamiento completado")
        print(f"{'='*50}\n")
    
    def guardar_checkpoint(self, episodio):
        ruta = self.carpeta_salida / f"checkpoint_ep{episodio}.pkl"
        datos = {
            'Q': dict(self.agente.Q),
            'N': dict(self.agente.N),
            'visitas': dict(self.agente.visitas_estado)
        }
        with open(ruta, 'wb') as f:
            pickle.dump(datos, f)
    
    def guardar_modelo_final(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        ruta_q = self.carpeta_salida / "q_table.pkl"
        with open(ruta_q, 'wb') as f:
            pickle.dump({
                'Q': dict(self.agente.Q),
                'N': dict(self.agente.N),
                'state_visits': dict(self.agente.visitas_estado)
            }, f)
        
        ruta_stats = self.carpeta_salida / f"training_stats_{timestamp}.json"
        with open(ruta_stats, 'w') as f:
            json.dump({
                'victorias': self.victorias,
                'derrotas': self.derrotas,
                'empates': self.empates,
                'q_values': len(self.agente.Q),
                'historial': self.historial_q
            }, f, indent=2)
        
        print(f"Guardado en: {self.carpeta_salida}")


if __name__ == "__main__":
    carpeta_train = Path(__file__).parent / "train"
    carpeta_train.mkdir(exist_ok=True)
    
    entrenador = Entrenador(
        episodios=1000,
        epsilon_inicio=1.0,
        epsilon_fin=0.1,
        epsilon_decay=0.995,
        gamma=0.95,
        alpha=0.1,
        ucb_c=2.0,
        prob_aleatorio=0.3
    )
    
    entrenador.entrenar()
