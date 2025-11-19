import tkinter as tk
from tkinter import messagebox, ttk
import numpy as np
from connect4.connect_state import ConnectState
from connect4.utils import find_importable_classes
from connect4.policy import Policy
import threading


class Connect4GUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Connect 4 - Juega contra IA")
        self.root.configure(bg='#2C3E50')
        
        # Constantes del tablero
        self.ROWS = 6
        self.COLS = 7
        self.CELL_SIZE = 60
        self.PADDING = 8
        
        # Colores
        self.BOARD_COLOR = '#3498DB'
        self.EMPTY_COLOR = '#ECF0F1'
        self.PLAYER_COLOR = '#E74C3C'  # Rojo
        self.AI_COLOR = '#F1C40F'      # Amarillo
        self.HOVER_COLOR = '#95A5A6'
        
        # Estado del juego
        self.game_state = ConnectState()
        self.selected_policy = None
        self.ai_instance = None
        self.game_over = False
        self.ai_thinking = False
        self.hover_col = None
        
        # Cargar políticas disponibles
        self.load_policies()
        
        # Crear interfaz
        self.create_ui()
        
    def load_policies(self):
        """Carga todas las políticas disponibles desde la carpeta groups"""
        participants = find_importable_classes("groups", Policy)
        self.policies = {}
        
        for name, policy_class in participants.items():
            self.policies[name] = policy_class
            
        print(f"Políticas cargadas: {list(self.policies.keys())}")
    
    def create_ui(self):
        """Crea la interfaz gráfica completa"""
        # Frame superior - Controles
        control_frame = tk.Frame(self.root, bg='#2C3E50', pady=5)
        control_frame.pack(fill=tk.X)
        
        # Título
        title_label = tk.Label(
            control_frame,
            text="🎮 CONNECT 4 - IA CHALLENGE 🎮",
            font=('Arial', 16, 'bold'),
            bg='#2C3E50',
            fg='#ECF0F1'
        )
        title_label.pack(pady=(0, 5))
        
        # Frame para selección de política
        policy_frame = tk.Frame(control_frame, bg='#2C3E50')
        policy_frame.pack(pady=3)
        
        tk.Label(
            policy_frame,
            text="Selecciona tu oponente:",
            font=('Arial', 10),
            bg='#2C3E50',
            fg='#ECF0F1'
        ).pack(side=tk.LEFT, padx=5)
        
        # Dropdown para seleccionar política
        self.policy_var = tk.StringVar()
        policy_names = list(self.policies.keys())
        if policy_names:
            self.policy_var.set(policy_names[0])
        
        policy_dropdown = ttk.Combobox(
            policy_frame,
            textvariable=self.policy_var,
            values=policy_names,
            state='readonly',
            font=('Arial', 9),
            width=20
        )
        policy_dropdown.pack(side=tk.LEFT, padx=5)
        policy_dropdown.bind('<<ComboboxSelected>>', self.on_policy_changed)
        
        # Botón de nuevo juego
        new_game_btn = tk.Button(
            control_frame,
            text="🔄 NUEVO JUEGO",
            font=('Arial', 10, 'bold'),
            bg='#27AE60',
            fg='white',
            activebackground='#229954',
            activeforeground='white',
            cursor='hand2',
            padx=15,
            pady=5,
            command=self.new_game
        )
        new_game_btn.pack(pady=3)
        
        # Frame para información del juego
        info_frame = tk.Frame(control_frame, bg='#2C3E50')
        info_frame.pack(pady=2)
        
        self.status_label = tk.Label(
            info_frame,
            text="Selecciona un oponente y haz clic en 'NUEVO JUEGO'",
            font=('Arial', 9),
            bg='#2C3E50',
            fg='#ECF0F1'
        )
        self.status_label.pack()
        
        # Leyenda de colores
        legend_frame = tk.Frame(control_frame, bg='#2C3E50')
        legend_frame.pack(pady=2)
        
        tk.Label(
            legend_frame,
            text="●",
            font=('Arial', 14),
            bg='#2C3E50',
            fg=self.PLAYER_COLOR
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Label(
            legend_frame,
            text="TÚ",
            font=('Arial', 9, 'bold'),
            bg='#2C3E50',
            fg='#ECF0F1'
        ).pack(side=tk.LEFT, padx=3)
        
        tk.Label(
            legend_frame,
            text="●",
            font=('Arial', 14),
            bg='#2C3E50',
            fg=self.AI_COLOR
        ).pack(side=tk.LEFT, padx=(15, 2))
        
        tk.Label(
            legend_frame,
            text="IA",
            font=('Arial', 9, 'bold'),
            bg='#2C3E50',
            fg='#ECF0F1'
        ).pack(side=tk.LEFT, padx=3)
        
        # Frame para el tablero
        board_frame = tk.Frame(self.root, bg='#2C3E50', pady=5)
        board_frame.pack()
        
        # Canvas para el tablero
        canvas_width = self.COLS * self.CELL_SIZE + 2 * self.PADDING
        canvas_height = self.ROWS * self.CELL_SIZE + 2 * self.PADDING
        
        self.canvas = tk.Canvas(
            board_frame,
            width=canvas_width,
            height=canvas_height,
            bg=self.BOARD_COLOR,
            highlightthickness=0
        )
        self.canvas.pack()
        
        # Eventos del mouse
        self.canvas.bind('<Motion>', self.on_mouse_move)
        self.canvas.bind('<Button-1>', self.on_click)
        self.canvas.bind('<Leave>', self.on_mouse_leave)
        
        # Dibujar tablero inicial
        self.draw_board()
        
        # Inicializar con primera política si existe
        if self.policies:
            self.on_policy_changed(None)
    
    def on_policy_changed(self, event):
        """Maneja el cambio de política seleccionada"""
        selected_name = self.policy_var.get()
        if selected_name and selected_name in self.policies:
            self.selected_policy = self.policies[selected_name]
            self.status_label.config(
                text=f"Oponente seleccionado: {selected_name}. Haz clic en 'NUEVO JUEGO' para empezar."
            )
    
    def new_game(self):
        """Inicia un nuevo juego"""
        if not self.selected_policy:
            messagebox.showwarning(
                "Sin oponente",
                "Por favor selecciona un oponente primero."
            )
            return
        
        # Reiniciar estado
        self.game_state = ConnectState()
        self.ai_instance = self.selected_policy()
        self.ai_instance.mount()
        self.game_over = False
        self.ai_thinking = False
        
        # Actualizar interfaz
        self.draw_board()
        self.status_label.config(
            text=f"Tu turno - Jugando contra {self.policy_var.get()}",
            fg='#ECF0F1'
        )
    
    def draw_board(self):
        """Dibuja el tablero completo"""
        self.canvas.delete('all')
        
        # Dibujar celdas
        for row in range(self.ROWS):
            for col in range(self.COLS):
                x1 = col * self.CELL_SIZE + self.PADDING
                y1 = row * self.CELL_SIZE + self.PADDING
                x2 = x1 + self.CELL_SIZE
                y2 = y1 + self.CELL_SIZE
                
                # Fondo de celda
                self.canvas.create_rectangle(
                    x1, y1, x2, y2,
                    fill=self.BOARD_COLOR,
                    outline=''
                )
                
                # Círculo (ficha o vacío)
                center_x = x1 + self.CELL_SIZE / 2
                center_y = y1 + self.CELL_SIZE / 2
                radius = self.CELL_SIZE / 2 - 8
                
                # Determinar color según el estado
                cell_value = self.game_state.board[row, col]
                if cell_value == -1:  # Jugador (Rojo)
                    color = self.PLAYER_COLOR
                elif cell_value == 1:  # IA (Amarillo)
                    color = self.AI_COLOR
                else:  # Vacío
                    color = self.EMPTY_COLOR
                
                # Dibujar círculo
                self.canvas.create_oval(
                    center_x - radius, center_y - radius,
                    center_x + radius, center_y + radius,
                    fill=color,
                    outline='#34495E',
                    width=2
                )
        
        # Dibujar preview de hover si hay
        if self.hover_col is not None and not self.game_over and not self.ai_thinking:
            self.draw_hover_preview(self.hover_col)
    
    def draw_hover_preview(self, col):
        """Dibuja una preview de dónde caerá la ficha"""
        if not self.game_state.is_col_free(col):
            return
        
        # Encontrar la fila donde caerá
        for row in range(self.ROWS - 1, -1, -1):
            if self.game_state.board[row, col] == 0:
                x1 = col * self.CELL_SIZE + self.PADDING
                y1 = row * self.CELL_SIZE + self.PADDING
                center_x = x1 + self.CELL_SIZE / 2
                center_y = y1 + self.CELL_SIZE / 2
                radius = self.CELL_SIZE / 2 - 8
                
                # Dibujar círculo semi-transparente
                self.canvas.create_oval(
                    center_x - radius, center_y - radius,
                    center_x + radius, center_y + radius,
                    fill='',
                    outline=self.PLAYER_COLOR,
                    width=3,
                    dash=(5, 5),
                    tags='preview'
                )
                break
    
    def on_mouse_move(self, event):
        """Maneja el movimiento del mouse sobre el canvas"""
        if self.game_over or self.ai_thinking:
            return
        
        col = (event.x - self.PADDING) // self.CELL_SIZE
        if 0 <= col < self.COLS:
            if self.hover_col != col:
                self.hover_col = col
                self.draw_board()
    
    def on_mouse_leave(self, event):
        """Maneja cuando el mouse sale del canvas"""
        self.hover_col = None
        self.canvas.delete('preview')
    
    def on_click(self, event):
        """Maneja el clic en el tablero"""
        if self.game_over or self.ai_thinking or not self.ai_instance:
            return
        
        col = (event.x - self.PADDING) // self.CELL_SIZE
        
        if 0 <= col < self.COLS:
            self.make_player_move(col)
    
    def make_player_move(self, col):
        """Realiza el movimiento del jugador"""
        if not self.game_state.is_applicable(col):
            messagebox.showwarning("Movimiento inválido", "Esta columna está llena.")
            return
        
        # Aplicar movimiento
        self.game_state = self.game_state.transition(col)
        self.draw_board()
        
        # Verificar si el juego terminó
        if self.check_game_over():
            return
        
        # Turno de la IA
        self.ai_thinking = True
        self.status_label.config(text="IA está pensando...", fg='#F39C12')
        self.root.update()
        
        # Ejecutar movimiento de IA en thread separado
        threading.Thread(target=self.make_ai_move, daemon=True).start()
    
    def make_ai_move(self):
        """Realiza el movimiento de la IA"""
        try:
            # Obtener movimiento de la IA
            ai_col = self.ai_instance.act(self.game_state.board)
            
            # Aplicar movimiento (debe ejecutarse en el hilo principal)
            self.root.after(300, lambda: self.apply_ai_move(ai_col))
        except Exception as e:
            print(f"Error en movimiento de IA: {e}")
            self.root.after(0, lambda: messagebox.showerror(
                "Error",
                f"La IA encontró un error: {str(e)}"
            ))
            self.ai_thinking = False
    
    def apply_ai_move(self, col):
        """Aplica el movimiento de la IA (debe ejecutarse en hilo principal)"""
        try:
            if self.game_state.is_applicable(col):
                self.game_state = self.game_state.transition(col)
                self.draw_board()
                
                if not self.check_game_over():
                    self.status_label.config(
                        text=f"Tu turno - Jugando contra {self.policy_var.get()}",
                        fg='#ECF0F1'
                    )
            else:
                messagebox.showerror(
                    "Error de IA",
                    f"La IA intentó un movimiento inválido (columna {col})"
                )
        finally:
            self.ai_thinking = False
    
    def check_game_over(self):
        """Verifica si el juego ha terminado"""
        if self.game_state.is_final():
            self.game_over = True
            winner = self.game_state.get_winner()
            
            if winner == -1:  # Jugador ganó
                self.status_label.config(
                    text="🎉 ¡GANASTE! 🎉",
                    fg='#27AE60'
                )
                messagebox.showinfo(
                    "¡Victoria!",
                    "¡Felicidades! Has derrotado a la IA."
                )
            elif winner == 1:  # IA ganó
                self.status_label.config(
                    text="😔 La IA ganó",
                    fg='#E74C3C'
                )
                messagebox.showinfo(
                    "Derrota",
                    "La IA ha ganado esta vez. ¡Intenta de nuevo!"
                )
            else:  # Empate
                self.status_label.config(
                    text="🤝 Empate",
                    fg='#F39C12'
                )
                messagebox.showinfo(
                    "Empate",
                    "El juego terminó en empate."
                )
            return True
        return False


def main():
    root = tk.Tk()
    app = Connect4GUI(root)
    
    # Centrar ventana
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()
