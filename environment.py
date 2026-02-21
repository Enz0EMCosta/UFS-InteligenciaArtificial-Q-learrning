import numpy as np

class GridWorld:
    def __init__(self):
        # Ambiente GridWorld inspirado no exemplo clássico de Russell & Norvig, é aqui
        # que efinimos o "mundo" onde o agente irá aprender por tentativa e erro, e cada um dos
        # símbolos do mapa tem um significado, exemplo:
        # S = início, F = estado seguro, H = buraco (terminal negativo), G = objetivo (terminal positivo).
        self.map = [
            ['S', 'F', 'F', 'F', 'F'],
            ['F', 'H', 'F', 'H', 'F'],
            ['F', 'F', 'F', 'H', 'F'],
            ['H', 'F', 'F', 'F', 'H'],
            ['F', 'F', 'F', 'F', 'G']
        ]

        # Aqui são as dimensões do grid — usadas para impedir que o agente saia do mapa.
        self.rows = 5
        self.cols = 5

        # E junto do conjunto de ações possíveis no ambiente, pois o
        # agente de Q-learning usará exatamente essa lista para montar a Q-table.
        self.actions = ['Cima', 'Baixo', 'Esquerda', 'Direita']
        
    def get_start_state(self):
        # sse geet_star)state retorna sempre a posição inicial do agente, e é
        # esse método que é chamado no começo de cada episódio de treinamento.
        return (0, 0)

    def step(self, row, col, action):
        # Esta função implementa a dinâmica do ambiente (função de transição do MDP)e 
        # recebe estado atual + ação e devolve:
        # (próximo_estado, recompensa, terminou)

        # Nesse trecho primeiro calculamos a nova posição respeitando os limites do grid.
        new_r, new_c = row, col
        if action == 'Cima':    
            new_r = max(0, row - 1)
        elif action == 'Baixo': 
            new_r = min(self.rows - 1, row + 1)
        elif action == 'Esquerda': 
            new_c = max(0, col - 1)
        elif action == 'Direita': 
            new_c = min(self.cols - 1, col + 1)
        
        # Após mover, verificamos o conteúdo da célula de destino, e
        # é justamente isso que determina a recompensa e se o episódio termina.
        cell = self.map[new_r][new_c]
        
        # Por padrão, cada passo tem custo -0.04 e isso incentiva o agente
        # a encontrar caminhos mais curtos.
        reward = -0.04 
        done = False
        
        # Estados terminais:
        # Objetivo → recompensa positiva.
        # Buraco → penalidade forte.
        if cell == 'G':
            reward = 1.0
            done = True
        elif cell == 'H':
            reward = -1.0
            done = True
            
        # Retorno padrão de ambientes de aprendizado por reforço.
        return (new_r, new_c), reward, done
