import numpy as np
import random

class QLearningAgent:
    def __init__(self, actions, gamma=0.95, ne=5, r_plus=2.0):
        # Este agente implementa o Q-Learning seguindo de perto o pseudocódigo
        # que nos foi aprsentado. Ele aprende uma política ótima por interação
        # direta com o ambiente, sem precisar conhecer o modelo do mundo.

        # Aqui ele lista as ações possíveis (vinda do ambiente).
        self.actions = actions

        # Fator de desconto (gamma): controla o peso das recompensas futuras.
        self.gamma = gamma

        # Parâmetros da função de exploração f(u,n):
        # ne → número mínimo de visitas para confiar no valor Q
        # r_plus → recompensa otimista para incentivar exploração
        self.ne = ne
        self.r_plus = r_plus
        
        # Estruturas persistentes do pseudocódigo:
        # Q      → tabela de valores estado-ação
        # N_sa   → contagem de visitas por (s,a)
        # s, a   → memória de curto prazo do último passo
        self.Q = {}
        self.N_sa = {}
        self.s = None
        self.a = None

    # Função de exploração f(u, n) do livro.
    # Enquanto uma ação foi pouco explorada (n < ne),
    # retornamos r_plus para forçar o agente a explorá-la.
    def f(self, u, n):
        if n < self.ne:
            return self.r_plus
        else:
            return u

    # Funções auxiliares para acessar Q e N_sa com valor padrão.
    # Isso evita KeyError e implementa Q inicial igual a zero.
    def get_q(self, state, action):
        return self.Q.get((state, action), 0.0)

    def get_n(self, state, action):
        return self.N_sa.get((state, action), 0)

    # Implementação principal do agente (Q-LEARNING-AGENT).
    # Recebe o percept (estado atual, recompensa recebida)
    # e retorna a próxima ação.
    def agent_step(self, percept):
        s_prime, r = percept
        
        # Parte de aprendizado:
        # Se já temos um estado anterior, atualizamos Q(s,a)
        # usando a equação clássica do Q-learning.
        if self.s is not None:
            prev_n = self.get_n(self.s, self.a)
            self.N_sa[(self.s, self.a)] = prev_n + 1
            
            # Alpha decrescente para garantir convergência.
            # Essa forma vem diretamente do estilo do livro.
            alpha = 60.0 / (59.0 + prev_n + 1)
            
            # Calcula max_a' Q(s', a')
            q_next_vals = [self.get_q(s_prime, act) for act in self.actions]
            max_q_next = max(q_next_vals) if q_next_vals else 0.0
            
            # Atualização principal do Q-learning:
            # Q <- Q + alpha * (r + gamma * maxQ - Q)
            current_q = self.get_q(self.s, self.a)
            self.Q[(self.s, self.a)] = current_q + alpha * (
                r + self.gamma * max_q_next - current_q
            )

        # Atualiza estado atual na memória do agente.
        self.s = s_prime
        
        # Seleção de ação usando argmax da função de exploração.
        # Isso implementa exploração otimista do Russell & Norvig.
        best_action = None
        max_val = -float('inf')
        
        # Embaralhar evita viés fixo quando há empate.
        random_actions = list(self.actions)
        random.shuffle(random_actions)

        for action in random_actions:
            u = self.get_q(s_prime, action)
            n = self.get_n(s_prime, action)
            val = self.f(u, n)
            
            if val > max_val:
                max_val = val
                best_action = action
        
        # Guarda a ação escolhida (memória de curto prazo).
        self.a = best_action
        return self.a
    
    # Método utilitário para reiniciar episódios, e parte
    # importante: não apaga a Q-table, apenas limpa a memória
    # do último estado/ação a fim de otimizar o resultado!
    def reset_episode(self):
        self.s = None
        self.a = None
