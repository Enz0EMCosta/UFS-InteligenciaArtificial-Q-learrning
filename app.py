
import streamlit as st
import pandas as pd
import time
from environment import GridWorld
from q_learning_agent import QLearningAgent

# Importa a biblioteca Streamlit para criar a interface web interativa
# Pandas é usado para mostrar a grade como tabela (DataFrame)
# Biblioteca de tempo para controlar delays na animação
# Importa o ambiente GridWorld (onde o agente vive)
# Importa o agente de Q-Learning



st.set_page_config(
    page_title="Q-Learning Russell & Norvig",
    layout="wide"
)

# Título principal da aplicação
st.title("Q-Learning: Agente Explorador ")


st.markdown("""
Turma 01 - Professor Hendrik- Equipe 11 
""")


col1, col2 = st.columns([1, 2])
with col1:
    st.header("Parâmetros")


# Cria duas colunas: esquerda (controles) e direita (visualização)
    
    # Slider para escolher quantos episódios de treino
    episodes = st.slider("Número de Episódios", 100, 2000, 500)
    
    # Slider para controlar velocidade da animação
    delay = st.slider("Velocidade (apenas visualização final)", 0.01, 0.5, 0.1)
    
    # Botão que inicia o treinamento
    train_btn = st.button("Treinar Agente")
    
    # Caixa de informação com legenda do ambiente
    st.info("""
    **Legenda:**
    \n S: Início 🏁 
    \nG: Objetivo (+1.0) 💰 
    \n H: Buraco (-1.0) 🕳️ 
    \n F: Seguro (-0.04) ✅
    """)


# Aqui ele verifica se o ambiente já foi criado na sessão
if 'env' not in st.session_state:
    
    # Cria o GridWorld
    # Cria o agente usando as ações do ambiente
    # (parâmetros padrão do livro Russell & Norvig)
    st.session_state.env = GridWorld()
    st.session_state.agent = QLearningAgent(st.session_state.env.actions)
    
    # Flag para saber se já foi treinado
    st.session_state.trained = False



if train_btn:
    
    # Recupera ambiente e agente
    env = st.session_state.env
    agent = st.session_state.agent
    
    # Barra de progresso visual
    progress_bar = st.progress(0)
    
    # Texto de status do episódio
    status_text = st.empty()
    
    # Marca tempo inicial
    start_time = time.time()
    
    # Loop principal de episódios (Q-learning)
    for ep in range(episodes):
        
        # Reinicia posição do agente no estado inicial
        r, c = env.get_start_state()
        
        # Limpa memória de curto prazo do agente
        agent.reset_episode()
        
        # Primeira percepção do agente (estado inicial, recompensa 0)
        action = agent.agent_step(((r, c), 0))
        
        done = False
        
        # Loop até chegar em estado terminal
        while not done:
            
            # Executa ação no ambiente
            (next_r, next_c), reward, done = env.step(r, c, action)
            
            # Se terminou, apenas aprende o valor final
            if done:
                agent.agent_step(((next_r, next_c), reward))
            else:
                # Senão, aprende e escolhe próxima ação
                action = agent.agent_step(((next_r, next_c), reward))
            
            # Atualiza posição atual
            r, c = next_r, next_c
        
        # Atualiza barra a cada 10 episódios (economiza renderização)
        if (ep + 1) % 10 == 0:
            progress_bar.progress((ep + 1) / episodes)
            status_text.text(f"Episódio {ep+1}/{episodes}")

    # Marca que já foi treinado
    st.session_state.trained = True
    
    # Mostra tempo total de treino
    st.success(f"Treinamento concluído em {time.time() - start_time:.2f} segundos!")



with col2:
    st.header("Ambiente e Resultados")
    
    # Placeholder para atualizar a grade dinamicamente
    grid_display = st.empty()
    
    # Só permite visualizar se já treinou
    if st.session_state.trained:
        
        st.subheader("Simulação do Agente Treinado")
        
        # Botão para assistir o agente
        if st.button("Assistir Agente"):
            
            env = st.session_state.env
            agent = st.session_state.agent
            
            # Posição inicial
            r, c = env.get_start_state()
            
            # Cópia do mapa para desenhar o caminho
            path_map = [row[:] for row in env.map]
            
            done = False
            steps = 0
            
            # Reseta memória do agente
            agent.reset_episode()
            
            # Primeira ação
            action = agent.agent_step(((r, c), 0))
            
            # Limite de passos para evitar loop infinito
            while not done and steps < 20:
                
                # Mostra grade atual
                df = pd.DataFrame(path_map)
                df.iloc[r, c] = "🤖"
                grid_display.dataframe(df)
                time.sleep(delay)
                
                # Executa ação
                (next_r, next_c), reward, done = env.step(r, c, action)
                
                # Limpa rastro se for célula segura
                if env.map[r][c] == 'F':
                    path_map[r][c] = '.'
                
                # Se terminou
                if done:
                    df.iloc[next_r, next_c] = "🏆" if reward > 0 else "💀"
                    grid_display.dataframe(df)
                    
                    # Solta balões se venceu
                    if reward > 0:
                        st.balloons()
                
                else:
                    # Decide próxima ação via agente
                    action = agent.agent_step(((next_r, next_c), reward))
                
                # Atualiza posição
                r, c = next_r, next_c
                steps += 1

    else:
        # Mensagem antes do treino
        st.write("O agente ainda não foi treinado. Clique em 'Treinar Agente'.")
        
        # Mostra mapa inicial
        st.dataframe(pd.DataFrame(st.session_state.env.map))
