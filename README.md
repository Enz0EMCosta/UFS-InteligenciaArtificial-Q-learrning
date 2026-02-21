# Identificação do Projeto de Inteligência Artificial
## Nome dos Alunos responsáveis:
- Enzo Emanuel Maia Costa 
- Jackson Santana Carvalho Junior
- Adam Guilherme Mendes Lima
- Matheus Henrique Silva de Melo
- Samuel Guimarães Silva






## Resumo descritivo

A área de Inteligência Artificial (IA) tem como um de seus principais objetivos o desenvolvimento de agentes capazes de aprender a partir da interação com o ambiente, 
dentro desse contexto, o Aprendizado por Reforço destaca-se por permitir que um agente aprenda por tentativa e erro, recebendo recompensas ou penalidades de acordo com suas ações! 
Este trabalho tem como objetivo a implementação do algoritmo Q-learning, um dos métodos mais clássicos e fundamentais do Aprendizado por Reforço, aplicando-o a um problema simples e didático: 
a navegação de um agente em um ambiente do tipo Grid World, onde a escolha desse problema visa facilitar a compreensão do funcionamento do algoritmo, bem como permitir uma clara correspondência entre o
pseudocódigo apresentado em aula e a implementação prática desenvolvida pela equipe. O foco principal do projeto é evidenciar como um agente inteligente pode aprender uma política ótima de ações a 
partir da interação contínua com o ambiente, sem possuir conhecimento prévio sobre sua dinâmica interna. Dessa forma, o projeto prioriza a clareza conceitual, a fidelidade ao pseudocódigo 
apresentado em aula e a facilidade de compreensão do processo de aprendizado.





## Descrição do Problema
O problema abordado neste trabalho consiste em permitir que um agente inteligente aprenda a se locomover de forma autônoma em um ambiente bidimensional representado por uma grade 
(Grid World), partindo de uma posição inicial fixa até alcançar um estado objetivo previamente definido. O ambiente é composto por uma matriz de dimensões 5x5, na qual cada célula representa um estado 
possível do agente. O estado inicial localiza-se no canto superior esquerdo da grade, enquanto o estado objetivo encontra-se no canto inferior direito, e o agente pode executar quatro ações distintas: 
mover-se para cima, para baixo, para a esquerda ou para a direita. A dinâmica do ambiente é determinística, logo, dada uma ação válida, o próximo estado do agente é sempre o mesmo, e caso
uma ação leve o agente para fora dos limites da grade, o agente permanece no estado atual, evitando estados inválidos. O problema caracteriza-se como um cenário de Aprendizado por Reforço, no qual o agente deve aprender,
ao longo de vários episódios, quais ações são mais vantajosas em cada estado para maximizar a recompensa acumulada ao longo do tempo.

## Visualização do Projeto

Para facilitar a visualização do aprendizado do agente, do ambiente estocástico e garantir a interatividade do projeto, 
implementamos uma interface gráfica completa utilizando a biblioteca Streamlit. O projeto também foi implantado na nuvem e
pode ser testado diretamente pelo navegador através do **Hugging Face Spaces**:

🔗 **[Clique aqui para acessar a Simulação Online](https://huggingface.co/spaces/Enzit0/Inteligencia-Artificial-Q-Learning-Equipe11)**
