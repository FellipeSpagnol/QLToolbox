import matplotlib.pyplot as plt
import numpy as np
from typing import Tuple, List

# Importa as classes e a função de treinamento do outro arquivo
from ql_core import Environment, Agent, train


def plot_rewards_history(rewards_history: List[float], window_size: int = 100):
    """
    Plota o histórico de recompensas usando uma média móvel para suavizar o gráfico.

    Args:
        rewards_history: Lista com a recompensa total de cada episódio.
        window_size: O tamanho da janela para calcular a média móvel.
    """
    # Calcula a média móvel para suavizar a curva de aprendizado
    moving_averages = []
    for i in range(len(rewards_history) - window_size + 1):
        window = rewards_history[i : i + window_size]
        moving_averages.append(sum(window) / window_size)

    plt.figure(figsize=(12, 6))
    plt.plot(moving_averages)
    plt.title("Histórico de Recompensas por Episódio (Média Móvel)")
    plt.xlabel(f"Episódios (janela de {window_size})")
    plt.ylabel("Recompensa Média")
    plt.grid(True)
    plt.show()


def visualize_policy(agent: Agent, environment: Environment, psi_layer_to_show: int):
    """
    Cria uma visualização da política aprendida pelo agente para uma orientação específica.

    Args:
        agent: O agente treinado.
        environment: A instância do ambiente.
        psi_layer_to_show: O índice da camada de orientação (psi) a ser visualizada.
    """
    grid_shape = (environment.state_shape[0], environment.state_shape[1])
    policy = np.zeros(grid_shape, dtype=int)

    # Extrai a melhor ação para cada célula na camada de orientação especificada
    for x in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            state = (x, y, psi_layer_to_show)
            policy[x, y] = agent.choose_action(state)

    # Mapeia os índices de ação para vetores (dx, dy) ou símbolos
    # Este mapeamento deve corresponder à definição de ações na classe Environment
    action_map = {
        0: (1, 0),  # Direita
        1: (-1, 0),  # Esquerda
        2: (0, 1),  # Cima
        3: (0, -1),  # Baixo
        4: (1, 1),  # Diagonal
        5: (-1, -1),  # Diagonal
        6: (1, -1),  # Diagonal
        7: (-1, 1),  # Diagonal
        8: "cw",  # Rotação Horária
        9: "ccw",  # Rotação Anti-horária
    }

    # Cria o grid para as setas
    X, Y = np.meshgrid(np.arange(grid_shape[0]), np.arange(grid_shape[1]))
    U, V = np.zeros(grid_shape), np.zeros(grid_shape)
    rot_x, rot_y, rot_markers = [], [], []

    for x in range(grid_shape[0]):
        for y in range(grid_shape[1]):
            action_index = policy[x, y]
            action_viz = action_map.get(action_index)

            if isinstance(action_viz, tuple):
                U[x, y], V[x, y] = action_viz
            else:  # Ação de rotação
                rot_x.append(x)
                rot_y.append(y)
                # 'o' para rotação, poderia ser outro marcador
                rot_markers.append("o" if action_viz == "cw" else "x")

    fig, ax = plt.subplots(figsize=(10, 10))
    # Desenha as setas para movimentos de translação
    ax.quiver(X, Y, U, V, pivot="tail")

    # Desenha marcadores para ações de rotação
    for i, marker in enumerate(rot_markers):
        ax.scatter(rot_x[i], rot_y[i], marker=marker, color="red", s=100)

    # Marca o início e o fim
    start_state = environment._start
    goal_state = environment._goal
    ax.text(
        start_state[0],
        start_state[1],
        "S",
        fontsize=20,
        color="green",
        ha="center",
        va="center",
    )
    ax.text(
        goal_state[0],
        goal_state[1],
        "G",
        fontsize=20,
        color="blue",
        ha="center",
        va="center",
    )

    ax.set_xticks(np.arange(-0.5, grid_shape[0], 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_shape[1], 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1)
    ax.set_xlim(-0.5, grid_shape[0] - 0.5)
    ax.set_ylim(-0.5, grid_shape[1] - 0.5)
    ax.set_aspect("equal", adjustable="box")
    ax.set_title(f"Política Aprendida (para orientação psi = {psi_layer_to_show})")
    plt.gca().invert_yaxis()  # Inverte o eixo Y para corresponder ao sistema de coordenadas de matriz
    plt.show()
