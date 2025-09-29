import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ql_core import Oriented2DGrid, QLAgent


def plot_rewards(rewards_history: list[float]):
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, alpha=0.6, label="Recompensa por Episódio")

    # Adiciona uma média móvel para visualizar a tendência de aprendizado
    # A linha vermelha deve subir, indicando que o agente está a melhorar
    moving_avg = np.convolve(rewards_history, np.ones(100) / 100, mode="valid")
    plt.plot(moving_avg, color="red", linewidth=2, label="Média Móvel (100 episódios)")

    plt.title("Recompensa Total por Episódio Durante o Treinamento")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Total")
    plt.grid(True)
    plt.legend()
    plt.show()


def visualize_policy(agent: QLAgent, environment: Oriented2DGrid):
    # Mapeamento do índice da ação para um símbolo intuitivo
    action_symbols = {
        0: "→",  # Direita
        1: "←",  # Esquerda
        2: "↑",  # Cima
        3: "↓",  # Baixo
        4: "↗",  # Cima-Direita
        5: "↙",  # Baixo-Esquerda
        6: "↘",  # Baixo-Direita
        7: "↖",  # Cima-Esquerda
        8: "↻",  # Girar no sentido horário
        9: "↺",  # Girar no sentido anti-horário
    }

    x_size, y_size = environment.state_shape[0], environment.state_shape[1]
    psi_size = agent.q_table.shape[2]

    # Acessa a grade de obstáculos de forma segura
    obs_grid = getattr(environment, "_obs_grid", None)

    # Cria 8 subplots, um para cada orientação (psi)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle(
        f"Política Aprendida para Cada Orientação (ψ)",
        fontsize=20,
    )
    axes = axes.flatten()

    for psi in range(psi_size):
        ax = axes[psi]
        ax.set_title(f"Orientação ψ = {psi} ({psi*45}°)")
        ax.set_xlim(-0.5, x_size - 0.5)
        ax.set_ylim(-0.5, y_size - 0.5)
        ax.set_xticks(np.arange(x_size))
        ax.set_yticks(np.arange(y_size))
        ax.grid(True)
        ax.set_aspect("equal")

        for x in range(x_size):
            for y in range(y_size):
                # MODIFICAÇÃO: Verifica se há um obstáculo na posição (x, y)
                if obs_grid is not None and obs_grid[x, y] != 0:
                    # Desenha um quadrado preto para representar o obstáculo
                    rect = patches.Rectangle(
                        (x - 0.5, y - 0.5),
                        1,
                        1,
                        linewidth=1,
                        edgecolor="none",
                        facecolor="black",
                    )
                    ax.add_patch(rect)
                    continue  # Pula para a próxima célula

                state = (x, y, psi)

                # Encontra a melhor ação para o estado (x, y, psi)
                best_action_index = np.argmax(agent.q_table[state])
                symbol = action_symbols[int(best_action_index)]

                # Exibe o símbolo da ação na célula
                ax.text(
                    x, y, symbol, ha="center", va="center", color="black", fontsize=14
                )

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def plot_epsilon_history(epsilon_history: list[float]):
    """
    Gera um gráfico mostrando o decaimento do valor de epsilon
    ao longo dos episódios de treinamento.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(epsilon_history)
    plt.title("Histórico de Decaimento do Epsilon")
    plt.xlabel("Episódio")
    plt.ylabel("Valor de Epsilon")
    plt.grid(True)
    # A escala logarítmica ajuda a visualizar melhor o decaimento exponencial
    plt.yscale("log")
    plt.show()
