import numpy as np
import matplotlib.pyplot as plt


def plot_rewards(rewards_history):
    """
    Gera um gráfico mostrando a recompensa total por episódio ao longo do treinamento.
    """
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


def visualize_policy(agent, grid_size, goal):
    """
    Cria uma visualização da política aprendida pelo agente.
    Para cada orientação possível, mostra a melhor ação em cada célula do grid.
    """
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

    x_size, y_size = grid_size
    psi_size = agent._q_table.shape[2]

    # Cria 8 subplots, um para cada orientação (psi)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9))
    fig.suptitle("Política Aprendida para Cada Orientação (ψ)", fontsize=20)
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
        # A linha a seguir foi removida para que o eixo Y cresça para cima
        # ax.invert_yaxis()

        for x in range(x_size):
            for y in range(y_size):
                state = (x, y, psi)

                # Marca o estado objetivo com um 'G' verde
                if (x, y) == (goal[0], goal[1]):
                    ax.text(
                        x,
                        y,
                        "G",
                        ha="center",
                        va="center",
                        color="green",
                        fontsize=16,
                        weight="bold",
                    )
                    continue

                # Encontra a melhor ação para o estado (x, y, psi)
                best_action_index = np.argmax(agent._q_table[state])
                symbol = action_symbols[best_action_index]

                # Exibe o símbolo da ação na célula
                ax.text(
                    x, y, symbol, ha="center", va="center", color="black", fontsize=14
                )

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.show()
