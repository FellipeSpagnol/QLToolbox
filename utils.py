import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from ql_core import Oriented2DGrid, QLAgent
import math as m


def plot_rewards(rewards_history: list[float]):
    """
    Gera um gráfico da recompensa total por episódio com uma média móvel.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(rewards_history, alpha=0.6, label="Recompensa por Episódio")

    moving_avg = np.convolve(rewards_history, np.ones(100) / 100, mode="valid")
    plt.plot(moving_avg, color="red", linewidth=2, label="Média Móvel (100 episódios)")

    plt.title("Recompensa Total por Episódio Durante o Treinamento")
    plt.xlabel("Episódio")
    plt.ylabel("Recompensa Total")
    plt.grid(True)
    plt.legend()
    plt.show()


def visualize_policy(agent: QLAgent, environment: Oriented2DGrid):
    """
    Visualiza a política aprendida pelo agente, destacando o início/objetivo
    e adaptando-se dinamicamente ao número de orientações.
    """
    x_size, y_size, psi_size = environment.state_shape
    obs_grid = getattr(environment, "_obs_grid", None)

    # <<< SUGESTÃO 1: Obter estados de início e objetivo >>>
    start_state = getattr(environment, "_start", (None, None, None))
    goal_state = getattr(environment, "_goal", (None, None, None))

    # <<< SUGESTÃO 2: Layout de subplot dinâmico >>>
    # Calcula o número de linhas e colunas para os subplots
    cols = min(psi_size, 4)
    rows = int(m.ceil(psi_size / cols))

    fig, axes = plt.subplots(
        rows, cols, figsize=(cols * 4.5, rows * 4.5), subplot_kw={"aspect": "equal"}
    )
    fig.suptitle(
        f"Política Aprendida - Agente '{environment._actions_type}'", fontsize=20
    )
    axes = axes.flatten()

    for psi in range(psi_size):
        ax = axes[psi]
        angle_deg = int(psi * 360 / psi_size)
        ax.set_title(f"Orientação ψ = {psi} ({angle_deg}°)")
        ax.set_xlim(-0.5, x_size - 0.5)
        ax.set_ylim(-0.5, y_size - 0.5)
        ax.set_xticks(np.arange(x_size))
        ax.set_yticks(np.arange(y_size))
        ax.grid(True)

        environment._state = (0, 0, psi)
        current_actions = environment._actions
        action_map = _generate_action_map(environment._actions_type, current_actions)

        for x in range(x_size):
            for y in range(y_size):
                if obs_grid is not None and obs_grid[x, y] != 0:
                    rect = patches.Rectangle(
                        (x - 0.5, y - 0.5), 1, 1, facecolor="black", alpha=0.8
                    )
                    ax.add_patch(rect)
                    continue

                state = (x, y, psi)
                if np.sum(agent.q_table[state]) == 0:
                    continue

                best_action_index = np.argmax(agent.q_table[state])
                symbol = action_map.get(
                    best_action_index, {"symbol": "?", "is_move": False}
                )

                rotation_angle = 0
                if environment._actions_type == "diff" and symbol["is_move"]:
                    rotation_angle = angle_deg

                ax.text(
                    x,
                    y,
                    symbol["symbol"],
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=16,
                    rotation=rotation_angle,
                    rotation_mode="anchor",
                )

        # <<< SUGESTÃO 1 (Continuação): Desenha os marcadores de início e objetivo >>>
        if start_state[0] is not None:
            ax.plot(
                start_state[0],
                start_state[1],
                "bo",
                markersize=10,
                label="Início",
                alpha=0.6,
            )
        if goal_state[0] is not None:
            ax.plot(
                goal_state[0],
                goal_state[1],
                "g*",
                markersize=15,
                label="Objetivo",
                alpha=0.8,
            )

    # Esconde eixos de subplots não utilizados
    for i in range(psi_size, len(axes)):
        axes[i].axis("off")

    plt.tight_layout(rect=(0, 0, 1, 0.95))
    plt.show()


def _generate_action_map(actions_type: str, actions: list) -> dict:
    base_symbols = {
        (0, 0, 1): {"symbol": "↺", "is_move": False},
        (0, 0, -1): {"symbol": "↻", "is_move": False},
    }

    if actions_type == "omni":
        base_symbols.update(
            {
                (1, 0, 0): {"symbol": "→", "is_move": True},
                (-1, 0, 0): {"symbol": "←", "is_move": True},
                (0, 1, 0): {"symbol": "↑", "is_move": True},
                (0, -1, 0): {"symbol": "↓", "is_move": True},
                (1, 1, 0): {"symbol": "↗", "is_move": True},
                (-1, -1, 0): {"symbol": "↙", "is_move": True},
                (1, -1, 0): {"symbol": "↘", "is_move": True},
                (-1, 1, 0): {"symbol": "↖", "is_move": True},
            }
        )
    elif actions_type == "diff":
        if len(actions) > 0:
            base_symbols[actions[0]] = {"symbol": "→", "is_move": True}
        if len(actions) > 1:
            base_symbols[actions[1]] = {"symbol": "←", "is_move": True}

    action_map = {}
    for i, action_tuple in enumerate(actions):
        action_map[i] = base_symbols.get(
            action_tuple, {"symbol": "?", "is_move": False}
        )

    return action_map


def plot_epsilon_history(epsilon_history: list[float]):
    plt.figure(figsize=(12, 6))
    plt.plot(epsilon_history)
    plt.title("Histórico de Decaimento do Epsilon")
    plt.xlabel("Episódio")
    plt.ylabel("Valor de Epsilon")
    plt.grid(True)
    plt.yscale("log")
    plt.show()
