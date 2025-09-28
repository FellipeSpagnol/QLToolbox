import numpy as np
import matplotlib.pyplot as plt
import os


def criar_e_salvar_grade_de_obstaculos(
    tamanho_x, tamanho_y, nome_arquivo="grid_de_obstaculos.npy"
):
    """
    Cria uma janela interativa para o usuário desenhar uma grade de obstáculos
    e a salva como um arquivo .npy no formato (X, Y).

    Na interface gráfica:
    - O eixo X representa a direção horizontal (colunas).
    - O eixo Y representa a direção vertical (linhas).

    Args:
        tamanho_x (int): O número de células na direção horizontal (largura).
        tamanho_y (int): O número de células na direção vertical (altura).
        nome_arquivo (str, optional): O nome do arquivo para salvar a grade.
                                      Default é "grid_de_obstaculos.npy".

    Returns:
        numpy.ndarray: A matriz da grade de obstáculos salva.
                       O formato da matriz é (tamanho_x, tamanho_y).
    """
    # A matriz é criada com (linhas, colunas), que corresponde a (y, x)
    grid_obstaculos = np.zeros((tamanho_y, tamanho_x), dtype=int)
    fig, ax = plt.subplots(figsize=(8, 8))

    def on_click(event):
        """Função interna para lidar com eventos de clique do mouse."""
        # Ignora cliques fora da área do gráfico
        if event.xdata is None or event.ydata is None:
            return

        # Converte as coordenadas do clique para índices da matriz
        ix, iy = int(round(event.xdata)), int(round(event.ydata))

        # Verifica se o clique está dentro dos limites da grade
        if 0 <= ix < tamanho_x and 0 <= iy < tamanho_y:
            # Inverte o valor da célula (0 para 1, 1 para 0)
            # Acessamos a matriz como [linha, coluna] -> [iy, ix]
            grid_obstaculos[iy, ix] = 1 - grid_obstaculos[iy, ix]

            # Limpa e redesenha a grade para refletir a mudança
            ax.clear()
            ax.imshow(grid_obstaculos, cmap="Reds", origin="lower", vmin=0, vmax=1)

            # Configurações visuais da grade
            ax.set_xticks(np.arange(-0.5, tamanho_x, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, tamanho_y, 1), minor=True)
            ax.grid(which="minor", color="black", linestyle="-", linewidth=1.5)
            ax.tick_params(which="minor", size=0)
            ax.set_xticks(np.arange(0, tamanho_x, 1))
            ax.set_yticks(np.arange(0, tamanho_y, 1))
            ax.set_xlabel("Eixo X")
            ax.set_ylabel("Eixo Y")
            ax.set_title("Defina os obstáculos. Feche a janela para salvar.")
            fig.canvas.draw()

    # Exibição inicial da grade
    ax.imshow(grid_obstaculos, cmap="Reds", origin="lower", vmin=0, vmax=1)
    ax.set_xticks(np.arange(-0.5, tamanho_x, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, tamanho_y, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle="-", linewidth=1.5)
    ax.tick_params(which="minor", size=0)
    ax.set_xticks(np.arange(0, tamanho_x, 1))
    ax.set_yticks(np.arange(0, tamanho_y, 1))
    ax.set_xlabel("Eixo X")
    ax.set_ylabel("Eixo Y")
    ax.set_title("Defina os obstáculos. Feche a janela para salvar.")

    # Conecta o evento de clique à função on_click
    fig.canvas.mpl_connect("button_press_event", on_click)

    # Mostra a janela. O script pausa aqui até que a janela seja fechada.
    plt.show()

    # --- Lógica de Salvamento ---
    output_dir = "grids"
    # Cria o diretório 'grids' se ele não existir
    os.makedirs(output_dir, exist_ok=True)

    file_path = os.path.join(output_dir, nome_arquivo)

    # Transpõe a matriz para o padrão (X, Y) antes de salvar
    grid_para_salvar = grid_obstaculos.T

    # Salva a matriz transposta no formato binário do NumPy (.npy)
    np.save(file_path, grid_para_salvar)

    print(
        f"\nGrade definida. Matriz interna usada para desenho (Y, X) tem formato: {grid_obstaculos.shape}"
    )
    print(grid_obstaculos)
    print(
        f"\nMatriz salva no padrão (X, Y) com formato {grid_para_salvar.shape} em: {file_path}"
    )

    return grid_para_salvar


if __name__ == "__main__":
    # --- Exemplo de Uso ---
    # Defina aqui as dimensões desejadas para a sua grade
    LARGURA_GRID = 5  # Tamanho em X (horizontal)
    ALTURA_GRID = 5  # Tamanho em Y (vertical)

    print(f"Abrindo grade interativa de {LARGURA_GRID}x{ALTURA_GRID}...")

    # Chama a função para iniciar o processo
    grade_final = criar_e_salvar_grade_de_obstaculos(LARGURA_GRID, ALTURA_GRID)

    # Opcional: Verificar o arquivo salvo
    try:
        caminho_arquivo = os.path.join("grids", "grid_de_obstaculos.npy")
        grade_carregada = np.load(caminho_arquivo)
        print("\n--- Verificação ---")
        print(f"Arquivo '{caminho_arquivo}' carregado com sucesso.")
        if np.array_equal(grade_final, grade_carregada):
            print(
                "Verificação bem-sucedida: A matriz salva é idêntica à retornada pela função."
            )
        else:
            print(
                "Erro na verificação: A matriz salva é diferente da retornada pela função."
            )
    except Exception as e:
        print(f"Ocorreu um erro ao verificar o arquivo: {e}")
