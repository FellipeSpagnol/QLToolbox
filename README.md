QLToolbox: Q-Learning Path Planning Toolbox

⚠️ Status do Projeto: Em Desenvolvimento

Este software está atualmente em fase ativa de desenvolvimento. Funcionalidades, APIs e interfaces podem sofrer alterações.

📖 Sobre o Projeto

O QLToolbox é uma aplicação desktop interativa projetada para simular, treinar e visualizar agentes de navegação autônoma utilizando algoritmos de Aprendizado por Reforço (especificamente Q-Learning).

O objetivo principal é oferecer uma ferramenta visual onde pesquisadores, estudantes ou entusiastas possam:

Desenhar ambientes de grade personalizados (mapas com obstáculos).

Configurar parâmetros de recompensa e tipos de agentes (Omnidirecional ou Diferencial).

Treinar o agente em tempo real.

Visualizar a política aprendida e o caminho resultante.

A ferramenta abstrai a complexidade matemática do Q-Learning através de uma interface gráfica amigável (GUI), permitindo experimentação rápida sem a necessidade de reescrever código para cada cenário.

🚀 Tecnologias Utilizadas

O projeto foi construído utilizando uma stack robusta de Python para computação científica e interfaces gráficas:

Linguagem: Python 3.8+

Interface Gráfica (GUI): PySide6 (Qt for Python)

Computação Numérica: NumPy

Visualização de Dados: Matplotlib (Integrado ao Qt via FigureCanvasQTAgg)

Engine de Renderização: QtSvg (para ícones e vetores)

📋 Pré-requisitos

Antes de começar, certifique-se de ter instalado em sua máquina:

Python 3.x: O interpretador Python deve estar acessível via terminal.

pip: Gerenciador de pacotes do Python.

🔧 Instalação e Configuração

Siga os passos abaixo para configurar o ambiente de desenvolvimento local:

1. Clonar o Repositório

git clone [https://github.com/seu-usuario/ql-toolbox.git](https://github.com/seu-usuario/ql-toolbox.git)
cd ql-toolbox


2. Criar um Ambiente Virtual (Recomendado)

É uma boa prática isolar as dependências do projeto.

Windows:

python -m venv venv
.\venv\Scripts\activate


Linux/macOS:

python3 -m venv venv
source venv/bin/activate


3. Instalar Dependências

Com base nas importações do código (gui.py e ql_core.py), instale as bibliotecas necessárias:

pip install numpy matplotlib PySide6


4. Configuração de Assets e Estilos

O código espera uma estrutura de arquivos específica para funcionar corretamente. Certifique-se de que os seguintes arquivos estejam presentes no diretório raiz:

Estilos: O arquivo style.qss deve estar na raiz (fornecido no repositório).

Imagens: O arquivo gui.py tenta carregar uma imagem de logo. Crie uma pasta imgs e adicione o arquivo:

Caminho esperado: ./imgs/logo.pdf

Nota: Se o arquivo não existir, o sistema usará um placeholder cinza, mas lançará um aviso no console.

🕹️ Como Usar

Para iniciar a aplicação, execute o arquivo principal:

python gui.py


Fluxo de Trabalho da Aplicação:

Tela Inicial: Clique em "Start Configuration".

Editor de Grade (Unified Grid):

Defina o tamanho da grade (linhas e colunas).

Use o mouse para desenhar obstáculos (botão esquerdo desenha/apaga).

Posicione o Start (Início) e o Goal (Objetivo).

Configure a orientação inicial e final (ou marque "Goal Orientation Irrelevant").

Opções: Você pode Importar/Salvar mapas (.npy).

Configuração de Treinamento:

Tipo de Agente: Escolha entre Omnidirectional (move-se em 8 direções) ou Differential (modelo tipo tanque/carro).

Recompensas: Ajuste os pesos para Autonomia (custo de movimento), Agilidade (custo de curva) e Segurança (proximidade de obstáculos).

Treinamento:

Acompanhe o processo de treinamento através do spinner de carregamento. O algoritmo Q-Learning rodará em uma thread separada.

Resultados Interativos:

Clique em qualquer célula livre do grid para definir um ponto de partida.

Clique em "Visualize Path" para ver o caminho guloso (greedy) gerado pela política aprendida.

Use "Add Obstacles" para testar a robustez da política (adicionar barreiras após o treino).

Exporte o caminho gerado para análise posterior.

📂 Estrutura do Projeto

Abaixo está a organização dos principais arquivos do projeto:

ql-toolbox/
├── gui.py              # Ponto de entrada da aplicação. Gerencia a UI e navegação.
├── ql_core.py          # Lógica de negócio: Ambiente (Grid) e Agente (Q-Learning).
├── style.qss           # Folha de estilos (CSS-like) para customização do PySide6.
├── .gitignore          # Arquivos ignorados pelo Git (caches, etc).
└── imgs/               # [Necessário criar] Diretório para assets gráficos.
    └── logo.pdf        # Logo exibido na tela inicial.


⚙️ Detalhes de Implementação e Customização

Lógica do Q-Learning (ql_core.py)

O núcleo do aprendizado reside na classe QLAgent. Se você precisar ajustar hiperparâmetros de aprendizado que não estão na GUI, edite as seguintes variáveis padrão na inicialização da classe ou na função train:

learning_rate (alpha): 0.2

discount_factor (gamma): 0.9

epsilon_start: 1.0 (Decaimento exponencial implementado).

Estilização (style.qss)

A aparência da aplicação é controlada externamente. Você pode alterar cores, fontes e bordas editando o arquivo style.qss sem precisar tocar no código Python.

Desenvolvido com foco em Educação e Prototipagem em Robótica.
