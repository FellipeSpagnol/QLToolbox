# QLToolbox

QLToolbox is a desktop application developed in Python for the study and application of Q-Learning, a reinforcement learning algorithm. The tool allows users to configure, train, and visualize Q-Learning agents in a 2D grid environment with obstacles.

## Description

The main goal of this project is to provide an intuitive and interactive graphical interface for a better understanding of the Q-Learning algorithm. With this tool, it is possible to:

- Create custom 2D grid environments, adding or removing obstacles.
- Import and export environment maps in `.npy` format.
- Configure the agent's movement type (omnidirectional or differential).
- Adjust the reward function parameters to change the agent's behavior.
- Train the Q-Learning agent in a separate thread to not freeze the interface.
- Interactively visualize the learned policy, generating paths from different starting points.
- Export the generated paths in `.npy` format for other uses.

## Technologies Used

- **Python 3**
- **PySide6:** For the graphical user interface.
- **NumPy:** For numerical operations and grid manipulation.
- **Matplotlib:** For plotting the grid, paths, and policies.

## Prerequisites

Before you begin, you will need to have [Python](https://www.python.org/downloads/) installed on your system (version 3.7 or higher is recommended).

## Installation and Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/your-username/QLToolbox.git
    cd QLToolbox
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the dependencies:**
    ```bash
    pip install PySide6 numpy matplotlib
    ```

## How to Use

To run the application, simply execute the `gui.py` file from the root of the project:

```bash
python gui.py
```

The application will open, and you can follow the steps in the interface to configure the environment, train the agent, and visualize the results.

## Project Structure

```
QLToolbox/
├── .gitignore
├── gui.py            # Main file for the graphical interface (PySide6)
├── ql_core.py        # Core logic of the Q-Learning algorithm and environment
├── style.qss         # Stylesheet for the application's interface
├── grids/            # Directory for saving and loading grid maps
│   ├── comp_map.npy
│   └── ...
└── imgs/             # Directory for images and icons used in the UI
    └── logo.pdf
```

- **`gui.py`**: Contains all the code related to the graphical interface, including windows, widgets, and event handling. It is the entry point of the application.
- **`ql_core.py`**: Implements the main logic of the reinforcement learning environment. It includes the `Oriented2DGrid` class (the environment), the `QLAgent` class (the agent), and the `train` function.
- **`style.qss`**: A Qt Stylesheet file used to customize the appearance of the graphical interface, providing a modern and consistent look.
- **`grids/`**: A directory where environment maps (obstacle grids) can be saved and loaded from.
- **`imgs/`**: Contains images and other visual assets used in the application's UI, such as the logo.
