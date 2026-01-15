import sys
import os
from typing import Optional, Tuple, Any, Dict, List
import numpy as np
from dataclasses import dataclass, fields
import math

# Use PySide6 as specified in the original code
from PySide6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QLabel,
    QStackedWidget,
    QFormLayout,
    QSpinBox,
    QDoubleSpinBox,
    QGridLayout,
    QSizePolicy,
    QButtonGroup,
    QFileDialog,
    QMessageBox,
    QGroupBox,
)
from PySide6.QtCore import Signal, Qt, QThread, QObject, QTimer, QSettings
from PySide6.QtGui import QScreen, QPainter, QColor, QPen, QPaintEvent, QPixmap
from PySide6.QtSvg import QSvgRenderer

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib.backend_bases import MouseEvent
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import matplotlib

# Import classes and functions from the ql_core
from ql_core import Oriented2DGrid, QLAgent, train

matplotlib.use("qtagg")


@dataclass
class AppData:
    """
    Data class to hold the application's state.
    """

    grid_size: Tuple[int, int] = (10, 10)
    obstacle_map: Optional[np.ndarray] = None
    start: Tuple[int, int, int] = (0, 0, 0)
    goal: Tuple[int, int, int] = (9, 9, 0)
    goal_orientation_irrelevant: bool = False
    agent_type: str = "Omnidirectional"
    # Reward gains for direct numerical input
    reward_gain_safety: float = 0.2
    reward_gain_agility: float = 0.1
    reward_gain_autonomy: float = 0.1


class LoadingSpinner(QWidget):
    """
    A widget that displays a spinning loading animation.
    """

    def __init__(
        self, parent: Optional[QWidget] = None, speed: int = 8, smoothness_ms: int = 20
    ):
        super().__init__(parent)
        self.angle = 0
        self.speed = speed
        self.smoothness_ms = smoothness_ms
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._update_angle)
        self.setMinimumSize(80, 80)
        self.setMaximumSize(80, 80)
        self.setObjectName("loadingSpinner")

    def _update_angle(self):
        self.angle = (self.angle + self.speed) % 360
        self.update()

    def paintEvent(self, event: QPaintEvent):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = self.rect().adjusted(5, 5, -5, -5)
        pen = QPen(QColor("#3498db"), 8, Qt.SolidLine, Qt.RoundCap)
        painter.setPen(pen)
        painter.drawArc(rect, self.angle * 16, 90 * 16)

    def start(self):
        self.timer.start(self.smoothness_ms)

    def stop(self):
        self.timer.stop()


class TrainingWorker(QObject):
    """
    Runs the training task in a separate thread to avoid freezing the GUI.
    """

    training_finished = Signal(QLAgent, Oriented2DGrid, dict)

    def __init__(self, app_data: AppData):
        super().__init__()
        self.data = app_data
        self.reward_params: Dict[str, float] = {}

    def run(self):
        print("Starting training in the worker thread...")

        start_angle = (self.data.start[2] * 45 * np.pi) / 180.0
        goal_angle = (self.data.goal[2] * 45 * np.pi) / 180.0

        self.set_reward_params()

        agent_type_map = {"Omnidirectional": "omni", "Differential": "diff"}
        action_type = agent_type_map.get(self.data.agent_type, "omni")

        obs_grid_for_core = (
            self.data.obstacle_map.T if self.data.obstacle_map is not None else None
        )

        start_pos = (self.data.start[0], self.data.start[1], start_angle)
        goal_pos = (self.data.goal[0], self.data.goal[1], goal_angle)
        if self.data.goal_orientation_irrelevant:
            print(
                "Warning: Goal orientation irrelevance is not natively supported by ql_core."
            )

        env = Oriented2DGrid(
            grid_size=self.data.grid_size,
            start=start_pos,
            goal=goal_pos,
            actions_type=action_type,
            reward_gains=self.reward_params,
            obs_grid=obs_grid_for_core,
        )

        agent = QLAgent(
            state_shape=env.state_shape,
            n_actions=env.n_actions,
        )

        metrics = train(
            agent,
            env,
        )

        print("Training finished.")
        self.training_finished.emit(agent, env, metrics)

    def set_reward_params(self):
        self.reward_params = {
            "goal": 100.0,
            "invalid": 100.0,
            "move": self.data.reward_gain_autonomy,
            "turn": self.data.reward_gain_agility,
            "nearby_obs": self.data.reward_gain_safety,
        }
        print(f"Reward parameters set: {self.reward_params}")


class CompassWidget(QWidget):
    orientation_changed = Signal(int)

    def __init__(self, title: str, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.current_orientation = 0
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setObjectName("compassTitle")
        grid_layout = QGridLayout()
        grid_layout.setSpacing(2)
        self.button_map = {
            (1, 2): 0,
            (0, 2): 1,
            (0, 1): 2,
            (0, 0): 3,
            (1, 0): 4,
            (2, 0): 5,
            (2, 1): 6,
            (2, 2): 7,
        }
        arrows = {0: "→", 1: "↗", 2: "↑", 3: "↖", 4: "←", 5: "↙", 6: "↓", 7: "↘"}
        self.index_to_angle = {i: i * 45 for i in range(8)}
        self.buttons: Dict[int, QPushButton] = {}
        for (row, col), orientation in self.button_map.items():
            button = QPushButton(arrows[orientation])
            button.setObjectName("compassButton")
            button.setCheckable(True)
            button.clicked.connect(lambda _, o=orientation: self.set_orientation(o))
            grid_layout.addWidget(button, row, col)
            self.buttons[orientation] = button
        self.orientation_label = QLabel("0°")
        self.orientation_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.orientation_label.setObjectName("compassValue")
        grid_layout.addWidget(self.orientation_label, 1, 1)
        layout.addWidget(title_label)
        layout.addLayout(grid_layout)
        self.set_orientation(0)

    def set_orientation(self, orientation: int):
        if self.current_orientation in self.buttons:
            self.buttons[self.current_orientation].setChecked(False)
        self.current_orientation = orientation
        self.buttons[orientation].setChecked(True)
        angle = self.index_to_angle[orientation]
        self.orientation_label.setText(f"{angle}°")
        self.orientation_changed.emit(orientation)

    def get_orientation(self) -> int:
        return self.current_orientation


class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("QLToolbox")
        screen = self.screen()
        if screen:
            screen_geometry = screen.availableGeometry()
            screen_width = int(screen_geometry.width() * 0.8)
            screen_height = int(screen_geometry.height() * 0.8)
            self.setGeometry(
                screen_geometry.center().x() - screen_width // 2,
                screen_geometry.center().y() - screen_height // 2,
                screen_width,
                screen_height,
            )

        try:
            with open("style.qss", "r") as f:
                self.setStyleSheet(f.read())
        except Exception as e:
            print(f"Warning: Could not load 'style.qss'. {e}")

        self.data = AppData()
        self.trained_agent: Optional[QLAgent] = None
        self.trained_env: Optional[Oriented2DGrid] = None
        self.training_metrics: Optional[Dict[str, List[Any]]] = None

        self.stacked_widget = QStackedWidget(self)
        self.setCentralWidget(self.stacked_widget)

        self.start_page = StartPage(self)
        self.unified_grid_page = UnifiedGridPage(self)
        self.training_config_page = TrainingConfigurationPage(self)
        self.training_page = TrainingScreen(self)
        self.results_page = InteractiveResultsPage(self)
        self.end_page = EndPage(self)

        self.stacked_widget.addWidget(self.start_page)
        self.stacked_widget.addWidget(self.unified_grid_page)
        self.stacked_widget.addWidget(self.training_config_page)
        self.stacked_widget.addWidget(self.training_page)
        self.stacked_widget.addWidget(self.results_page)
        self.stacked_widget.addWidget(self.end_page)

        self.connect_signals_and_navigation()
        self.stacked_widget.setCurrentIndex(0)

    def connect_signals_and_navigation(self):
        self.start_page.button_next.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(1)
        )
        self.unified_grid_page.grid_confirmed.connect(self.go_to_training_config)
        self.training_config_page.button_finish.clicked.connect(self.start_training)
        self.results_page.button_finish.clicked.connect(
            lambda: self.stacked_widget.setCurrentIndex(5)
        )
        self.end_page.button_close.clicked.connect(self.close)
        self.end_page.button_new_training.clicked.connect(self.reset_application)

        self.unified_grid_page.data_changed.connect(self.update_data)
        self.training_config_page.agent_type_changed.connect(
            lambda agent: self.update_data("agent_type", agent)
        )
        self.training_config_page.param_changed.connect(self.update_data)

    def go_to_training_config(self, map_data, start, goal, grid_size, goal_irrelevant):
        self.update_data("obstacle_map", map_data)
        self.update_data("start", start)
        self.update_data("goal", goal)
        self.update_data("grid_size", grid_size)
        self.update_data("goal_orientation_irrelevant", goal_irrelevant)
        self.stacked_widget.setCurrentIndex(2)

    def start_training(self):
        self.stacked_widget.setCurrentIndex(3)
        self.training_page.start_worker()

    def update_data(self, field_name: str, value: Any):
        if hasattr(self.data, field_name):
            setattr(self.data, field_name, value)
            print(f"Updating self.data.{field_name} to {value}")
        else:
            print(f"Error: Field '{field_name}' doesn't exist.")

    def on_training_complete(
        self, agent: QLAgent, env: Oriented2DGrid, metrics: Dict[str, List[Any]]
    ):
        print("MainWindow notified of training completion. Navigating to results page.")
        self.trained_agent = agent
        self.trained_env = env
        self.training_metrics = metrics
        self.results_page.setup_page()
        self.stacked_widget.setCurrentIndex(4)

    def reset_application(self):
        print("Resetting application for new training.")
        # self.data = AppData()
        self.trained_agent = None
        self.trained_env = None
        self.training_metrics = None
        self.unified_grid_page.reset_page_to_defaults()
        self.stacked_widget.setCurrentIndex(1)


class StartPage(QWidget):
    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)

        main_layout = QVBoxLayout(self)

        header_layout = QHBoxLayout()
        header_layout.addStretch()

        # --- Image Loading Section ---
        # Load a PNG file from the same directory as the script.
        # The file should be named 'logo.png'.
        image_path = "imgs/logo.pdf"
        pixmap = QPixmap(image_path)

        # Check if the image was loaded successfully. If not, print a warning.
        if pixmap.isNull():
            print(f"Warning: Could not load image from '{image_path}'.")
            print("Please ensure 'logo.png' is in the same folder as the script.")
            # Create a gray placeholder if the image is missing
            pixmap = QPixmap(100, 100)
            pixmap.fill(Qt.GlobalColor.lightGray)

        image_label = QLabel()
        image_label.setPixmap(
            pixmap.scaled(
                200,
                200,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )
        image_label.setObjectName("titleImage")
        image_label.setAlignment(
            Qt.AlignmentFlag.AlignBottom | Qt.AlignmentFlag.AlignRight
        )
        header_layout.addWidget(image_label)

        main_layout.addLayout(header_layout)
        main_layout.addStretch(1)

        title_layout = QVBoxLayout()
        title_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_layout.setSpacing(20)

        label = QLabel("Q-Learning Path Planning Toolbox")
        label.setObjectName("titleLabel")
        label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.button_next = QPushButton("Start Configuration")
        self.button_next.setProperty("class", "navigation")

        title_layout.addWidget(label)
        title_layout.addWidget(self.button_next)

        main_layout.addLayout(title_layout)
        main_layout.addStretch(2)


class UnifiedGridPage(QWidget):
    data_changed = Signal(str, object)
    grid_confirmed = Signal(object, tuple, tuple, tuple, bool)

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        self.placement_mode = "obstacle"  # Modes: "obstacle", "start", "goal"

        main_layout = QHBoxLayout(self)

        # --- Left Controls Panel ---
        controls_panel = QWidget()
        controls_panel.setObjectName("controlsPanel")
        controls_panel.setFixedWidth(350)
        controls_layout = QVBoxLayout(controls_panel)
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignTop)
        controls_layout.setSpacing(15)

        # Grid Size Group
        size_group = QGroupBox("Grid Size")
        size_layout = QFormLayout(size_group)
        self.spinbox_grid_cols = QSpinBox()
        self.spinbox_grid_cols.setRange(4, 100)
        self.spinbox_grid_rows = QSpinBox()
        self.spinbox_grid_rows.setRange(4, 100)
        size_layout.addRow("Columns (X):", self.spinbox_grid_cols)
        size_layout.addRow("Rows (Y):", self.spinbox_grid_rows)
        controls_layout.addWidget(size_group)

        # File Operations Group
        file_group = QGroupBox("File Operations")
        file_layout = QVBoxLayout(file_group)
        self.import_button = QPushButton("Import Map")
        self.import_button.setObjectName("importButton")
        self.save_button = QPushButton("Save Map")
        self.save_button.setObjectName("saveButton")
        file_layout.addWidget(self.import_button)
        file_layout.addWidget(self.save_button)
        controls_layout.addWidget(file_group)

        # Edit Mode Group
        mode_group = QGroupBox("Editing Mode")
        mode_layout = QVBoxLayout(mode_group)
        self.mode_button_group = QButtonGroup(self)
        self.mode_button_group.setExclusive(True)

        btn_obstacle = QPushButton("Obstacles")
        btn_obstacle.setCheckable(True)
        btn_obstacle.setChecked(True)
        btn_obstacle.setProperty("class", "modeButton")
        btn_obstacle.clicked.connect(lambda: self.set_placement_mode("obstacle"))

        btn_start = QPushButton("Set Start")
        btn_start.setCheckable(True)
        btn_start.setProperty("class", "modeButton")
        btn_start.clicked.connect(lambda: self.set_placement_mode("start"))

        btn_goal = QPushButton("Set Goal")
        btn_goal.setCheckable(True)
        btn_goal.setProperty("class", "modeButton")
        btn_goal.clicked.connect(lambda: self.set_placement_mode("goal"))

        self.mode_button_group.addButton(btn_obstacle)
        self.mode_button_group.addButton(btn_start)
        self.mode_button_group.addButton(btn_goal)
        mode_layout.addWidget(btn_obstacle)
        mode_layout.addWidget(btn_start)
        mode_layout.addWidget(btn_goal)
        controls_layout.addWidget(mode_group)

        # Orientation Group
        orientation_group = QGroupBox("Orientation")
        orientation_layout = QHBoxLayout(orientation_group)
        self.start_compass = CompassWidget("Start")
        self.goal_compass = CompassWidget("Goal")
        orientation_layout.addWidget(self.start_compass)
        orientation_layout.addWidget(self.goal_compass)
        controls_layout.addWidget(orientation_group)

        self.button_irrelevant = QPushButton("Goal Orientation Irrelevant")
        self.button_irrelevant.setCheckable(True)
        self.button_irrelevant.setProperty("class", "toggleButton")
        controls_layout.addWidget(self.button_irrelevant)

        controls_layout.addStretch()

        self.confirm_button = QPushButton("Confirm Grid and Continue")
        self.confirm_button.setObjectName("confirmButton")
        controls_layout.addWidget(self.confirm_button)

        main_layout.addWidget(controls_panel)

        # --- Right Matplotlib Canvas ---
        self.fig: Figure = Figure(figsize=(8, 8), tight_layout=True)
        self.canvas: FigureCanvas = FigureCanvas(self.fig)
        self.ax: Axes = self.fig.add_subplot(111)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        main_layout.addWidget(self.canvas, 1)

        # --- Class Attributes ---
        colors = [
            "#ffffff",
            "#e74c3c",
            "#2ecc71",
            "#3498db",
        ]  # White, Obstacle, Start, Goal
        self.cmap = ListedColormap(colors)
        bounds = [-0.5, 0.5, 1.5, 2.5, 3.5]
        self.norm = BoundaryNorm(bounds, self.cmap.N)
        self.nx, self.ny = 0, 0
        self.start_pos: Optional[Tuple[int, int, int]] = None
        self.goal_pos: Optional[Tuple[int, int, int]] = None
        self.display_map: Optional[np.ndarray] = None
        self.is_drawing, self.is_dragged = False, False
        self.last_pos: Optional[Tuple[int, int]] = None
        self.drag_mode = "draw"

        self.connect_events()
        self.reset_page_to_defaults()

    def reset_page_to_defaults(self):
        defaults = AppData()
        self.spinbox_grid_cols.setValue(defaults.grid_size[0])
        self.spinbox_grid_rows.setValue(defaults.grid_size[1])
        self.start_pos = defaults.start
        self.goal_pos = defaults.goal
        self.button_irrelevant.setChecked(defaults.goal_orientation_irrelevant)
        self.start_compass.set_orientation(defaults.start[2])
        self.goal_compass.set_orientation(defaults.goal[2])
        self._on_grid_dimensions_changed()

    def set_placement_mode(self, mode: str):
        self.placement_mode = mode
        print(f"Placement mode set to: {self.placement_mode}")

    def connect_events(self):
        self.spinbox_grid_cols.valueChanged.connect(self._on_grid_dimensions_changed)
        self.spinbox_grid_rows.valueChanged.connect(self._on_grid_dimensions_changed)
        self.import_button.clicked.connect(self._handle_import)
        self.save_button.clicked.connect(self.on_save_map)
        self.confirm_button.clicked.connect(self.on_confirm)
        self.start_compass.orientation_changed.connect(self.on_start_orientation_change)
        self.goal_compass.orientation_changed.connect(self.on_goal_orientation_change)
        self.button_irrelevant.toggled.connect(self.on_relevance_changed)
        self.canvas.mpl_connect("button_press_event", self.on_press)
        self.canvas.mpl_connect("button_release_event", self.on_release)
        self.canvas.mpl_connect("motion_notify_event", self.on_motion)

    def _on_grid_dimensions_changed(self):
        if self.start_pos is None or self.goal_pos is None:
            return

        self.nx = self.spinbox_grid_cols.value()
        self.ny = self.spinbox_grid_rows.value()
        self.display_map = np.zeros((self.ny, self.nx), dtype=int)

        # Ensure start/goal are within new bounds
        sx, sy, so = self.start_pos
        gx, gy, go = self.goal_pos
        self.start_pos = (min(sx, self.nx - 1), min(sy, self.ny - 1), so)
        self.goal_pos = (min(gx, self.nx - 1), min(gy, self.ny - 1), go)

        # Avoid start and goal collision
        if self.start_pos[:2] == self.goal_pos[:2]:
            self.goal_pos = (max(0, self.start_pos[0] - 1), self.goal_pos[1], go)

        self.update_map_markers()
        self.setup_plot()

    def update_map_markers(self):
        if self.display_map is None or self.start_pos is None or self.goal_pos is None:
            return
        # Clear old start/goal markers before placing new ones
        self.display_map[self.display_map > 1] = 0
        self.display_map[self.start_pos[1], self.start_pos[0]] = 2
        self.display_map[self.goal_pos[1], self.goal_pos[0]] = 3

    def setup_plot(self):
        self.ax.clear()
        if self.display_map is None:
            return
        self.im = self.ax.imshow(
            self.display_map,
            cmap=self.cmap,
            norm=self.norm,
            interpolation="nearest",
            origin="lower",
        )
        self.ax.set_xticks(np.arange(-0.5, self.nx, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, self.ny, 1), minor=True)
        self.ax.grid(which="minor", color="black", linestyle="-", linewidth=0.5)
        self.ax.tick_params(which="minor", size=0)
        self.ax.tick_params(
            axis="x", which="major", bottom=False, top=False, labelbottom=False
        )
        self.ax.tick_params(
            axis="y", which="major", left=False, right=False, labelleft=False
        )
        self.ax.set_xlim(-0.5, self.nx - 0.5)
        self.ax.set_ylim(-0.5, self.ny - 0.5)
        self.ax.set_title("Design your environment grid")
        self.ax.set_xlabel("Columns (X)")
        self.ax.set_ylabel("Rows (Y)")
        self._draw_orientation_arrows()
        self.canvas.draw()

    def _draw_orientation_arrows(self):
        if self.start_pos is None or self.goal_pos is None:
            return

        orientations = {
            0: (1, 0),
            1: (0.707, 0.707),
            2: (0, 1),
            3: (-0.707, 0.707),
            4: (-1, 0),
            5: (-0.707, -0.707),
            6: (0, -1),
            7: (0.707, -0.707),
        }

        sx, sy, so = self.start_pos
        dx, dy = orientations[so]
        self.ax.arrow(
            sx,
            sy,
            dx * 0.25,
            dy * 0.25,
            head_width=0.2,
            head_length=0.2,
            fc="k",
            ec="k",
        )

        if not self.button_irrelevant.isChecked():
            gx, gy, go = self.goal_pos
            dx, dy = orientations[go]
            self.ax.arrow(
                gx,
                gy,
                dx * 0.25,
                dy * 0.25,
                head_width=0.2,
                head_length=0.2,
                fc="k",
                ec="k",
            )

    def on_press(self, event: MouseEvent):
        if not event.inaxes or self.display_map is None:
            return

        self.is_drawing = True
        self.is_dragged = False
        ix, iy = int(round(float(event.xdata))), int(round(float(event.ydata)))

        if not (0 <= ix < self.nx and 0 <= iy < self.ny):
            return

        if self.placement_mode == "obstacle":
            self.drag_mode = "erase" if self.display_map[iy, ix] == 1 else "draw"
        else:  # Start or Goal placement
            self.handle_placement(ix, iy)

    def on_release(self, event: MouseEvent):
        if not self.is_drawing or self.display_map is None:
            return

        if not self.is_dragged and event.inaxes and self.placement_mode == "obstacle":
            ix, iy = int(round(float(event.xdata))), int(round(float(event.ydata)))
            if (
                self.start_pos is not None
                and self.goal_pos is not None
                and (ix, iy) != self.start_pos[:2]
                and (ix, iy) != self.goal_pos[:2]
            ):
                self.display_map[iy, ix] = 1 - self.display_map[iy, ix]
                self.im.set_data(self.display_map)
                self.canvas.draw_idle()
        self.is_drawing = False

    def on_motion(self, event: MouseEvent):
        if (
            not self.is_drawing
            or not event.inaxes
            or self.display_map is None
            or self.placement_mode != "obstacle"
        ):
            return

        self.is_dragged = True
        ix, iy = int(round(float(event.xdata))), int(round(float(event.ydata)))

        if 0 <= ix < self.nx and 0 <= iy < self.ny and (ix, iy) != self.last_pos:
            if (
                self.start_pos is not None
                and self.goal_pos is not None
                and (ix, iy) != self.start_pos[:2]
                and (ix, iy) != self.goal_pos[:2]
            ):
                self.display_map[iy, ix] = 1 if self.drag_mode == "draw" else 0
                self.im.set_data(self.display_map)
                self.canvas.draw_idle()
            self.last_pos = (ix, iy)

    def handle_placement(self, x: int, y: int):
        if self.display_map is None or self.start_pos is None or self.goal_pos is None:
            return

        # Prevent placing on an obstacle
        if self.display_map[y, x] == 1:
            print("Cannot place start/goal on an obstacle.")
            return

        if self.placement_mode == "start":
            if self.goal_pos[:2] == (x, y):
                return  # Avoid collision
            self.start_pos = (x, y, self.start_compass.get_orientation())
        elif self.placement_mode == "goal":
            if self.start_pos[:2] == (x, y):
                return  # Avoid collision
            self.goal_pos = (x, y, self.goal_compass.get_orientation())

        self.update_map_markers()
        self.setup_plot()

    def on_start_orientation_change(self, orientation: int):
        if self.start_pos is None:
            return
        self.start_pos = (self.start_pos[0], self.start_pos[1], orientation)
        self.setup_plot()

    def on_goal_orientation_change(self, orientation: int):
        if self.goal_pos is None:
            return
        self.goal_pos = (self.goal_pos[0], self.goal_pos[1], orientation)
        self.setup_plot()

    def on_relevance_changed(self, is_irrelevant: bool):
        self.goal_compass.setEnabled(not is_irrelevant)
        self.setup_plot()

    def on_confirm(self):
        if self.display_map is None or self.start_pos is None or self.goal_pos is None:
            return
        final_obstacle_map = np.where(self.display_map == 1, 1, 0)
        grid_size = (self.nx, self.ny)
        goal_irrelevant = self.button_irrelevant.isChecked()
        self.grid_confirmed.emit(
            final_obstacle_map,
            self.start_pos,
            self.goal_pos,
            grid_size,
            goal_irrelevant,
        )

    def on_save_map(self):
        if self.display_map is None:
            return
        settings = QSettings("MyRLApp", "GridGenerator")
        last_dir = settings.value("last_save_dir", "")
        final_obstacle_map = np.where(self.display_map == 1, 1, 0)
        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Map", str(last_dir), "Numpy files (*.npy)"
        )
        if file_path:
            try:
                np.save(file_path, final_obstacle_map)
                directory = os.path.dirname(file_path)
                settings.setValue("last_save_dir", directory)
                QMessageBox.information(
                    self, "Success", f"Map successfully saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save map.\nError: {e}")

    def _handle_import(self):
        settings = QSettings("MyRLApp", "GridGenerator")
        last_dir = settings.value("last_save_dir", "")
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Import Grid", str(last_dir), "Numpy files (*.npy);;CSV files (*.csv)"
        )
        if not file_path:
            return
        try:
            if file_path.endswith(".npy"):
                grid_map = np.load(file_path)
            elif file_path.endswith(".csv"):
                grid_map = np.loadtxt(file_path, delimiter=",")
            else:
                raise ValueError("Unsupported file type")

            if (
                not isinstance(grid_map, np.ndarray)
                or grid_map.ndim != 2
                or not np.all(np.isin(grid_map, [0, 1]))
            ):
                raise ValueError("Map must be a 2D numpy array of 0s and 1s.")

            directory = os.path.dirname(file_path)
            settings.setValue("last_save_dir", directory)

            rows, cols = grid_map.shape
            self.spinbox_grid_cols.setValue(cols)
            self.spinbox_grid_rows.setValue(
                rows
            )  # This will trigger _on_grid_dimensions_changed

            # Now, apply the loaded obstacle map
            self.display_map = grid_map.copy().astype(int)
            self.update_map_markers()
            self.setup_plot()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to import grid.\nError: {e}")


class TrainingConfigurationPage(QWidget):
    agent_type_changed = Signal(str)
    param_changed = Signal(str, float)

    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        self.agent_types = ["Omnidirectional", "Differential"]

        main_layout = QVBoxLayout(self)
        main_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.setSpacing(40)

        title = QLabel("Select Agent Type")
        title.setObjectName("titleLabel")
        main_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(15)
        self.button_group = QButtonGroup(self)
        self.button_group.setExclusive(True)
        self.buttons: Dict[str, QPushButton] = {}
        for agent_name in self.agent_types:
            button = QPushButton(agent_name)
            button.setProperty("class", "agentButton")
            button.setCheckable(True)
            button.clicked.connect(
                lambda checked, name=agent_name: (
                    self.select_agent(name) if checked else None
                )
            )
            buttons_layout.addWidget(button)
            self.buttons[agent_name] = button
            self.button_group.addButton(button)
        main_layout.addLayout(buttons_layout)

        params_title = QLabel("Reward Parameters")
        params_title.setObjectName("titleLabel")
        main_layout.addWidget(params_title, alignment=Qt.AlignmentFlag.AlignCenter)

        params_container = QWidget()
        params_layout = QFormLayout(params_container)
        params_layout.setSpacing(20)
        params_layout.setLabelAlignment(Qt.AlignmentFlag.AlignRight)
        params_container.setObjectName("paramsContainer")

        self.param_map = {
            "reward_gain_autonomy": "Move Cost (Autonomy):",
            "reward_gain_agility": "Turn Cost (Agility):",
            "reward_gain_safety": "Proximity Cost (Safety):",
        }

        self.spinboxes: Dict[str, QDoubleSpinBox] = {}
        for param_name, label_text in self.param_map.items():
            spinbox = QDoubleSpinBox()
            spinbox.setRange(0.0, 100.0)
            spinbox.setSingleStep(0.01)
            spinbox.setDecimals(3)
            initial_value = getattr(self.main_window.data, param_name, 0.1)
            spinbox.setValue(initial_value)

            spinbox.valueChanged.connect(
                lambda value, name=param_name: self.param_changed.emit(name, value)
            )
            params_layout.addRow(label_text, spinbox)
            self.spinboxes[param_name] = spinbox

        main_layout.addWidget(params_container)
        main_layout.addStretch()

        self.button_finish = QPushButton("Train Agent")
        self.button_finish.setProperty("class", "navigation")
        main_layout.addWidget(
            self.button_finish, alignment=Qt.AlignmentFlag.AlignCenter
        )

        if self.agent_types:
            self.buttons[self.agent_types[0]].setChecked(True)
            self.select_agent(self.agent_types[0])

    def select_agent(self, agent_name: str):
        self.agent_type_changed.emit(agent_name)


class TrainingScreen(QWidget):
    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(30)
        title = QLabel("Training Agent")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.spinner = LoadingSpinner(self, speed=4, smoothness_ms=30)
        self.status_label = QLabel("Preparing to start training...")
        self.status_label.setObjectName("trainingStatusLabel")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        layout.addStretch()
        layout.addWidget(self.spinner, 0, Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        layout.addStretch()

        self.thread: Optional[QThread] = None
        self.worker: Optional[TrainingWorker] = None

    def start_worker(self):
        self.status_label.setText("Training in progress... Please wait.")
        self.spinner.start()
        self.thread = QThread()
        self.worker = TrainingWorker(self.main_window.data)
        self.worker.moveToThread(self.thread)
        self.thread.started.connect(self.worker.run)
        self.worker.training_finished.connect(self.spinner.stop)
        self.worker.training_finished.connect(self.main_window.on_training_complete)
        self.worker.training_finished.connect(self.thread.quit)
        self.worker.training_finished.connect(self.worker.deleteLater)
        self.thread.finished.connect(self.thread.deleteLater)
        self.thread.start()


class InteractiveResultsPage(QWidget):
    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        self.main_window = main_window
        self.selected_start_pos: Optional[Tuple[int, int]] = None
        self.current_path: Optional[List[Tuple[int, int, int]]] = None

        main_layout = QVBoxLayout(self)
        title = QLabel("Learned Policy Visualization")
        title.setObjectName("titleLabel")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        main_layout.addWidget(title)
        content_layout = QHBoxLayout()
        main_layout.addLayout(content_layout, 1)

        controls_container = QWidget()
        controls_container.setObjectName("controlsContainer")
        controls_container.setFixedWidth(350)
        controls_layout = QVBoxLayout(controls_container)
        controls_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        content_layout.addWidget(controls_container)

        self.instruction_label = QLabel("Click on the grid to select a start point.")
        self.instruction_label.setObjectName("instructionLabel")
        self.instruction_label.setWordWrap(True)
        self.instruction_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

        self.start_compass = CompassWidget("Select Start Orientation")
        self.visualize_button = QPushButton("Visualize Path")
        self.visualize_button.setProperty("class", "navigation")

        # --- NEW BUTTON: ADD OBSTACLES ---
        self.add_obs_button = QPushButton("Add Obstacles")
        self.add_obs_button.setCheckable(True)
        self.add_obs_button.setProperty(
            "class", "modeButton"
        )  # Reuse existing class for consistent look

        export_controls_container = QWidget()
        export_controls_container.setObjectName("exportControlsContainer")
        export_layout = QVBoxLayout(export_controls_container)
        export_layout.setSpacing(15)

        resolution_form = QFormLayout()
        self.resolution_spinbox = QDoubleSpinBox()
        self.resolution_spinbox.setRange(0.01, 100.0)
        self.resolution_spinbox.setValue(1.0)
        self.resolution_spinbox.setSingleStep(0.1)
        self.resolution_spinbox.setSuffix(" m")
        resolution_form.addRow("Resolution (m/cell):", self.resolution_spinbox)

        self.save_path_button = QPushButton("Save Path")
        self.save_path_button.setObjectName("savePathButton")
        export_layout.addLayout(resolution_form)
        export_layout.addWidget(
            self.save_path_button, alignment=Qt.AlignmentFlag.AlignCenter
        )

        colors = ["#e74c3c", "#2ecc71", "#3498db", "#fffb00"]
        self.cmap = ListedColormap(colors)

        controls_layout.addStretch(1)
        controls_layout.addWidget(self.instruction_label)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.start_compass)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(self.add_obs_button)  # Added button to layout
        controls_layout.addWidget(self.visualize_button)
        controls_layout.addSpacing(20)
        controls_layout.addWidget(export_controls_container)
        controls_layout.addStretch(1)

        self.fig: Figure = Figure(figsize=(8, 8), tight_layout=True)
        self.canvas: FigureCanvas = FigureCanvas(self.fig)
        self.ax: Axes = self.fig.add_subplot(111)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        content_layout.addWidget(self.canvas, 1)

        self.button_finish = QPushButton("Finish Visualization")
        self.button_finish.setProperty("class", "navigation")
        main_layout.addWidget(
            self.button_finish, alignment=Qt.AlignmentFlag.AlignCenter
        )

        self.visualize_button.clicked.connect(self.plot_greedy_path_from_selection)
        self.save_path_button.clicked.connect(self.save_path_to_file)
        self.canvas.mpl_connect("button_press_event", self.on_canvas_click)

    def setup_page(self):
        env = self.main_window.trained_env
        if not env:
            return
        self.current_path = None
        start_x, start_y, start_k = env._start
        self.selected_start_pos = (start_x, start_y)
        self.start_compass.set_orientation(start_k)
        self.instruction_label.setText(
            f"Start point selected at (X={start_x}, Y={start_y}).\n"
            "Click grid to change or visualize path."
        )
        self.draw_base_grid()
        self.draw_path_from_point(self.selected_start_pos, start_k)

    def draw_base_grid(self):
        self.ax.clear()
        env = self.main_window.trained_env
        if not env:
            self.ax.text(
                0.5, 0.5, "Error: Training data not found.", ha="center", va="center"
            )
            self.canvas.draw()
            return

        W, H = env._x_size, env._y_size
        obs_grid_for_plot = env._obs_grid.T
        goal_x, goal_y, _ = env._goal
        bg = np.ones((H, W, 3))
        bg[obs_grid_for_plot == 1] = (0.2, 0.2, 0.2)
        self.ax.imshow(bg, origin="lower", interpolation="nearest")
        self.ax.set_xticks(np.arange(-0.5, W, 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, H, 1), minor=True)
        self.ax.grid(which="minor", color="k", linestyle="-", linewidth=0.5, alpha=0.2)
        self.ax.tick_params(which="minor", size=0)
        self.ax.tick_params(
            axis="x", which="major", bottom=False, top=False, labelbottom=False
        )
        self.ax.tick_params(
            axis="y", which="major", left=False, right=False, labelleft=False
        )
        self.ax.set_xlim([-0.5, W - 0.5])
        self.ax.set_ylim([-0.5, H - 0.5])

        # Update Title based on mode
        if self.add_obs_button.isChecked():
            self.ax.set_title("Mode: ADDING OBSTACLES (Click to place/remove)")
        else:
            self.ax.set_title("Click a cell to select a start, then visualize")

        self.ax.set_xlabel("Columns (X)")
        self.ax.set_ylabel("Rows (Y)")
        legend_elements = [
            Patch(facecolor=self.cmap.colors[1], edgecolor="black", label="Start"),
            Patch(facecolor=self.cmap.colors[0], edgecolor="black", label="Goal"),
        ]
        self.ax.legend(
            handles=legend_elements,
            bbox_to_anchor=(1.04, 1),
            loc="upper left",
            borderaxespad=0.0,
        )
        self.ax.scatter(
            goal_x,
            goal_y,
            marker="*",
            c=self.cmap.colors[0],
            s=250,
            zorder=5,
            label="Goal",
            edgecolors="black",
        )
        self.canvas.draw()

    def on_canvas_click(self, event: MouseEvent):
        if not event.inaxes:
            return

        env = self.main_window.trained_env
        if not env:
            return

        c, r = int(round(float(event.xdata))), int(round(float(event.ydata)))
        W, H = env._x_size, env._y_size

        if not (0 <= c < W and 0 <= r < H):
            return

        # --- LOGIC FOR ADDING OBSTACLES ---
        if self.add_obs_button.isChecked():
            # Check if trying to overwrite goal
            goal_x, goal_y, _ = env._goal
            if (c, r) == (goal_x, goal_y):
                QMessageBox.warning(
                    self, "Invalid Operation", "Cannot place obstacle on Goal."
                )
                return

            # Toggle obstacle
            env._obs_grid[c, r] = 1 - env._obs_grid[c, r]

            # IMPORTANT: Recompute safety matrix so the agent "sees" the new wall
            env._nearby_obs_grid = env._precompute_safety_penalty_matrix(min_dist=2.0)

            # Clear current path visualization as it might be invalid now
            self.current_path = None

            self.draw_base_grid()

            # If start pos still exists, redraw it
            if self.selected_start_pos:
                sx, sy = self.selected_start_pos
                self.ax.scatter(
                    sx,
                    sy,
                    marker="o",
                    c=self.cmap.colors[1],
                    s=150,
                    zorder=5,
                    label="Selected Start",
                    edgecolors="black",
                )
                self.canvas.draw()
            return

        # --- ORIGINAL LOGIC (SELECT START) ---
        self.current_path = None
        obs_grid = env._obs_grid

        if obs_grid[c, r] == 1:
            QMessageBox.warning(
                self, "Invalid Start", "The selected cell is an obstacle."
            )
            return

        self.selected_start_pos = (c, r)
        self.instruction_label.setText(
            f"Start point selected at (X={c}, Y={r}).\n"
            "Adjust orientation and visualize."
        )
        self.draw_base_grid()
        self.ax.scatter(
            c,
            r,
            marker="o",
            c=self.cmap.colors[1],
            s=150,
            zorder=5,
            label="Selected Start",
            edgecolors="black",
        )
        self.canvas.draw()

    def plot_greedy_path_from_selection(self):
        if self.selected_start_pos is None:
            QMessageBox.warning(
                self,
                "No Start Point",
                "Please click on the grid to select a starting point first.",
            )
            return
        start_k = self.start_compass.current_orientation
        self.draw_path_from_point(self.selected_start_pos, start_k)

    def draw_path_from_point(self, start_pos: Tuple[int, int], start_orientation: int):
        env = self.main_window.trained_env
        agent = self.main_window.trained_agent
        if not env or not agent:
            QMessageBox.critical(self, "Error", "Training data not found.")
            return

        start_c, start_r = start_pos
        start_k = start_orientation
        self.draw_base_grid()
        eps_bak = agent.epsilon
        agent.epsilon = 0.0

        env._state = (start_c, start_r, start_k)
        s = env._state
        path: List[Tuple[int, int, int]] = [s]
        limit = env._x_size * env._y_size * 2
        for i in range(limit):
            a = agent.choose_action(s)
            s, _, done = env.step(a)
            path.append(s)
            if done:
                break
            if i == limit - 1:
                QMessageBox.warning(
                    self,
                    "Path Limit Reached",
                    "The path limit was reached without finding a solution.",
                )
                break
        self.current_path = path
        agent.epsilon = eps_bak

        self.ax.scatter(
            start_c,
            start_r,
            marker="o",
            c=self.cmap.colors[1],
            s=150,
            zorder=5,
            label="Selected Start",
            edgecolors="black",
        )
        goal_x, goal_y, _ = env._goal
        obs_grid_T = env._obs_grid.T

        for px, py, _ in path:
            if obs_grid_T[py, px] == 0 and (px, py) != (goal_x, goal_y):
                self.ax.add_patch(
                    plt.Rectangle(
                        (px - 0.5, py - 0.5),
                        1,
                        1,
                        fill=True,
                        alpha=0.3,
                        color=self.cmap.colors[2],
                        zorder=3,
                    )
                )

        xs, ys, us, vs = [], [], [], []
        arrow_scale = 0.35
        for px, py, kk in path:
            ang = env._index2rad(kk)
            xs.append(px)
            ys.append(py)
            us.append(math.cos(ang) * arrow_scale)
            vs.append(math.sin(ang) * arrow_scale)
        if xs:
            self.ax.quiver(
                xs,
                ys,
                us,
                vs,
                angles="xy",
                scale_units="xy",
                scale=1,
                width=0.015,
                color=self.cmap.colors[3],
                zorder=6,
            )

        self.ax.set_title(
            f"Greedy Path from (X={start_c}, Y={start_r}, θ={start_k*45}°) | Steps: {len(path)-1}"
        )
        self.canvas.draw()

    def save_path_to_file(self):
        if not self.current_path:
            QMessageBox.warning(
                self,
                "No Path Available",
                "Please visualize a path first before saving.",
            )
            return
        env = self.main_window.trained_env
        if not env:
            QMessageBox.critical(self, "Error", "Environment data not found.")
            return
        resolution = self.resolution_spinbox.value()
        metric_path = []
        for c, r, k in self.current_path:
            x_meters, y_meters = c * resolution, r * resolution
            orientation_radians = env._index2rad(k)
            metric_path.append((x_meters, y_meters, orientation_radians))

        path_array = np.array(metric_path, dtype=float)
        settings = QSettings("MyRLApp", "PathSaver")
        last_dir = settings.value("last_path_save_dir", "")

        file_path, _ = QFileDialog.getSaveFileName(
            self, "Save Path", str(last_dir), "Numpy files (*.npy)"
        )
        if file_path:
            try:
                np.save(file_path, path_array)
                directory = os.path.dirname(file_path)
                settings.setValue("last_path_save_dir", directory)
                QMessageBox.information(
                    self, "Success", f"Path successfully saved to:\n{file_path}"
                )
            except Exception as e:
                QMessageBox.critical(self, "Error", f"Failed to save path.\nError: {e}")


class EndPage(QWidget):
    def __init__(self, main_window: QMainWindow):
        super().__init__(main_window)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(20)

        label = QLabel("Process Finished")
        label.setObjectName("titleLabel")

        self.button_new_training = QPushButton("New Training")
        self.button_new_training.setProperty("class", "navigation")

        self.button_close = QPushButton("Close Window")
        self.button_close.setObjectName("closeButton")

        layout.addWidget(label)
        layout.addWidget(self.button_new_training)
        layout.addWidget(self.button_close)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_window = MainWindow()
    main_window.show()
    sys.exit(app.exec())
