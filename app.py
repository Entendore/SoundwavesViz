import sys
import numpy as np
import sounddevice as sd
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
    QSlider, QLabel, QDockWidget, QPushButton, QComboBox, QGridLayout, 
    QFileDialog, QGroupBox, QCheckBox
)
from PySide6.QtCore import Qt, QTimer, QElapsedTimer
import pyqtgraph as pg
import pyqtgraph.opengl as gl

class SoundWaveVisualizer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Spatial Acoustic Simulator & Interactive Audio")
        self.resize(1600, 900)

        pg.setConfigOptions(antialias=True)
        app = QApplication.instance()
        app.setStyleSheet("""
            QMainWindow { background-color: #121212; }
            QWidget { background-color: #1e1e1e; color: #e0e0e0; font-size: 10pt; }
            QDockWidget { titlebar-close-icon: none; titlebar-normal-icon: none; }
            QDockWidget::title { text-align: center; background: #252525; padding: 6px; }
            QGroupBox { border: 1px solid #444; border-radius: 6px; margin-top: 12px; font-weight: bold; }
            QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
            QPushButton { background-color: #0078d7; border: none; padding: 6px 12px; border-radius: 4px; }
            QPushButton:hover { background-color: #006cbd; }
            QPushButton:pressed { background-color: #005a9e; }
            QPushButton:checked { background-color: #00aaff; }
            QComboBox { border: 1px solid #444; padding: 4px; border-radius: 4px; }
            QCheckBox { spacing: 8px; }
            QSlider::groove:horizontal { border: 1px solid #444; height: 6px; background: #2e2e2e; border-radius: 3px; }
            QSlider::handle:horizontal { background: #00aaff; border: 2px solid #00aaff; width: 12px; margin: -5px 0; border-radius: 7px; }
            QLabel { font-weight: normal; }
        """)

        # State variables
        self.t = 0.0
        self.is_playing = True
        self.amplitude = 1.0
        self.frequency = 1.0
        self.speed = 1.0
        self.damping = 0.2
        self.phase = 0.0
        self.steering_deg = 0.0
        self.wave_type = "Radial"
        self.grid_size = 80
        self.use_db_scale = False
        self.show_particles = True
        
        # Audio state
        self.audio_stream = None
        self.audio_phase = 0.0
        self.audio_waveform = "Sine"
        self.listener_pos = np.array([2.0, 2.0]) # Default Mic position
        
        self.cmap_3d = pg.colormap.get('plasma')
        self.cmap_2d_linear = pg.colormap.get('coolwarm')
        self.cmap_2d_db = pg.colormap.get('inferno')

        self._setup_ui()
        self._setup_controls()
        self.init_mesh_data()
        self.update_params()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(16)
        
        self.fps_timer = QElapsedTimer()
        self.fps_timer.start()
        self.frame_count = 0
        self.last_fps_time = 0

    def _setup_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        central_widget.setMaximumHeight(200)
        
        # --- 1. 3D View Dock ---
        self.dock_3d = QDockWidget("3D Acoustic Surface", self)
        self.dock_3d.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.view_3d = gl.GLViewWidget()
        self.view_3d.setBackgroundColor('#1a1a1a')
        self.dock_3d.setWidget(self.view_3d)
        self.addDockWidget(Qt.LeftDockWidgetArea, self.dock_3d)
        
        self.surf_3d = gl.GLSurfacePlotItem(shader='normalColor', smooth=True, drawEdges=False)
        self.view_3d.addItem(self.surf_3d)
        
        self.emitter_scatter = gl.GLScatterPlotItem(pos=np.array([[0,0,0]]), size=10, color=(1,1,0,1))
        self.view_3d.addItem(self.emitter_scatter)
        
        # 3D Listener indicator
        self.listener_scatter_3d = gl.GLScatterPlotItem(pos=np.array([[2,2,0]]), size=12, color=(0,1,0,1))
        self.view_3d.addItem(self.listener_scatter_3d)
        
        self.view_3d.opts['distance'] = 15
        self.view_3d.opts['elevation'] = 30

        # --- 2. 2D Cross-Section Dock ---
        self.dock_2d = QDockWidget("2D Cross-Section (Slice)", self)
        self.dock_2d.setAllowedAreas(Qt.AllDockWidgetAreas)
        self.plot_2d = pg.PlotWidget()
        self.plot_2d.setBackground('#1a1a1a')
        self.plot_2d.setLabel('left', 'Pressure Amplitude')
        self.plot_2d.setLabel('bottom', 'Distance (X)')
        self.plot_2d.showGrid(x=True, y=True, alpha=0.3)
        self.plot_2d.setYRange(-2.5, 2.5)
        self.dock_2d.setWidget(self.plot_2d)
        self.addDockWidget(Qt.RightDockWidgetArea, self.dock_2d)

        self.fill_item = pg.FillBetweenItem(pg.PlotDataItem(), pg.PlotDataItem(), brush=pg.mkBrush(0, 170, 255, 50))
        self.plot_2d.addItem(self.fill_item)
        self.line_2d = self.plot_2d.plot([], [], pen=pg.mkPen(color='c', width=2))
        
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(color='y', style=Qt.DashLine, width=1))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(color='y', style=Qt.DashLine, width=1))
        self.plot_2d.addItem(self.vLine, ignoreBounds=True)
        self.plot_2d.addItem(self.hLine, ignoreBounds=True)
        self.crosshair_label = pg.TextItem(text="", color='y', anchor=(0,1))
        self.plot_2d.addItem(self.crosshair_label)
        self.plot_2d.scene().sigMouseMoved.connect(self.mouse_moved_2d)

        # --- 3. Top-Down Heatmap Dock ---
        self.dock_heat = QDockWidget("Top-Down Intensity Map (Click to move Mic)", self)
        self.dock_heat.setAllowedAreas(Qt.AllDockWidgetAreas)
        
        heat_widget = QWidget()
        heat_layout = QHBoxLayout(heat_widget)
        heat_layout.setContentsMargins(0,0,0,0)
        
        self.plot_heat = pg.PlotWidget()
        self.plot_heat.setBackground('#1a1a1a')
        self.plot_heat.setLabel('bottom', 'Distance X')
        self.plot_heat.setLabel('left', 'Distance Y')
        heat_layout.addWidget(self.plot_heat, stretch=1)
        
        self.heat_img = pg.ImageItem()
        self.plot_heat.addItem(self.heat_img)
        self.cbar = pg.ColorBarItem(values=(-1, 1), cmap=self.cmap_2d_linear, interactive=True)
        self.cbar.setImageItem(self.heat_img, insert_in=self.plot_heat)
        
        self.contour = pg.IsocurveItem(level=0.0, pen='w')
        self.plot_heat.addItem(self.contour)
        
        self.particle_scatter = pg.ScatterPlotItem(size=3, color='w')
        self.plot_heat.addItem(self.particle_scatter)
        
        # Interactive Listener Marker (Microphone)
        self.listener_item = pg.ScatterPlotItem(
            pos=[self.listener_pos], size=15, pen=pg.mkPen('w', width=2), brush=pg.mkBrush('g')
        )
        self.plot_heat.addItem(self.listener_item)
        
        # Connect click event to move mic
        self.plot_heat.scene().sigMouseClicked.connect(self.mouse_clicked_heat)
        
        self.plot_heat.setAspectLocked(True)
        self.plot_heat.setRange(xRange=[-5, 5], yRange=[-5, 5])
        
        self.dock_heat.setWidget(heat_widget)
        self.splitDockWidget(self.dock_2d, self.dock_heat, Qt.Vertical)

    def _setup_controls(self):
        central_widget = self.centralWidget()
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # --- Group 1: Wave Physics ---
        grp_physics = QGroupBox("Wave Physics")
        l1 = QGridLayout(grp_physics)
        
        l1.addWidget(QLabel("Wave Type:"), 0, 0)
        self.combo_wave = QComboBox()
        self.combo_wave.addItems(["Radial", "Plane", "Interference", "Array (Beamforming)", "Pulse (Traveling)"])
        self.combo_wave.currentTextChanged.connect(self.update_params)
        l1.addWidget(self.combo_wave, 0, 1)

        self.amp_slider = self._make_slider(10, 200, 100, "Amplitude of the sound pressure.")
        self.amp_label = QLabel("1.00")
        l1.addWidget(QLabel("Amplitude:"), 1, 0)
        l1.addWidget(self.amp_slider, 1, 1)
        l1.addWidget(self.amp_label, 1, 2)

        self.freq_slider = self._make_slider(10, 500, 100, "Frequency of the wave. Also controls audio pitch.")
        self.freq_label = QLabel("1.00")
        l1.addWidget(QLabel("Frequency:"), 2, 0)
        l1.addWidget(self.freq_slider, 2, 1)
        l1.addWidget(self.freq_label, 2, 2)

        self.damp_slider = self._make_slider(0, 100, 20, "How fast the wave loses energy over distance.")
        self.damp_label = QLabel("0.20")
        l1.addWidget(QLabel("Damping:"), 3, 0)
        l1.addWidget(self.damp_slider, 3, 1)
        l1.addWidget(self.damp_label, 3, 2)

        self.steer_slider = self._make_slider(-45, 45, 0, "Steers the Array beam left/right.")
        self.steer_label = QLabel("0°")
        self.lbl_steer = QLabel("Steering Angle:")
        l1.addWidget(self.lbl_steer, 4, 0)
        l1.addWidget(self.steer_slider, 4, 1)
        l1.addWidget(self.steer_label, 4, 2)

        main_layout.addWidget(grp_physics)

        # --- Group 2: Simulation ---
        grp_sim = QGroupBox("Simulation")
        l2 = QGridLayout(grp_sim)

        self.speed_slider = self._make_slider(0, 300, 100, "Speed of the animation.")
        self.speed_label = QLabel("1.00")
        l2.addWidget(QLabel("Speed:"), 0, 0)
        l2.addWidget(self.speed_slider, 0, 1)
        l2.addWidget(self.speed_label, 0, 2)

        self.phase_slider = self._make_slider(0, 360, 0, "Phase shift of the wave.")
        self.phase_label = QLabel("0°")
        l2.addWidget(QLabel("Phase Shift:"), 1, 0)
        l2.addWidget(self.phase_slider, 1, 1)
        l2.addWidget(self.phase_label, 1, 2)

        self.res_slider = QSlider(Qt.Orientation.Horizontal)
        self.res_slider.setRange(30, 150)
        self.res_slider.setValue(80)
        self.res_slider.valueChanged.connect(self.update_resolution)
        self.res_label = QLabel("80")
        l2.addWidget(QLabel("Mesh Res:"), 2, 0)
        l2.addWidget(self.res_slider, 2, 1)
        l2.addWidget(self.res_label, 2, 2)

        self.slice_slider = QSlider(Qt.Orientation.Horizontal)
        self.slice_slider.setRange(0, self.grid_size - 1)
        self.slice_slider.setValue(self.grid_size // 2)
        self.slice_slider.valueChanged.connect(self.update_params)
        self.slice_label = QLabel("Y = 0.00")
        l2.addWidget(QLabel("Slice Y:"), 3, 0)
        l2.addWidget(self.slice_slider, 3, 1)
        l2.addWidget(self.slice_label, 3, 2)

        main_layout.addWidget(grp_sim)

        # --- Group 3: Display & Interactive Audio ---
        grp_disp = QGroupBox("Display & Interactive Audio")
        l3 = QVBoxLayout(grp_disp)

        # Audio Waveform Selector
        audio_layout = QHBoxLayout()
        audio_layout.addWidget(QLabel("Audio Wave:"))
        self.combo_audio_wave = QComboBox()
        self.combo_audio_wave.addItems(["Sine", "Square", "Sawtooth", "Triangle"])
        self.combo_audio_wave.currentTextChanged.connect(self.update_params)
        audio_layout.addWidget(self.combo_audio_wave)
        l3.addLayout(audio_layout)

        self.chk_db = QCheckBox("View Intensity in Decibels (dB)")
        self.chk_db.stateChanged.connect(self.update_params)
        l3.addWidget(self.chk_db)

        self.chk_particles = QCheckBox("Show Air Molecule Displacement")
        self.chk_particles.setChecked(True)
        self.chk_particles.stateChanged.connect(self.update_params)
        l3.addWidget(self.chk_particles)

        btn_layout = QHBoxLayout()
        self.btn_play = QPushButton("Pause")
        self.btn_play.clicked.connect(self.toggle_play)
        self.btn_reset = QPushButton("Reset Time")
        self.btn_reset.clicked.connect(self.reset_time)
        self.btn_reset_cam = QPushButton("Reset 3D Cam")
        self.btn_reset_cam.clicked.connect(self.reset_camera)
        btn_layout.addWidget(self.btn_play)
        btn_layout.addWidget(self.btn_reset)
        btn_layout.addWidget(self.btn_reset_cam)
        l3.addLayout(btn_layout)

        export_layout = QHBoxLayout()
        self.btn_audio = QPushButton("Play Spatial Audio")
        self.btn_audio.setCheckable(True)
        self.btn_audio.clicked.connect(self.toggle_audio)
        self.btn_save_3d = QPushButton("Save 3D PNG")
        self.btn_save_3d.clicked.connect(lambda: self.save_snapshot(True))
        self.btn_save_csv = QPushButton("Export CSV")
        self.btn_save_csv.clicked.connect(self.export_csv)
        export_layout.addWidget(self.btn_audio)
        export_layout.addWidget(self.btn_save_3d)
        export_layout.addWidget(self.btn_save_csv)
        l3.addLayout(export_layout)
        
        main_layout.addWidget(grp_disp)

        self.status_bar = self.statusBar()
        self.mic_label = QLabel("Mic Pos: (0.00, 0.00)")
        self.status_bar.addWidget(self.mic_label)
        self.fps_label = QLabel("FPS: 0")
        self.status_bar.addPermanentWidget(self.fps_label)

    def init_mesh_data(self):
        x = np.linspace(-5, 5, self.grid_size)
        y = np.linspace(-5, 5, self.grid_size)
        self.x, self.y = x, y
        self.X, self.Y = np.meshgrid(x, y)
        self.R = np.sqrt(self.X**2 + self.Y**2)
        self.surf_3d.setData(x=self.x, y=self.y, z=np.zeros_like(self.X))
        
        step = max(1, self.grid_size // 20)
        self.particle_base_X = self.X[::step, ::step].flatten()
        self.particle_base_Y = self.Y[::step, ::step].flatten()
        self.particle_idx_x = np.arange(0, self.grid_size, step)
        self.particle_idx_y = np.arange(0, self.grid_size, step)
        self.particle_mesh_X, self.particle_mesh_Y = np.meshgrid(self.particle_idx_x, self.particle_idx_y)

    def _make_slider(self, min_val, max_val, init_val, tooltip=""):
        s = QSlider(Qt.Orientation.Horizontal)
        s.setRange(min_val, max_val)
        s.setValue(init_val)
        s.valueChanged.connect(self.update_params)
        s.setToolTip(tooltip)
        return s

    def update_resolution(self):
        self.res_label.setText(str(self.res_slider.value()))
        if not self.res_slider.isSliderDown():
            self.grid_size = self.res_slider.value()
            self.init_mesh_data()
            self.slice_slider.setRange(0, self.grid_size - 1)
            self.slice_slider.setValue(self.grid_size // 2)
            self.update_params()

    def update_params(self):
        self.wave_type = self.combo_wave.currentText()
        self.amplitude = self.amp_slider.value() / 100.0
        self.frequency = self.freq_slider.value() / 100.0
        self.speed = self.speed_slider.value() / 100.0
        self.damping = self.damp_slider.value() / 100.0
        self.phase = self.phase_slider.value() * (np.pi / 180.0)
        self.steering_deg = self.steer_slider.value()
        self.use_db_scale = self.chk_db.isChecked()
        self.show_particles = self.chk_particles.isChecked()
        self.audio_waveform = self.combo_audio_wave.currentText()

        self.amp_label.setText(f"{self.amplitude:.2f}")
        self.freq_label.setText(f"{self.frequency:.2f}")
        self.speed_label.setText(f"{self.speed:.2f}")
        self.damp_label.setText(f"{self.damping:.2f}")
        self.phase_label.setText(f"{np.degrees(self.phase):.0f}°")
        self.steer_label.setText(f"{self.steering_deg}°")
        
        slice_idx = self.slice_slider.value()
        y_val = self.y[slice_idx]
        self.slice_label.setText(f"Y = {y_val:.2f}")

        is_array = self.wave_type.startswith("Array")
        self.lbl_steer.setVisible(is_array)
        self.steer_slider.setVisible(is_array)
        self.steer_label.setVisible(is_array)

    def mouse_moved_2d(self, pos):
        if self.plot_2d.plotItem.vb.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_2d.plotItem.vb.mapSceneToView(pos)
            self.vLine.setPos(mousePoint.x())
            self.hLine.setPos(mousePoint.y())
            self.crosshair_label.setPos(mousePoint.x(), mousePoint.y())
            self.crosshair_label.setText(f"X: {mousePoint.x():.2f}\nZ: {mousePoint.y():.2f}")

    def mouse_clicked_heat(self, event):
        # Move microphone on click
        pos = event.scenePos()
        if self.plot_heat.sceneBoundingRect().contains(pos):
            mousePoint = self.plot_heat.plotItem.vb.mapSceneToView(pos)
            self.listener_pos = np.array([mousePoint.x(), mousePoint.y()])
            self.listener_item.setData(pos=[self.listener_pos])
            self.listener_scatter_3d.setData(pos=np.array([[self.listener_pos[0], self.listener_pos[1], 0]]))
            self.mic_label.setText(f"Mic Pos: ({self.listener_pos[0]:.2f}, {self.listener_pos[1]:.2f})")

    def toggle_play(self):
        self.is_playing = not self.is_playing
        self.btn_play.setText("Play" if not self.is_playing else "Pause")

    def reset_time(self):
        self.t = 0.0

    def reset_camera(self):
        self.view_3d.opts['distance'] = 15
        self.view_3d.opts['elevation'] = 30
        self.view_3d.opts['azimuth'] = 0
        self.view_3d.update()

    def toggle_audio(self, checked):
        if checked:
            try:
                # 2 channels for Stereo panning
                self.audio_stream = sd.OutputStream(samplerate=44100, channels=2, callback=self.audio_callback)
                self.audio_stream.start()
                self.btn_audio.setText("Stop Spatial Audio")
            except Exception as e:
                self.btn_audio.setChecked(False)
                self.btn_audio.setText("Play Spatial Audio")
                self.status_bar.showMessage(f"Audio Error: {e}", 5000)
        else:
            if self.audio_stream:
                self.audio_stream.stop()
                self.audio_stream.close()
                self.audio_stream = None
            self.btn_audio.setText("Play Spatial Audio")

    def audio_callback(self, outdata, frames, time, status):
        if not self.audio_stream: return
        
        t_audio = (np.arange(frames) + self.audio_phase) / 44100.0
        f_aud = 100.0 + (self.frequency * 180.0)
        phase_audio = 2 * np.pi * f_aud * t_audio
        
        emitters = self.get_emitter_locations()
        lx, ly = self.listener_pos
        
        left_ch = np.zeros(frames)
        right_ch = np.zeros(frames)
        
        for ex, ey, ez in emitters:
            # Distance from emitter to listener
            R = np.sqrt((lx - ex)**2 + (ly - ey)**2)
            
            # Spatial phase delay (maps visual frequency to audio phase)
            k_vis = 2 * np.pi * self.frequency
            spatial_phase = k_vis * R
            
            # Beamforming phase delay
            extra_phase = 0.0
            if self.wave_type.startswith("Array"):
                k = 2 * np.pi * self.frequency
                theta_rad = np.radians(self.steering_deg)
                extra_phase = k * ey * np.sin(theta_rad)
                
            total_phase = phase_audio + spatial_phase - extra_phase
            
            # Amplitude damping based on distance
            A = 1.0 / (R + 0.1)
            decay = A ** self.damping
            
            # Generate Waveform
            if self.audio_waveform == "Sine":
                wave = np.sin(total_phase)
            elif self.audio_waveform == "Square":
                wave = np.sign(np.sin(total_phase))
            elif self.audio_waveform == "Sawtooth":
                wave = 2.0 * (total_phase / (2*np.pi) - np.floor(0.5 + total_phase / (2*np.pi)))
            elif self.audio_waveform == "Triangle":
                wave = 2.0 * np.abs(2.0 * (total_phase / (2*np.pi) - np.floor(0.5 + total_phase / (2*np.pi)))) - 1.0
                
            val = decay * wave
            
            # Stereo Panning based on relative X position
            # Pan goes from -1 (left) to 1 (right)
            pan = np.clip((lx - ex) / 5.0, -1.0, 1.0)
            left_gain = 0.5 * (1.0 - pan)
            right_gain = 0.5 * (1.0 + pan)
            
            left_ch += val * left_gain
            right_ch += val * right_gain
            
        # Normalize to prevent clipping
        max_val = max(np.max(np.abs(left_ch)), np.max(np.abs(right_ch)), 1.0)
        left_ch = (left_ch / max_val) * 0.3 * self.amplitude
        right_ch = (right_ch / max_val) * 0.3 * self.amplitude
        
        outdata[:, 0] = left_ch
        outdata[:, 1] = right_ch
        
        self.audio_phase += frames

    def save_snapshot(self, is_3d):
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("PNG Files (*.png)")
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                if is_3d:
                    img = self.view_3d.grabFramebuffer()
                    img.save(file_paths[0])

    def export_csv(self):
        file_dialog = QFileDialog(self)
        file_dialog.setAcceptMode(QFileDialog.AcceptSave)
        file_dialog.setNameFilter("CSV Files (*.csv)")
        if file_dialog.exec():
            file_paths = file_dialog.selectedFiles()
            if file_paths:
                slice_idx = self.slice_slider.value()
                x_2d = self.X[:, slice_idx]
                z_2d = self.calculate_wave(self.X, self.Y, self.R, self.t)[:, slice_idx]
                data = np.column_stack((x_2d, z_2d))
                np.savetxt(file_paths[0], data, delimiter=",", header="Distance_X,Pressure_Z", comments="")
                self.status_bar.showMessage(f"Exported to {file_paths[0]}", 3000)

    def get_emitter_locations(self):
        if self.wave_type.startswith("Array"):
            return np.array([[-4.0, y, 0] for y in [-1.0, -0.5, 0.0, 0.5, 1.0]])
        elif self.wave_type == "Interference":
            return np.array([[-2.0, 0, 0], [2.0, 0, 0]])
        elif self.wave_type == "Plane" or self.wave_type.startswith("Pulse"):
            return np.array([[-5.0, 0, 0]])
        else:
            return np.array([[0.0, 0, 0]])

    def calculate_wave(self, X, Y, R, t):
        eps = 0.1 

        if self.wave_type == "Radial":
            decay = 1.0 / (R + eps)
            z = self.amplitude * np.sin(2 * np.pi * self.frequency * R - t + self.phase) * (decay ** self.damping)
            
        elif self.wave_type == "Plane":
            decay = 1.0 / (np.abs(X) + eps)
            z = self.amplitude * np.sin(2 * np.pi * self.frequency * X - t + self.phase) * (decay ** self.damping)
            
        elif self.wave_type == "Interference":
            R1 = np.sqrt((X + 2.0)**2 + Y**2)
            R2 = np.sqrt((X - 2.0)**2 + Y**2)
            decay1 = 1.0 / (R1 + eps)
            decay2 = 1.0 / (R2 + eps)
            w1 = self.amplitude * np.sin(2 * np.pi * self.frequency * R1 - t + self.phase) * (decay1 ** self.damping)
            w2 = self.amplitude * np.sin(2 * np.pi * self.frequency * R2 - t + self.phase) * (decay2 ** self.damping)
            z = w1 + w2
            
        elif self.wave_type.startswith("Array"):
            z = np.zeros_like(X)
            theta_rad = np.radians(self.steering_deg)
            array_y = np.array([-1.0, -0.5, 0.0, 0.5, 1.0])
            array_x = -4.0
            k = 2 * np.pi * self.frequency
            phase_delays = k * array_y * np.sin(theta_rad)
            
            for i, y_pos in enumerate(array_y):
                R_arr = np.sqrt((X - array_x)**2 + (Y - y_pos)**2)
                decay_arr = 1.0 / (R_arr + eps)
                z += self.amplitude * np.sin(2 * np.pi * self.frequency * R_arr - t + self.phase - phase_delays[i]) * (decay_arr ** self.damping)
            z = z / 3.0 

        elif self.wave_type == "Pulse (Traveling)":
            c = 3.0 
            t_wrapped = t % (10.0 / c)
            pulse_center = -5.0 + c * t_wrapped
            sigma = 0.8
            envelope = np.exp(-((X - pulse_center)**2) / (2 * sigma**2))
            sine_wave = np.sin(2 * np.pi * self.frequency * X - t + self.phase)
            z = self.amplitude * sine_wave * envelope
            
        return z

    def update_frame(self):
        if self.is_playing:
            self.t += 0.15 * self.speed

        Z_linear = self.calculate_wave(self.X, self.Y, self.R, self.t)
        Z_display = Z_linear.copy()
        max_val = self.amplitude * 2.5
        
        if self.use_db_scale:
            Z_display = 10 * np.log10(Z_linear**2 + 1e-9)
            max_val = 10 * np.log10((self.amplitude * 2.5)**2)
            Z_display = np.clip(Z_display, max_val - 40, max_val)

        # --- Update 3D Surface ---
        norm_z = np.clip((Z_display + max_val) / (2 * max_val), 0, 1)
        colors_3d = self.cmap_3d.map(norm_z.flatten(), mode='float')
        self.surf_3d.setData(z=Z_display, colors=colors_3d)
        
        emitters = self.get_emitter_locations()
        pulse = np.abs(np.sin(-self.t + self.phase))
        emitter_colors = np.array([[1.0, 1.0, 0.0, 0.5 + 0.5*pulse]] * len(emitters))
        emitter_sizes = np.array([10.0 + 10.0*pulse] * len(emitters))
        self.emitter_scatter.setData(pos=emitters, color=emitter_colors, size=emitter_sizes)

        # --- Update 2D Cross-Section ---
        slice_idx = self.slice_slider.value()
        x_2d = self.X[:, slice_idx]
        z_2d = Z_display[:, slice_idx]
        self.line_2d.setData(x_2d, z_2d)
        baseline = np.zeros_like(z_2d)
        self.fill_item.setCurves(self.line_2d, pg.PlotDataItem(x_2d, baseline))

        # --- Update Top-Down Heatmap ---
        Z_heat = Z_display.T
        
        if self.use_db_scale:
            norm_heat = np.clip((Z_heat - (max_val - 40)) / 40, 0, 1)
            heat_colors = self.cmap_2d_db.map(norm_heat.flatten(), mode='byte').reshape(Z_heat.shape[0], Z_heat.shape[1], 4)
            self.cbar.setValues((max_val - 40, max_val))
            self.cbar.setGradient(self.cmap_2d_db.getGradient())
        else:
            norm_heat = np.clip((Z_heat / max_val + 1) / 2, 0, 1)
            heat_colors = self.cmap_2d_linear.map(norm_heat.flatten(), mode='byte').reshape(Z_heat.shape[0], Z_heat.shape[1], 4)
            self.cbar.setValues((-max_val, max_val))
            self.cbar.setGradient(self.cmap_2d_linear.getGradient())
            
        self.heat_img.setImage(heat_colors, levels=[0, 255])
        
        if not self.use_db_scale:
            levels = np.linspace(-self.amplitude, self.amplitude, 7)
            self.contour.setData(Z_heat, levels=levels)
            self.contour.show()
        else:
            self.contour.hide()

        # --- Update Air Particles ---
        if self.show_particles:
            particle_Z = Z_linear[self.particle_mesh_X, self.particle_mesh_Y].flatten()
            disp_scale = 0.2 / (self.amplitude + 0.1)
            px = self.particle_base_X + particle_Z * disp_scale
            py = self.particle_base_Y
            
            p_colors = np.clip(particle_Z / (self.amplitude + 0.1), -1, 1)
            cols = np.zeros((len(p_colors), 4))
            cols[:, 3] = 200 
            cols[p_colors > 0, 0] = np.abs(p_colors[p_colors > 0]) * 255
            cols[p_colors < 0, 2] = np.abs(p_colors[p_colors < 0]) * 255
            
            self.particle_scatter.setData(pos=np.column_stack([px, py]), pxMode=False, size=4, color=cols)
            self.particle_scatter.show()
        else:
            self.particle_scatter.hide()

        # --- FPS Calculation ---
        self.frame_count += 1
        elapsed_ms = self.fps_timer.elapsed()
        if elapsed_ms - self.last_fps_time >= 500:
            fps = (self.frame_count * 1000) / (elapsed_ms - self.last_fps_time)
            self.fps_label.setText(f"FPS: {fps:.0f}")
            self.frame_count = 0
            self.last_fps_time = elapsed_ms

    def closeEvent(self, event):
        # Clean up audio stream on exit
        if self.audio_stream:
            self.audio_stream.stop()
            self.audio_stream.close()
        event.accept()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = SoundWaveVisualizer()
    window.show()
    sys.exit(app.exec())