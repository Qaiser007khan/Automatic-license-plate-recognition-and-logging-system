"""
License Plate Recognition System - Beautiful GUI
A modern graphical interface for the LPR system with all features
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import queue
import os
import sys
from pathlib import Path
import json

# Try to import PIL for better image display
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False


class LPRGui:
    def __init__(self, root):
        self.root = root
        self.root.title("License Plate Recognition System - GPU Accelerated")
        self.root.geometry("1200x800")
        self.root.minsize(1000, 700)
        
        # Set modern color scheme
        self.colors = {
            'bg': '#1e1e2e',
            'fg': '#cdd6f4',
            'accent': '#89b4fa',
            'success': '#a6e3a1',
            'warning': '#f9e2af',
            'error': '#f38ba8',
            'panel': '#313244',
            'button': '#89b4fa',
            'button_hover': '#74c7ec',
        }
        
        self.root.configure(bg=self.colors['bg'])
        
        # Configuration variables
        self.setup_variables()
        
        self.log_queue = queue.Queue()
        
        # Load saved settings
        self.load_settings()
        
        # Create GUI
        self.create_gui()
        
        # Processing state
        self.processing = False
        self.process_thread = None
        self.log_queue = queue.Queue()
        
        # Start log update loop
        self.update_logs()
        
    def setup_variables(self):
        """Initialize all configuration variables"""
        # Source settings
        self.source_type = tk.StringVar(value="video")
        self.source_path = tk.StringVar(value="")
        self.rtsp_url = tk.StringVar(value="rtsp://")
        self.webcam_index = tk.StringVar(value="0")
        
        # Model settings
        self.coco_model = tk.StringVar(value="yolov8n.pt")
        self.lp_model = tk.StringVar(value="license_plate_detector.pt")
        
        # OCR settings
        self.ocr_engine = tk.StringVar(value="easyocr")
        self.use_gpu_ocr = tk.BooleanVar(value=True)
        self.batch_ocr = tk.BooleanVar(value=False)
        
        # Processing settings
        self.conf_threshold = tk.DoubleVar(value=0.5)
        self.skip_frames = tk.IntVar(value=0)
        
        # Output settings
        self.save_video = tk.BooleanVar(value=True)
        self.output_video_path = tk.StringVar(value="output.mp4")
        self.output_csv_path = tk.StringVar(value="results.csv")
        self.interpolate = tk.BooleanVar(value=True)
        self.live_view = tk.BooleanVar(value=True)
        
    def create_gui(self):
        """Create the main GUI layout"""
        # Title Bar
        self.create_title_bar()
        
        # Main container with two columns
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Left panel - Settings
        left_panel = self.create_left_panel(main_container)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, expand=False, padx=(0, 5))
        
        # Right panel - Logs and Control
        right_panel = self.create_right_panel(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Status bar
        self.create_status_bar()
        
        # Configure styles
        self.configure_styles()
        
    def create_title_bar(self):
        """Create the title bar with logo and info"""
        title_frame = tk.Frame(self.root, bg=self.colors['accent'], height=80)
        title_frame.pack(fill=tk.X, padx=0, pady=0)
        title_frame.pack_propagate(False)
        
        # Title
        title_label = tk.Label(
            title_frame,
            text="🚗 License Plate Recognition System",
            font=("Segoe UI", 24, "bold"),
            bg=self.colors['accent'],
            fg='white'
        )
        title_label.pack(side=tk.LEFT, padx=20, pady=15)
        
        # Subtitle
        subtitle_label = tk.Label(
            title_frame,
            text="GPU-Accelerated • Real-time Processing • High Accuracy",
            font=("Segoe UI", 10),
            bg=self.colors['accent'],
            fg='white'
        )
        subtitle_label.pack(side=tk.LEFT, padx=0, pady=20)
        
    def create_left_panel(self, parent):
        """Create left settings panel"""
        panel = tk.Frame(parent, bg=self.colors['panel'], width=400)
        panel.pack_propagate(False)
        
        # Canvas with scrollbar
        canvas = tk.Canvas(panel, bg=self.colors['panel'], highlightthickness=0)
        scrollbar = ttk.Scrollbar(panel, orient="vertical", command=canvas.yview)
        scrollable_frame = tk.Frame(canvas, bg=self.colors['panel'])
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        # Add all setting sections
        self.create_source_section(scrollable_frame)
        self.create_model_section(scrollable_frame)
        self.create_ocr_section(scrollable_frame)
        self.create_processing_section(scrollable_frame)
        self.create_output_section(scrollable_frame)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        return panel
        
    def create_right_panel(self, parent):
        """Create right panel with logs and controls"""
        panel = tk.Frame(parent, bg=self.colors['panel'])
        
        # Control buttons at top
        control_frame = tk.Frame(panel, bg=self.colors['panel'])
        control_frame.pack(fill=tk.X, padx=10, pady=10)
        
        self.start_button = tk.Button(
            control_frame,
            text="▶ START PROCESSING",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['success'],
            fg='white',
            activebackground=self.colors['button_hover'],
            command=self.start_processing,
            relief=tk.FLAT,
            padx=30,
            pady=15,
            cursor="hand2"
        )
        self.start_button.pack(side=tk.LEFT, padx=5)
        
        self.stop_button = tk.Button(
            control_frame,
            text="⏹ STOP",
            font=("Segoe UI", 14, "bold"),
            bg=self.colors['error'],
            fg='white',
            activebackground='#e64553',
            command=self.stop_processing,
            relief=tk.FLAT,
            padx=30,
            pady=15,
            state=tk.DISABLED,
            cursor="hand2"
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        
        # Save/Load Settings buttons
        settings_frame = tk.Frame(control_frame, bg=self.colors['panel'])
        settings_frame.pack(side=tk.RIGHT)
        
        tk.Button(
            settings_frame,
            text="💾 Save Settings",
            font=("Segoe UI", 10),
            bg=self.colors['button'],
            fg='white',
            command=self.save_settings,
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=2)
        
        tk.Button(
            settings_frame,
            text="📂 Load Settings",
            font=("Segoe UI", 10),
            bg=self.colors['button'],
            fg='white',
            command=self.load_settings,
            relief=tk.FLAT,
            padx=15,
            pady=8,
            cursor="hand2"
        ).pack(side=tk.LEFT, padx=2)
        
        # Statistics display
        stats_frame = tk.LabelFrame(
            panel,
            text="📊 Statistics",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            relief=tk.GROOVE,
            borderwidth=2
        )
        stats_frame.pack(fill=tk.X, padx=10, pady=5)
        
        self.stats_labels = {}
        stats = [
            ("Frames Processed:", "0"),
            ("Vehicles Detected:", "0"),
            ("Plates Read:", "0"),
            ("Processing FPS:", "0.0"),
            ("Average OCR Time:", "0 ms")
        ]
        
        for i, (label, value) in enumerate(stats):
            row_frame = tk.Frame(stats_frame, bg=self.colors['panel'])
            row_frame.pack(fill=tk.X, padx=10, pady=3)
            
            tk.Label(
                row_frame,
                text=label,
                font=("Segoe UI", 10),
                bg=self.colors['panel'],
                fg=self.colors['fg'],
                anchor=tk.W
            ).pack(side=tk.LEFT)
            
            value_label = tk.Label(
                row_frame,
                text=value,
                font=("Segoe UI", 10, "bold"),
                bg=self.colors['panel'],
                fg=self.colors['accent'],
                anchor=tk.E
            )
            value_label.pack(side=tk.RIGHT)
            self.stats_labels[label] = value_label
        
        # Progress bar
        self.progress_var = tk.DoubleVar()
        self.progress_bar = ttk.Progressbar(
            panel,
            variable=self.progress_var,
            maximum=100,
            mode='determinate',
            length=400
        )
        self.progress_bar.pack(fill=tk.X, padx=10, pady=5)
        
        # Log display
        log_frame = tk.LabelFrame(
            panel,
            text="📝 Processing Log",
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            relief=tk.GROOVE,
            borderwidth=2
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            font=("Consolas", 9),
            bg='#0d1117',
            fg='#c9d1d9',
            insertbackground='white',
            relief=tk.FLAT,
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Add welcome message
        self.log("=" * 70)
        self.log("License Plate Recognition System - Ready")
        self.log("=" * 70)
        self.log("Configure your settings and click START PROCESSING")
        self.log("")
        
        return panel
    
    def create_source_section(self, parent):
        """Create video source section"""
        frame = self.create_section_frame(parent, "🎥 Video Source")
        
        # Source type
        type_frame = tk.Frame(frame, bg=self.colors['panel'])
        type_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Radiobutton(
            type_frame,
            text="Video File",
            variable=self.source_type,
            value="video",
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            selectcolor=self.colors['bg'],
            activebackground=self.colors['panel']
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Radiobutton(
            type_frame,
            text="Webcam",
            variable=self.source_type,
            value="webcam",
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            selectcolor=self.colors['bg'],
            activebackground=self.colors['panel']
        ).pack(side=tk.LEFT, padx=5)
        
        tk.Radiobutton(
            type_frame,
            text="RTSP Stream",
            variable=self.source_type,
            value="rtsp",
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            selectcolor=self.colors['bg'],
            activebackground=self.colors['panel']
        ).pack(side=tk.LEFT, padx=5)
        
        # Video file path
        video_frame = tk.Frame(frame, bg=self.colors['panel'])
        video_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Entry(
            video_frame,
            textvariable=self.source_path,
            font=("Segoe UI", 10),
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            insertbackground=self.colors['fg']
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        tk.Button(
            video_frame,
            text="Browse",
            command=self.browse_video,
            font=("Segoe UI", 9),
            bg=self.colors['button'],
            fg='white',
            relief=tk.FLAT,
            cursor="hand2"
        ).pack(side=tk.RIGHT)
        
        # RTSP URL
        tk.Entry(
            frame,
            textvariable=self.rtsp_url,
            font=("Segoe UI", 10),
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            insertbackground=self.colors['fg']
        ).pack(fill=tk.X, padx=10, pady=5)
        
        # Webcam index
        webcam_frame = tk.Frame(frame, bg=self.colors['panel'])
        webcam_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            webcam_frame,
            text="Webcam Index:",
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg']
        ).pack(side=tk.LEFT)
        
        tk.Entry(
            webcam_frame,
            textvariable=self.webcam_index,
            font=("Segoe UI", 10),
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            width=5,
            insertbackground=self.colors['fg']
        ).pack(side=tk.LEFT, padx=10)
        
    def create_model_section(self, parent):
        """Create model selection section"""
        frame = self.create_section_frame(parent, "🤖 Models")
        
        # COCO Model
        self.create_file_selector(
            frame, "Vehicle Detection Model:", self.coco_model,
            lambda: self.browse_model(self.coco_model, "COCO Model")
        )
        
        # LP Model
        self.create_file_selector(
            frame, "License Plate Model:", self.lp_model,
            lambda: self.browse_model(self.lp_model, "License Plate Model")
        )
        
    def create_ocr_section(self, parent):
        """Create OCR settings section"""
        frame = self.create_section_frame(parent, "📖 OCR Settings")
        
        # OCR Engine
        ocr_frame = tk.Frame(frame, bg=self.colors['panel'])
        ocr_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            ocr_frame,
            text="OCR Engine:",
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg']
        ).pack(side=tk.LEFT)
        
        tk.Radiobutton(
            ocr_frame,
            text="EasyOCR",
            variable=self.ocr_engine,
            value="easyocr",
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            selectcolor=self.colors['bg'],
            activebackground=self.colors['panel']
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Radiobutton(
            ocr_frame,
            text="PaddleOCR",
            variable=self.ocr_engine,
            value="paddleocr",
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            selectcolor=self.colors['bg'],
            activebackground=self.colors['panel']
        ).pack(side=tk.LEFT)
        
        # Options
        tk.Checkbutton(
            frame,
            text="Use GPU for OCR",
            variable=self.use_gpu_ocr,
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            selectcolor=self.colors['bg'],
            activebackground=self.colors['panel']
        ).pack(anchor=tk.W, padx=10, pady=2)
        
        tk.Checkbutton(
            frame,
            text="Batch OCR Processing",
            variable=self.batch_ocr,
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            selectcolor=self.colors['bg'],
            activebackground=self.colors['panel']
        ).pack(anchor=tk.W, padx=10, pady=2)
        
    def create_processing_section(self, parent):
        """Create processing settings section"""
        frame = self.create_section_frame(parent, "⚙️ Processing Settings")
        
        # Confidence threshold
        self.create_slider(
            frame, "Confidence Threshold:", self.conf_threshold, 0.0, 1.0, 0.01
        )
        
        # Skip frames
        skip_frame = tk.Frame(frame, bg=self.colors['panel'])
        skip_frame.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            skip_frame,
            text="Skip Frames:",
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg']
        ).pack(side=tk.LEFT)
        
        tk.Spinbox(
            skip_frame,
            from_=0,
            to=10,
            textvariable=self.skip_frames,
            font=("Segoe UI", 10),
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            width=5
        ).pack(side=tk.LEFT, padx=10)
        
        tk.Label(
            skip_frame,
            text="(0 = process all frames)",
            font=("Segoe UI", 8),
            bg=self.colors['panel'],
            fg=self.colors['fg']
        ).pack(side=tk.LEFT)
        
    def create_output_section(self, parent):
        """Create output settings section"""
        frame = self.create_section_frame(parent, "💾 Output Settings")
        
        # Options
        tk.Checkbutton(
            frame,
            text="Save Output Video",
            variable=self.save_video,
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            selectcolor=self.colors['bg'],
            activebackground=self.colors['panel']
        ).pack(anchor=tk.W, padx=10, pady=2)
        
        tk.Checkbutton(
            frame,
            text="Show Live View",
            variable=self.live_view,
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            selectcolor=self.colors['bg'],
            activebackground=self.colors['panel']
        ).pack(anchor=tk.W, padx=10, pady=2)
        
        tk.Checkbutton(
            frame,
            text="Interpolate Results",
            variable=self.interpolate,
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            selectcolor=self.colors['bg'],
            activebackground=self.colors['panel']
        ).pack(anchor=tk.W, padx=10, pady=2)
        
        # Output paths
        self.create_file_selector(
            frame, "Output Video:", self.output_video_path,
            lambda: self.browse_save("Video", "*.mp4", self.output_video_path)
        )
        
        self.create_file_selector(
            frame, "Output CSV:", self.output_csv_path,
            lambda: self.browse_save("CSV", "*.csv", self.output_csv_path)
        )
        
    def create_section_frame(self, parent, title):
        """Create a labeled section frame"""
        frame = tk.LabelFrame(
            parent,
            text=title,
            font=("Segoe UI", 11, "bold"),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            relief=tk.GROOVE,
            borderwidth=2
        )
        frame.pack(fill=tk.X, padx=5, pady=5)
        return frame
        
    def create_file_selector(self, parent, label, variable, command):
        """Create a file selector row"""
        container = tk.Frame(parent, bg=self.colors['panel'])
        container.pack(fill=tk.X, padx=10, pady=5)
        
        tk.Label(
            container,
            text=label,
            font=("Segoe UI", 9),
            bg=self.colors['panel'],
            fg=self.colors['fg']
        ).pack(anchor=tk.W)
        
        entry_frame = tk.Frame(container, bg=self.colors['panel'])
        entry_frame.pack(fill=tk.X, pady=2)
        
        tk.Entry(
            entry_frame,
            textvariable=variable,
            font=("Segoe UI", 9),
            bg=self.colors['bg'],
            fg=self.colors['fg'],
            insertbackground=self.colors['fg']
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        tk.Button(
            entry_frame,
            text="...",
            command=command,
            font=("Segoe UI", 9),
            bg=self.colors['button'],
            fg='white',
            relief=tk.FLAT,
            width=3,
            cursor="hand2"
        ).pack(side=tk.RIGHT)
        
    def create_slider(self, parent, label, variable, from_, to, resolution):
        """Create a slider with label and value"""
        container = tk.Frame(parent, bg=self.colors['panel'])
        container.pack(fill=tk.X, padx=10, pady=5)
        
        label_frame = tk.Frame(container, bg=self.colors['panel'])
        label_frame.pack(fill=tk.X)
        
        tk.Label(
            label_frame,
            text=label,
            font=("Segoe UI", 10),
            bg=self.colors['panel'],
            fg=self.colors['fg']
        ).pack(side=tk.LEFT)
        
        value_label = tk.Label(
            label_frame,
            text=f"{variable.get():.2f}",
            font=("Segoe UI", 10, "bold"),
            bg=self.colors['panel'],
            fg=self.colors['accent']
        )
        value_label.pack(side=tk.RIGHT)
        
        slider = ttk.Scale(
            container,
            from_=from_,
            to=to,
            variable=variable,
            orient=tk.HORIZONTAL,
            command=lambda v: value_label.config(text=f"{float(v):.2f}")
        )
        slider.pack(fill=tk.X, pady=2)
        
    def create_status_bar(self):
        """Create status bar at bottom"""
        self.status_bar = tk.Label(
            self.root,
            text="Ready",
            font=("Segoe UI", 9),
            bg=self.colors['panel'],
            fg=self.colors['fg'],
            anchor=tk.W,
            relief=tk.SUNKEN,
            padx=10
        )
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)
        
    def configure_styles(self):
        """Configure ttk styles"""
        style = ttk.Style()
        style.theme_use('clam')
        
        style.configure(
            "TProgressbar",
            background=self.colors['accent'],
            troughcolor=self.colors['bg'],
            bordercolor=self.colors['panel'],
            lightcolor=self.colors['accent'],
            darkcolor=self.colors['accent']
        )
        
        style.configure(
            "TScale",
            background=self.colors['panel'],
            troughcolor=self.colors['bg']
        )
        
    # Helper methods
    def browse_video(self):
        """Browse for video file"""
        filename = filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv"),
                ("All files", "*.*")
            ]
        )
        if filename:
            self.source_path.set(filename)
            
    def browse_model(self, variable, title):
        """Browse for model file"""
        filename = filedialog.askopenfilename(
            title=f"Select {title}",
            filetypes=[
                ("Model files", "*.pt *.engine"),
                ("All files", "*.*")
            ]
        )
        if filename:
            variable.set(filename)
            
    def browse_save(self, title, pattern, variable):
        """Browse for save location"""
        filename = filedialog.asksaveasfilename(
            title=f"Save {title}",
            defaultextension=pattern.replace("*", ""),
            filetypes=[
                (f"{title} files", pattern),
                ("All files", "*.*")
            ]
        )
        if filename:
            variable.set(filename)
            
    def log(self, message, level="INFO"):
        """Add message to log"""
        self.log_queue.put((message, level))
        
    def update_logs(self):
        """Update log display from queue"""
        try:
            while True:
                message, level = self.log_queue.get_nowait()
                self.log_text.insert(tk.END, message + "\n")
                self.log_text.see(tk.END)
        except queue.Empty:
            pass
        
        self.root.after(100, self.update_logs)
        
    def update_status(self, text):
        """Update status bar"""
        self.status_bar.config(text=text)
        
    def save_settings(self):
        """Save current settings to file"""
        settings = {
            'source_type': self.source_type.get(),
            'source_path': self.source_path.get(),
            'rtsp_url': self.rtsp_url.get(),
            'webcam_index': self.webcam_index.get(),
            'coco_model': self.coco_model.get(),
            'lp_model': self.lp_model.get(),
            'ocr_engine': self.ocr_engine.get(),
            'use_gpu_ocr': self.use_gpu_ocr.get(),
            'batch_ocr': self.batch_ocr.get(),
            'conf_threshold': self.conf_threshold.get(),
            'skip_frames': self.skip_frames.get(),
            'save_video': self.save_video.get(),
            'output_video_path': self.output_video_path.get(),
            'output_csv_path': self.output_csv_path.get(),
            'interpolate': self.interpolate.get(),
            'live_view': self.live_view.get(),
        }
        
        try:
            with open('lpr_settings.json', 'w') as f:
                json.dump(settings, f, indent=4)
            self.log("✓ Settings saved successfully")
            messagebox.showinfo("Success", "Settings saved successfully!")
        except Exception as e:
            self.log(f"✗ Error saving settings: {e}", "ERROR")
            messagebox.showerror("Error", f"Failed to save settings:\n{e}")
            
    def load_settings(self):
        """Load settings from file"""
        try:
            if os.path.exists('lpr_settings.json'):
                with open('lpr_settings.json', 'r') as f:
                    settings = json.load(f)
                
                self.source_type.set(settings.get('source_type', 'video'))
                self.source_path.set(settings.get('source_path', ''))
                self.rtsp_url.set(settings.get('rtsp_url', 'rtsp://'))
                self.webcam_index.set(settings.get('webcam_index', '0'))
                self.coco_model.set(settings.get('coco_model', 'yolov8n.pt'))
                self.lp_model.set(settings.get('lp_model', 'license_plate_detector.pt'))
                self.ocr_engine.set(settings.get('ocr_engine', 'easyocr'))
                self.use_gpu_ocr.set(settings.get('use_gpu_ocr', True))
                self.batch_ocr.set(settings.get('batch_ocr', False))
                self.conf_threshold.set(settings.get('conf_threshold', 0.5))
                self.skip_frames.set(settings.get('skip_frames', 0))
                self.save_video.set(settings.get('save_video', True))
                self.output_video_path.set(settings.get('output_video_path', 'output.mp4'))
                self.output_csv_path.set(settings.get('output_csv_path', 'results.csv'))
                self.interpolate.set(settings.get('interpolate', True))
                self.live_view.set(settings.get('live_view', True))
                
                self.log("✓ Settings loaded successfully")
        except Exception as e:
            self.log(f"⚠ Could not load settings: {e}", "WARNING")
            
    def start_processing(self):
        """Start the processing in a separate thread"""
        # Validate inputs
        source_type = self.source_type.get()
        
        if source_type == "video":
            if not self.source_path.get() or not os.path.exists(self.source_path.get()):
                messagebox.showerror("Error", "Please select a valid video file!")
                return
            source = self.source_path.get()
        elif source_type == "webcam":
            source = "webcam"
        elif source_type == "rtsp":
            if not self.rtsp_url.get() or self.rtsp_url.get() == "rtsp://":
                messagebox.showerror("Error", "Please enter a valid RTSP URL!")
                return
            source = self.rtsp_url.get()
        else:
            messagebox.showerror("Error", "Invalid source type!")
            return
        
        # Check if models exist
        if not os.path.exists(self.coco_model.get()):
            messagebox.showerror("Error", f"COCO model not found: {self.coco_model.get()}")
            return
            
        if not os.path.exists(self.lp_model.get()):
            messagebox.showerror("Error", f"License plate model not found: {self.lp_model.get()}")
            return
        
        # Update UI state
        self.processing = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)
        self.update_status("Processing...")
        self.progress_var.set(0)
        
        # Clear log
        self.log_text.delete(1.0, tk.END)
        self.log("=" * 70)
        self.log("Starting License Plate Recognition")
        self.log("=" * 70)
        
        # Build command
        cmd = self.build_command(source)
        self.log(f"Command: {' '.join(cmd)}")
        self.log("")
        
        # Start processing thread
        self.process_thread = threading.Thread(target=self.run_processing, args=(cmd,))
        self.process_thread.daemon = True
        self.process_thread.start()
        
    def build_command(self, source):
        """Build command line arguments"""
        cmd = [sys.executable, "unified_lpr.py"]
        
        cmd.extend(["--source", source])
        
        # Models
        cmd.extend(["--coco-model", self.coco_model.get()])
        cmd.extend(["--license-model", self.lp_model.get()])
        
        # OCR settings
        cmd.extend(["--ocr-engine", self.ocr_engine.get()])
        if not self.use_gpu_ocr.get():
            cmd.append("--no-gpu-ocr")
        if self.batch_ocr.get():
            cmd.append("--batch-ocr")
        
        # Processing settings
        cmd.extend(["--conf", str(self.conf_threshold.get())])
        if self.skip_frames.get() > 0:
            cmd.extend(["--skip-frames", str(self.skip_frames.get())])
        
        # Output settings
        if not self.live_view.get():
            cmd.append("--no-display")
        if self.save_video.get():
            cmd.append("--save-video")
            cmd.extend(["--output-video", self.output_video_path.get()])
        cmd.extend(["--output-csv", self.output_csv_path.get()])
        if not self.interpolate.get():
            cmd.append("--no-interpolate")
        
        return cmd
        
    def run_processing(self, cmd):
        """Run the processing command"""
        import subprocess
        
        try:
            # Run command and capture output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            
            # Read output line by line
            for line in iter(process.stdout.readline, ''):
                if not self.processing:
                    process.terminate()
                    break
                
                line = line.strip()
                if line:
                    self.log(line)
                    
                    # Update statistics if possible
                    self.parse_output(line)
            
            process.wait()
            
            if process.returncode == 0:
                self.log("")
                self.log("=" * 70)
                self.log("✓ Processing completed successfully!")
                self.log("=" * 70)
                self.update_status("Processing completed")
                messagebox.showinfo("Success", "Processing completed successfully!")
            else:
                self.log("")
                self.log("=" * 70)
                self.log("✗ Processing failed or was interrupted")
                self.log("=" * 70)
                self.update_status("Processing failed")
                
        except Exception as e:
            self.log(f"✗ Error: {e}", "ERROR")
            self.update_status("Error occurred")
            messagebox.showerror("Error", f"An error occurred:\n{e}")
        
        finally:
            # Reset UI state
            self.processing = False
            self.start_button.config(state=tk.NORMAL)
            self.stop_button.config(state=tk.DISABLED)
            self.progress_var.set(100)
            
    def parse_output(self, line):
        """Parse output line and update statistics"""
        try:
            # Parse frame progress
            if "Frame" in line and "/" in line and "%" in line:
                parts = line.split()
                for i, part in enumerate(parts):
                    if part == "Frame" and i + 1 < len(parts):
                        frame_info = parts[i + 1].split("/")
                        if len(frame_info) == 2:
                            current = int(frame_info[0])
                            total = int(frame_info[1])
                            progress = (current / total) * 100
                            self.progress_var.set(progress)
                            self.stats_labels["Frames Processed:"].config(text=str(current))
                    
                    if "FPS" in part and i > 0:
                        fps_value = parts[i - 1]
                        self.stats_labels["Processing FPS:"].config(text=fps_value)
            
            # Parse statistics
            if "Total Vehicles Detected:" in line:
                value = line.split(":")[-1].strip()
                self.stats_labels["Vehicles Detected:"].config(text=value)
                
            if "Total License Plates Read:" in line:
                value = line.split(":")[-1].strip()
                self.stats_labels["Plates Read:"].config(text=value)
                
            if "Average OCR Time:" in line:
                value = line.split(":")[-1].strip()
                self.stats_labels["Average OCR Time:"].config(text=value)
                
        except Exception:
            pass  # Ignore parsing errors
            
    def stop_processing(self):
        """Stop the current processing"""
        if self.processing:
            result = messagebox.askyesno(
                "Confirm Stop",
                "Are you sure you want to stop processing?"
            )
            if result:
                self.processing = False
                self.log("")
                self.log("⏹ Stopping processing...")
                self.update_status("Stopping...")


def main():
    """Main entry point"""
    root = tk.Tk()
    app = LPRGui(root)
    
    # Handle window close
    def on_closing():
        if app.processing:
            result = messagebox.askyesno(
                "Confirm Exit",
                "Processing is in progress. Are you sure you want to exit?"
            )
            if not result:
                return
            app.processing = False
        
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()


if __name__ == "__main__":
    main()