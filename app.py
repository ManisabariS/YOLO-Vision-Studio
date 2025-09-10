import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk
from ultralytics import YOLO
import os
import datetime
import time
import numpy as np
import threading
import queue
import json
import csv
from collections import deque

class WebcamApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Advanced Real-Time Object Detection with YOLOv8")
        self.root.geometry("1800x1000")
        self.root.configure(bg='#2c3e50')
        self.root.minsize(1600, 900)
        
        # Style configuration
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        # Create directory for annotated frames
        self.output_dir = "annotated_frames"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Performance optimization
        self.frame_queue = queue.Queue(maxsize=2)
        self.processed_frame = None
        self.processing_lock = threading.Lock()
        
        # YOLO model
        self.model = None
        self.model_path = "raghu_model.pt"
        
        # Webcam
        self.cap = None
        self.is_camera_active = False
        self.current_camera = 0
        
        # Camera properties with better defaults
        self.camera_properties = {
            'gain': 0,
            'exposure': -6,  # Better default for most cameras
            'brightness': 50,
            'contrast': 50,
            'saturation': 50,
            'hue': 0,
            'sharpness': 50,
            'gamma': 100,
            'white_balance': 5000,
            'backlight_compensation': 1
        }
        
        # Performance metrics
        self.prev_time = 0
        self.curr_time = 0
        self.fps = 0
        self.target_fps = 60
        self.detection_count = 0
        self.avg_inference_time = 0
        self.fps_history = deque(maxlen=30)
        self.inference_history = deque(maxlen=30)
        
        # Detection statistics
        self.class_counts = {}
        self.detection_log = []
        
        # Create main panels
        self.create_ui()
        
        # Try to load model
        self.load_model_on_start()
        
        # Start the video update loop
        self.update_video()
        
        # Start the frame processing thread
        self.processing_thread = None
        self.stop_processing = False
        self.start_processing_thread()
    
    def configure_styles(self):
        # Configure styles with modern colors
        self.style.configure('TFrame', background='#2c3e50')
        self.style.configure('TLabel', background='#2c3e50', foreground='white', font=('Arial', 9))
        self.style.configure('Title.TLabel', background='#2c3e50', foreground='white', font=('Arial', 12, 'bold'))
        self.style.configure('TButton', background='#3498db', foreground='black', font=('Arial', 10, 'bold'))
        self.style.map('TButton', 
                      background=[('active', '#2980b9'), ('pressed', '#1c638e')],
                      foreground=[('active', 'white')])
        self.style.configure('TCheckbutton', background='#2c3e50', foreground='white')
        self.style.configure('TCombobox', fieldbackground='white', foreground='black')
        self.style.configure('TScale', background='#2c3e50')
        self.style.configure('Horizontal.TScale', background='#2c3e50')
        self.style.configure('Header.TFrame', background='#34495e')
        self.style.configure('Header.TLabel', background='#34495e', foreground='white', font=('Arial', 10, 'bold'))
        self.style.configure('White.TLabel', background='#2c3e50', foreground='white')
        self.style.configure('Success.TLabel', background='#2c3e50', foreground='#2ecc71')
        self.style.configure('Warning.TLabel', background='#2c3e50', foreground='#f39c12')
        self.style.configure('Error.TLabel', background='#2c3e50', foreground='#e74c3c')
        
        # Notebook style
        self.style.configure('TNotebook', background='#2c3e50', borderwidth=0)
        self.style.configure('TNotebook.Tab', background='#34495e', foreground='white', padding=[10, 5])
        self.style.map('TNotebook.Tab', background=[('selected', '#3498db'), ('active', '#2980b9')])
    
    def create_ui(self):
        # Main container
        main_container = ttk.Frame(self.root)
        main_container.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Left panel for controls (fixed width)
        left_panel = ttk.Frame(main_container, width=400)
        left_panel.pack(side=tk.LEFT, fill=tk.Y, padx=(0, 10))
        left_panel.pack_propagate(False)
        
        # Right panel for video (expands)
        right_panel = ttk.Frame(main_container)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Create notebook for tabs
        notebook = ttk.Notebook(left_panel)
        notebook.pack(fill=tk.BOTH, expand=True)
        
        # Camera Settings Tab
        cam_frame = ttk.Frame(notebook, padding=10)
        notebook.add(cam_frame, text="Camera Settings")
        self.create_camera_tab(cam_frame)
        
        # Model Settings Tab
        model_frame = ttk.Frame(notebook, padding=10)
        notebook.add(model_frame, text="Model Settings")
        self.create_model_tab(model_frame)
        
        # Detection Settings Tab
        detect_frame = ttk.Frame(notebook, padding=10)
        notebook.add(detect_frame, text="Detection Settings")
        self.create_detection_tab(detect_frame)
        
        # Output Settings Tab
        output_frame = ttk.Frame(notebook, padding=10)
        notebook.add(output_frame, text="Output Settings")
        self.create_output_tab(output_frame)
        
        # Performance Tab
        perf_frame = ttk.Frame(notebook, padding=10)
        notebook.add(perf_frame, text="Performance")
        self.create_performance_tab(perf_frame)
        
        # Video display in right panel
        video_container = ttk.Frame(right_panel)
        video_container.pack(fill=tk.BOTH, expand=True, pady=(0, 10))
        
        # Add a header above the video
        video_header = ttk.Frame(video_container, style='Header.TFrame', height=30)
        video_header.pack(fill=tk.X, pady=(0, 5))
        video_header.pack_propagate(False)
        
        ttk.Label(video_header, text="Live Detection Feed", style='Header.TLabel').pack(side=tk.LEFT, padx=10)
        
        self.video_label = tk.Label(video_container, text="Webcam feed will appear here", 
                                   bg='black', fg='white', font=('Arial', 14))
        self.video_label.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        # Status bar at bottom
        status_bar = ttk.Frame(right_panel, height=30)
        status_bar.pack(fill=tk.X, pady=(10, 0))
        status_bar.pack_propagate(False)
        
        self.status_label = ttk.Label(status_bar, text="Status: Ready to start inspection", style='White.TLabel')
        self.status_label.pack(side=tk.LEFT, padx=10)
        
        self.fps_label = ttk.Label(status_bar, text="FPS: 0.0 | Inference: 0ms | Detections: 0", style='White.TLabel')
        self.fps_label.pack(side=tk.RIGHT, padx=10)
        
        # Detection info panel
        self.info_frame = ttk.LabelFrame(right_panel, text="Detection Information", height=150)
        self.info_frame.pack(fill=tk.X, pady=(10, 0))
        self.info_frame.pack_propagate(False)
        
        self.info_text = scrolledtext.ScrolledText(self.info_frame, height=8, wrap=tk.WORD, 
                                                  bg='#ecf0f1', fg='#2c3e50', font=('Consolas', 9))
        self.info_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)
        self.info_text.config(state=tk.DISABLED)
        
        # Action buttons at bottom of left panel
        action_frame = ttk.Frame(left_panel, height=50)
        action_frame.pack(fill=tk.X, side=tk.BOTTOM, pady=(10, 0))
        action_frame.pack_propagate(False)
        
        self.start_button = ttk.Button(action_frame, text="Start Inspection", 
                                      command=self.toggle_webcam)
        self.start_button.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))
        
        ttk.Button(action_frame, text="Open Output Folder", 
                  command=self.open_output_folder).pack(side=tk.RIGHT, fill=tk.X, expand=True)
    
    def create_camera_tab(self, parent):
        # Camera selection
        header = ttk.Frame(parent, style='Header.TFrame')
        header.pack(fill=tk.X, pady=(0, 10))
        ttk.Label(header, text="Camera Selection", style='Header.TLabel').pack(padx=5, pady=5)
        
        ttk.Label(parent, text="Camera Source:", style='White.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.camera_selector = ttk.Combobox(parent, values=self.get_available_cameras(), 
                                           state="readonly", width=25)
        self.camera_selector.current(0)
        self.camera_selector.pack(fill=tk.X, pady=(0, 15))
        self.camera_selector.bind('<<ComboboxSelected>>', self.on_camera_change)
        
        # Resolution settings
        ttk.Label(parent, text="Resolution:", style='White.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.res_var = tk.StringVar(value="1280x720")
        res_combo = ttk.Combobox(parent, textvariable=self.res_var, 
                                values=["640x480", "800x600", "1024x768", "1280x720", "1280x960", "1920x1080"], 
                                width=25, state="readonly")
        res_combo.pack(fill=tk.X, pady=(0, 15))
        res_combo.bind('<<ComboboxSelected>>', self.on_resolution_change)
        
        # FPS settings
        ttk.Label(parent, text="FPS Target:", style='White.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.fps_var = tk.IntVar(value=60)
        fps_combo = ttk.Combobox(parent, textvariable=self.fps_var, 
                                values=[15, 20, 25, 30, 60, 90, 120], 
                                width=25, state="readonly")
        fps_combo.pack(fill=tk.X, pady=(0, 15))
        fps_combo.bind('<<ComboboxSelected>>', self.on_fps_change)
        
        # Camera controls header
        header2 = ttk.Frame(parent, style='Header.TFrame')
        header2.pack(fill=tk.X, pady=(10, 10))
        ttk.Label(header2, text="Camera Controls", style='Header.TLabel').pack(padx=5, pady=5)
        
        # Create scrollable frame for camera controls
        canvas_frame = ttk.Frame(parent)
        canvas_frame.pack(fill=tk.BOTH, expand=True)
        
        canvas = tk.Canvas(canvas_frame, bg='#2c3e50', highlightthickness=0)
        scrollbar = ttk.Scrollbar(canvas_frame, orient=tk.VERTICAL, command=canvas.yview)
        scrollable_frame = ttk.Frame(canvas)
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
        )
        
        canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)
        
        canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Camera control sliders
        controls = [
            ('gain', 'Gain', 0, 100, 0),
            ('exposure', 'Exposure', -10, -1, -6),  # Adjusted range for better control
            ('brightness', 'Brightness', 0, 100, 50),
            ('contrast', 'Contrast', 0, 100, 50),
            ('saturation', 'Saturation', 0, 100, 50),
            ('hue', 'Hue', -180, 180, 0),
            ('sharpness', 'Sharpness', 0, 100, 50),
            ('gamma', 'Gamma', 100, 500, 100),
            ('white_balance', 'White Balance', 4000, 7000, 5000),
            ('backlight_compensation', 'Backlight Compensation', 0, 10, 1)
        ]
        
        self.camera_vars = {}
        for prop, name, min_val, max_val, default in controls:
            frame = ttk.Frame(scrollable_frame)
            frame.pack(fill=tk.X, pady=(0, 10))
            
            ttk.Label(frame, text=f"{name}:", style='White.TLabel', width=20).pack(anchor=tk.W, side=tk.LEFT)
            
            var = tk.IntVar(value=default)
            self.camera_vars[prop] = var
            
            scale = ttk.Scale(frame, from_=min_val, to=max_val, variable=var, 
                             orient=tk.HORIZONTAL, command=lambda v, p=prop: self.update_camera_property(p, v))
            scale.pack(fill=tk.X, expand=True, side=tk.LEFT, padx=(10, 5))
            
            value_label = ttk.Label(frame, text=f"{default}", style='White.TLabel', width=5)
            value_label.pack(side=tk.RIGHT, padx=(5, 0))
            
            # Store reference to update label
            var.trace('w', lambda *args, v=var, l=value_label: l.config(text=f"{v.get()}"))
    
    def create_model_tab(self, parent):
        ttk.Label(parent, text="Current Model:", style='Title.TLabel').pack(anchor=tk.W, pady=(0, 10))
        
        self.model_status_label = ttk.Label(parent, text="Model: Not Loaded", 
                                           wraplength=300, justify=tk.LEFT, style='White.TLabel')
        self.model_status_label.pack(anchor=tk.W, pady=(0, 15))
        
        ttk.Button(parent, text="Load Custom Model", 
                  command=self.load_model).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(parent, text="Reload Default Model", 
                  command=self.reload_default_model).pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(parent, text="Model Performance:", style='Title.TLabel').pack(anchor=tk.W, pady=(10, 10))
        
        # Model info frame
        info_frame = ttk.Frame(parent)
        info_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(info_frame, text="Input Size:", style='White.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        self.input_size_label = ttk.Label(info_frame, text="640x640", style='White.TLabel')
        self.input_size_label.grid(row=0, column=1, sticky=tk.E, pady=2)
        
        ttk.Label(info_frame, text="Classes:", style='White.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        self.classes_label = ttk.Label(info_frame, text="80", style='White.TLabel')
        self.classes_label.grid(row=1, column=1, sticky=tk.E, pady=2)
        
        ttk.Label(info_frame, text="Parameters:", style='White.TLabel').grid(row=2, column=0, sticky=tk.W, pady=2)
        self.params_label = ttk.Label(info_frame, text="0M", style='White.TLabel')
        self.params_label.grid(row=2, column=1, sticky=tk.E, pady=2)
        
        info_frame.columnconfigure(1, weight=1)
        
        # Model optimization options
        ttk.Label(parent, text="Optimization:", style='Title.TLabel').pack(anchor=tk.W, pady=(20, 10))
        
        self.half_precision_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Use Half Precision (FP16)", 
                       variable=self.half_precision_var, style='TCheckbutton').pack(anchor=tk.W, pady=(0, 5))
        
        self.optimize_model_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Optimize Model for Inference", 
                       variable=self.optimize_model_var, style='TCheckbutton').pack(anchor=tk.W, pady=(0, 5))
    
    def create_detection_tab(self, parent):
        ttk.Label(parent, text="Detection Parameters", style='Title.TLabel').pack(anchor=tk.W, pady=(0, 15))
        
        # Confidence threshold
        ttk.Label(parent, text="Confidence Threshold:", style='White.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.confidence_var = tk.DoubleVar(value=0.5)
        conf_scale = ttk.Scale(parent, from_=0.1, to=0.9, 
                              variable=self.confidence_var, orient=tk.HORIZONTAL)
        conf_scale.pack(fill=tk.X, pady=(0, 5))
        
        self.conf_label = ttk.Label(parent, text="0.5", style='White.TLabel')
        self.conf_label.pack(anchor=tk.E, pady=(0, 15))
        self.confidence_var.trace('w', lambda *args: self.conf_label.config(text=f"{self.confidence_var.get():.2f}"))
        
        # IoU threshold
        ttk.Label(parent, text="IoU Threshold:", style='White.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.iou_var = tk.DoubleVar(value=0.45)
        iou_scale = ttk.Scale(parent, from_=0.1, to=0.9, 
                             variable=self.iou_var, orient=tk.HORIZONTAL)
        iou_scale.pack(fill=tk.X, pady=(0, 5))
        
        self.iou_label = ttk.Label(parent, text="0.45", style='White.TLabel')
        self.iou_label.pack(anchor=tk.E, pady=(0, 15))
        self.iou_var.trace('w', lambda *args: self.iou_label.config(text=f"{self.iou_var.get():.2f}"))
        
        # Detection filtering
        ttk.Label(parent, text="Max Detections:", style='White.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.max_det_var = tk.IntVar(value=300)
        max_det_spin = ttk.Spinbox(parent, from_=1, to=1000, textvariable=self.max_det_var, width=10)
        max_det_spin.pack(anchor=tk.W, pady=(0, 15))
        
        # Class filtering
        ttk.Label(parent, text="Class Filtering:", style='Title.TLabel').pack(anchor=tk.W, pady=(20, 10))
        
        self.class_filter_var = tk.StringVar(value="")
        class_filter_entry = ttk.Entry(parent, textvariable=self.class_filter_var, width=20)
        class_filter_entry.pack(fill=tk.X, pady=(0, 5))
        ttk.Label(parent, text="Comma-separated class IDs to include (empty for all)", 
                 style='White.TLabel', font=('Arial', 8)).pack(anchor=tk.W, pady=(0, 15))
    
    def create_output_tab(self, parent):
        ttk.Label(parent, text="Output Settings", style='Title.TLabel').pack(anchor=tk.W, pady=(0, 15))
        
        # Save options
        self.save_frames_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Save Annotated Frames", 
                       variable=self.save_frames_var, style='TCheckbutton').pack(anchor=tk.W, pady=(0, 10))
        
        self.show_info_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(parent, text="Show Detection Info", 
                       variable=self.show_info_var, style='TCheckbutton').pack(anchor=tk.W, pady=(0, 10))
        
        self.save_csv_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(parent, text="Save Detection Log (CSV)", 
                       variable=self.save_csv_var, style='TCheckbutton').pack(anchor=tk.W, pady=(0, 15))
        
        # Save interval
        ttk.Label(parent, text="Save Interval (frames):", style='White.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.interval_var = tk.IntVar(value=30)
        interval_spin = ttk.Spinbox(parent, from_=1, to=300, textvariable=self.interval_var, width=10)
        interval_spin.pack(anchor=tk.W, pady=(0, 15))
        
        # Output quality
        ttk.Label(parent, text="JPEG Quality:", style='White.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.quality_var = tk.IntVar(value=95)
        quality_scale = ttk.Scale(parent, from_=50, to=100, 
                                 variable=self.quality_var, orient=tk.HORIZONTAL)
        quality_scale.pack(fill=tk.X, pady=(0, 5))
        
        self.quality_label = ttk.Label(parent, text="95", style='White.TLabel')
        self.quality_label.pack(anchor=tk.E, pady=(0, 15))
        self.quality_var.trace('w', lambda *args: self.quality_label.config(text=f"{self.quality_var.get()}"))
        
        # Export settings
        ttk.Label(parent, text="Export Settings:", style='Title.TLabel').pack(anchor=tk.W, pady=(20, 10))
        
        ttk.Button(parent, text="Export Current Settings", 
                  command=self.export_settings).pack(fill=tk.X, pady=(0, 5))
        
        ttk.Button(parent, text="Import Settings", 
                  command=self.import_settings).pack(fill=tk.X, pady=(0, 5))
    
    def create_performance_tab(self, parent):
        ttk.Label(parent, text="Performance Optimization", style='Title.TLabel').pack(anchor=tk.W, pady=(0, 15))
        
        # Frame skipping
        ttk.Label(parent, text="Frame Skip:", style='White.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.frame_skip_var = tk.IntVar(value=0)
        frame_skip_combo = ttk.Combobox(parent, textvariable=self.frame_skip_var, 
                                       values=[0, 1, 2, 3, 4, 5], width=10, state="readonly")
        frame_skip_combo.pack(anchor=tk.W, pady=(0, 15))
        ttk.Label(parent, text="Skip frames to improve FPS (0 = process every frame)", 
                 style='White.TLabel', font=('Arial', 8)).pack(anchor=tk.W, pady=(0, 15))
        
        # Inference device
        ttk.Label(parent, text="Inference Device:", style='White.TLabel').pack(anchor=tk.W, pady=(0, 5))
        self.device_var = tk.StringVar(value="CPU")
        device_combo = ttk.Combobox(parent, textvariable=self.device_var, 
                                   values=["CPU", "GPU"], width=10, state="readonly")
        device_combo.pack(anchor=tk.W, pady=(0, 15))
        
        # Performance monitoring
        ttk.Label(parent, text="Performance Metrics:", style='Title.TLabel').pack(anchor=tk.W, pady=(20, 10))
        
        metrics_frame = ttk.Frame(parent)
        metrics_frame.pack(fill=tk.X, pady=(0, 10))
        
        ttk.Label(metrics_frame, text="Avg FPS:", style='White.TLabel').grid(row=0, column=0, sticky=tk.W, pady=2)
        self.avg_fps_label = ttk.Label(metrics_frame, text="0.0", style='White.TLabel')
        self.avg_fps_label.grid(row=0, column=1, sticky=tk.E, pady=2)
        
        ttk.Label(metrics_frame, text="Avg Inference:", style='White.TLabel').grid(row=1, column=0, sticky=tk.W, pady=2)
        self.avg_inference_label = ttk.Label(metrics_frame, text="0ms", style='White.TLabel')
        self.avg_inference_label.grid(row=1, column=1, sticky=tk.E, pady=2)
        
        ttk.Label(metrics_frame, text="Total Detections:", style='White.TLabel').grid(row=2, column=0, sticky=tk.W, pady=2)
        self.total_detections_label = ttk.Label(metrics_frame, text="0", style='White.TLabel')
        self.total_detections_label.grid(row=2, column=1, sticky=tk.E, pady=2)
        
        metrics_frame.columnconfigure(1, weight=1)
        
        # Performance tips
        ttk.Label(parent, text="Performance Tips:", style='Title.TLabel').pack(anchor=tk.W, pady=(20, 10))
        
        tips_text = """• Lower resolution for higher FPS
• Use frame skipping for very fast detection
• Enable half precision if using GPU
• Close other applications using the camera
• Adjust exposure for better image quality"""
        
        tips_label = ttk.Label(parent, text=tips_text, style='White.TLabel', justify=tk.LEFT)
        tips_label.pack(anchor=tk.W, pady=(0, 10))
    
    def start_processing_thread(self):
        self.stop_processing = False
        self.processing_thread = threading.Thread(target=self.process_frames, daemon=True)
        self.processing_thread.start()
    
    def process_frames(self):
        skip_counter = 0
        while not self.stop_processing:
            try:
                # Get frame from queue with timeout
                frame = self.frame_queue.get(timeout=0.1)
                
                # Apply frame skipping
                if self.frame_skip_var.get() > 0:
                    skip_counter += 1
                    if skip_counter <= self.frame_skip_var.get():
                        continue
                    skip_counter = 0
                
                # Process frame if model is loaded
                if self.model and frame is not None:
                    try:
                        # Run inference with current settings
                        start_time = time.time()
                        
                        # Prepare class filter
                        class_filter = None
                        if self.class_filter_var.get().strip():
                            try:
                                class_filter = [int(x.strip()) for x in self.class_filter_var.get().split(',')]
                            except:
                                pass
                        
                        results = self.model(frame, verbose=False, 
                                           conf=self.confidence_var.get(),
                                           iou=self.iou_var.get(),
                                           max_det=self.max_det_var.get(),
                                           classes=class_filter)
                        
                        inference_time = (time.time() - start_time) * 1000  # Convert to ms
                        
                        # Update average inference time
                        self.inference_history.append(inference_time)
                        self.avg_inference_time = sum(self.inference_history) / len(self.inference_history) if self.inference_history else 0
                        
                        annotated_frame = results[0].plot()
                        
                        # Check if any detections were made
                        if len(results[0].boxes) > 0:
                            detections = []
                            for box in results[0].boxes:
                                class_id = int(box.cls)
                                class_name = results[0].names[class_id]
                                confidence = float(box.conf)
                                detections.append({
                                    'class_id': class_id,
                                    'class_name': class_name,
                                    'confidence': confidence,
                                    'bbox': box.xywh[0].tolist() if hasattr(box.xywh[0], 'tolist') else box.xywh[0]
                                })
                                
                                # Update class counts
                                if class_name in self.class_counts:
                                    self.class_counts[class_name] += 1
                                else:
                                    self.class_counts[class_name] = 1
                            
                            # Add to detection log
                            self.detection_log.append({
                                'timestamp': datetime.datetime.now().isoformat(),
                                'detections': detections
                            })
                            
                            # Save to CSV if enabled
                            if self.save_csv_var.get():
                                self.save_detection_to_csv(detections)
                        
                        # Store processed frame
                        with self.processing_lock:
                            self.processed_frame = (annotated_frame, len(results[0].boxes), inference_time)
                        
                    except Exception as e:
                        print(f"Error in processing: {e}")
                        with self.processing_lock:
                            self.processed_frame = (frame, 0, 0)
                else:
                    # No model, just pass through the frame
                    with self.processing_lock:
                        self.processed_frame = (frame, 0, 0)
                        
            except queue.Empty:
                continue
            except Exception as e:
                print(f"Error in processing thread: {e}")
                continue
    
    def save_detection_to_csv(self, detections):
        csv_path = os.path.join(self.output_dir, "detections_log.csv")
        file_exists = os.path.isfile(csv_path)
        
        with open(csv_path, 'a', newline='') as f:
            writer = csv.writer(f)
            if not file_exists:
                writer.writerow(['Timestamp', 'Class', 'Confidence', 'Bbox_X', 'Bbox_Y', 'Bbox_W', 'Bbox_H'])
            
            for detection in detections:
                bbox = detection['bbox']
                writer.writerow([
                    datetime.datetime.now().isoformat(),
                    detection['class_name'],
                    f"{detection['confidence']:.4f}",
                    f"{bbox[0]:.2f}" if len(bbox) > 0 else 0,
                    f"{bbox[1]:.2f}" if len(bbox) > 1 else 0,
                    f"{bbox[2]:.2f}" if len(bbox) > 2 else 0,
                    f"{bbox[3]:.2f}" if len(bbox) > 3 else 0
                ])
    
    def update_camera_property(self, property_name, value):
        if self.cap and self.cap.isOpened():
            try:
                # Map property names to OpenCV constants
                prop_map = {
                    'gain': cv2.CAP_PROP_GAIN,
                    'exposure': cv2.CAP_PROP_EXPOSURE,
                    'brightness': cv2.CAP_PROP_BRIGHTNESS,
                    'contrast': cv2.CAP_PROP_CONTRAST,
                    'saturation': cv2.CAP_PROP_SATURATION,
                    'hue': cv2.CAP_PROP_HUE,
                    'sharpness': cv2.CAP_PROP_SHARPNESS,
                    'gamma': cv2.CAP_PROP_GAMMA,
                    'white_balance': cv2.CAP_PROP_WB_TEMPERATURE,
                    'backlight_compensation': cv2.CAP_PROP_BACKLIGHT
                }
                
                if property_name in prop_map:
                    self.cap.set(prop_map[property_name], float(value))
            except Exception as e:
                print(f"Error setting camera property {property_name}: {e}")
    
    def on_camera_change(self, event):
        if self.is_camera_active:
            self.toggle_webcam()  # Stop current camera
            self.toggle_webcam()  # Start with new camera
    
    def on_resolution_change(self, event):
        if self.is_camera_active and self.cap and self.cap.isOpened():
            try:
                width, height = map(int, self.res_var.get().split('x'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except Exception as e:
                print(f"Error changing resolution: {e}")
    
    def on_fps_change(self, event):
        if self.is_camera_active and self.cap and self.cap.isOpened():
            try:
                self.cap.set(cv2.CAP_PROP_FPS, self.fps_var.get())
                self.target_fps = self.fps_var.get()
            except Exception as e:
                print(f"Error changing FPS: {e}")
    
    def load_model_on_start(self):
        try:
            # Determine device
            device = 'cuda' if self.device_var.get() == "GPU" else 'cpu'
            
            # Load model with optional optimization
            self.model = YOLO(self.model_path)
            
            # Apply optimizations if requested
            if self.half_precision_var.get() and device == 'cuda':
                self.model = self.model.half()
            
            self.model_status_label.config(text=f"Model loaded: {os.path.basename(self.model_path)}")
            
            # Update model info
            if hasattr(self.model, 'model'):
                # Try to get model information
                try:
                    self.input_size_label.config(text=f"{self.model.model.args.imgsz}x{self.model.model.args.imgsz}")
                    self.classes_label.config(text=str(self.model.model.nc))
                    # Estimate parameters (this is a rough estimate)
                    param_count = sum(p.numel() for p in self.model.model.parameters())
                    self.params_label.config(text=f"{param_count/1e6:.1f}M")
                except:
                    pass
                    
            print(f"Successfully loaded model from {self.model_path}")
        except Exception as e:
            self.model_status_label.config(text="Model: Failed to load")
            print(f"Error loading model: {str(e)}")
            self.model = None
    
    def reload_default_model(self):
        try:
            # Determine device
            device = 'cuda' if self.device_var.get() == "GPU" else 'cpu'
            
            self.model = YOLO(self.model_path)
            
            # Apply optimizations if requested
            if self.half_precision_var.get() and device == 'cuda':
                self.model = self.model.half()
                
            self.model_status_label.config(text=f"Model loaded: {os.path.basename(self.model_path)}")
            self.status_label.config(text="Status: Default model reloaded successfully")
        except Exception as e:
            self.model_status_label.config(text="Model: Failed to load")
            self.status_label.config(text=f"Status: Error reloading model: {str(e)}")
            self.model = None
    
    def get_available_cameras(self):
        cameras = []
        max_cameras_to_check = 10  # Increased from 5
        
        for i in range(max_cameras_to_check):
            try:
                # Try different backends
                for backend in [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY]:
                    cap = cv2.VideoCapture(i, backend)
                    if cap.isOpened():
                        ret, _ = cap.read()
                        if ret:
                            # Try to get camera name
                            camera_name = f"Camera {i}"
                            try:
                                # Try to get the backend name for more info
                                backend_name = {
                                    cv2.CAP_DSHOW: "DirectShow",
                                    cv2.CAP_MSMF: "Media Foundation",
                                    cv2.CAP_V4L2: "V4L2",
                                    cv2.CAP_ANY: "Any"
                                }.get(backend, "Unknown")
                                camera_name = f"Camera {i} ({backend_name})"
                            except:
                                pass
                            cameras.append(camera_name)
                        cap.release()
                        break
                time.sleep(0.1)
            except:
                pass
                
        return cameras if cameras else ["Camera 0"]
    
    def load_model(self):
        file_path = filedialog.askopenfilename(
            title="Select YOLO Model",
            filetypes=[("Model Files", "*.pt"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                # Determine device
                device = 'cuda' if self.device_var.get() == "GPU" else 'cpu'
                
                self.model = YOLO(file_path)
                
                # Apply optimizations if requested
                if self.half_precision_var.get() and device == 'cuda':
                    self.model = self.model.half()
                    
                self.model_path = file_path
                self.model_status_label.config(text=f"Model loaded: {os.path.basename(file_path)}")
                self.status_label.config(text=f"Status: Model loaded from {os.path.basename(file_path)}")
            except Exception as e:
                self.status_label.config(text=f"Error loading model: {str(e)}")
                messagebox.showerror("Load Error", f"Failed to load model:\n{str(e)}")
    
    def open_output_folder(self):
        try:
            os.startfile(self.output_dir)  # Works on Windows
        except:
            try:
                # Try alternative methods for other OS
                import subprocess
                subprocess.Popen(['xdg-open', self.output_dir])  # Linux
            except:
                messagebox.showinfo("Output Folder", f"Annotated frames are saved in:\n{os.path.abspath(self.output_dir)}")
    
    def toggle_webcam(self):
        if not self.is_camera_active:
            # Start webcam
            camera_index = int(self.camera_selector.get().split()[1])  # Get the camera index
            
            # Try different backends
            backends = [cv2.CAP_DSHOW, cv2.CAP_MSMF, cv2.CAP_V4L2, cv2.CAP_ANY]
            
            for backend in backends:
                try:
                    self.cap = cv2.VideoCapture(camera_index, backend)
                    if self.cap.isOpened():
                        break
                except:
                    continue
            
            if not self.cap or not self.cap.isOpened():
                self.status_label.config(text="Status: Failed to open camera")
                messagebox.showerror("Camera Error", 
                    "Failed to open camera. Please check:\n"
                    "1. Camera is connected\n"
                    "2. No other application is using the camera\n"
                    "3. Camera drivers are installed")
                return
            
            # Set camera resolution and FPS
            try:
                width, height = map(int, self.res_var.get().split('x'))
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
                self.cap.set(cv2.CAP_PROP_FPS, self.fps_var.get())
                self.target_fps = self.fps_var.get()
                
                # Apply camera properties
                for prop, var in self.camera_vars.items():
                    self.update_camera_property(prop, var.get())
                    
            except Exception as e:
                print(f"Error setting camera properties: {e}")
                
            self.is_camera_active = True
            self.start_button.config(text="Stop Inspection")
            self.frame_counter = 0
            self.detection_count = 0
            self.class_counts = {}
            self.detection_log = []
            
            self.status_label.config(text="Status: Inspection active")
            self.update_info_text("Inspection started. Waiting for detections...")
        else:
            # Stop webcam
            self.is_camera_active = False
            self.start_button.config(text="Start Inspection")
            if self.cap:
                self.cap.release()
                self.cap = None
            
            # Generate summary report
            summary = f"Inspection stopped. Total detections: {self.detection_count}\n"
            if self.class_counts:
                summary += "Class distribution:\n"
                for class_name, count in self.class_counts.items():
                    summary += f"  {class_name}: {count}\n"
            
            self.status_label.config(text="Status: Inspection stopped")
            self.update_info_text(summary)
    
    def update_info_text(self, message):
        self.info_text.config(state=tk.NORMAL)
        self.info_text.delete(1.0, tk.END)
        self.info_text.insert(tk.END, f"{datetime.datetime.now().strftime('%H:%M:%S')} - {message}\n")
        self.info_text.config(state=tk.DISABLED)
        self.info_text.see(tk.END)
    
    def append_info_text(self, message):
        if self.show_info_var.get():
            self.info_text.config(state=tk.NORMAL)
            self.info_text.insert(tk.END, f"{datetime.datetime.now().strftime('%H:%M:%S')} - {message}\n")
            self.info_text.config(state=tk.DISABLED)
            self.info_text.see(tk.END)
    
    def save_annotated_frame(self, frame):
        if self.save_frames_var.get():
            timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = os.path.join(self.output_dir, f"detection_{timestamp}.jpg")
            
            # Save with specified quality
            cv2.imwrite(filename, frame, [cv2.IMWRITE_JPEG_QUALITY, self.quality_var.get()])
            return filename
        return None
    
    def export_settings(self):
        file_path = filedialog.asksaveasfilename(
            title="Export Settings",
            defaultextension=".json",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                settings = {
                    'camera': {
                        'index': self.camera_selector.current(),
                        'resolution': self.res_var.get(),
                        'fps': self.fps_var.get(),
                        'properties': {k: v.get() for k, v in self.camera_vars.items()}
                    },
                    'model': {
                        'path': self.model_path,
                        'confidence': self.confidence_var.get(),
                        'iou': self.iou_var.get(),
                        'max_detections': self.max_det_var.get(),
                        'class_filter': self.class_filter_var.get(),
                        'half_precision': self.half_precision_var.get(),
                        'device': self.device_var.get()
                    },
                    'output': {
                        'save_frames': self.save_frames_var.get(),
                        'show_info': self.show_info_var.get(),
                        'save_csv': self.save_csv_var.get(),
                        'interval': self.interval_var.get(),
                        'quality': self.quality_var.get()
                    },
                    'performance': {
                        'frame_skip': self.frame_skip_var.get()
                    }
                }
                
                with open(file_path, 'w') as f:
                    json.dump(settings, f, indent=4)
                
                self.status_label.config(text=f"Settings exported to {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Export Error", f"Failed to export settings:\n{str(e)}")
    
    def import_settings(self):
        file_path = filedialog.askopenfilename(
            title="Import Settings",
            filetypes=[("JSON Files", "*.json"), ("All Files", "*.*")]
        )
        if file_path:
            try:
                with open(file_path, 'r') as f:
                    settings = json.load(f)
                
                # Apply camera settings
                if 'camera' in settings:
                    cam_settings = settings['camera']
                    if 'index' in cam_settings:
                        self.camera_selector.current(cam_settings['index'])
                    if 'resolution' in cam_settings:
                        self.res_var.set(cam_settings['resolution'])
                    if 'fps' in cam_settings:
                        self.fps_var.set(cam_settings['fps'])
                    if 'properties' in cam_settings:
                        for prop, value in cam_settings['properties'].items():
                            if prop in self.camera_vars:
                                self.camera_vars[prop].set(value)
                
                # Apply model settings
                if 'model' in settings:
                    model_settings = settings['model']
                    if 'confidence' in model_settings:
                        self.confidence_var.set(model_settings['confidence'])
                    if 'iou' in model_settings:
                        self.iou_var.set(model_settings['iou'])
                    if 'max_detections' in model_settings:
                        self.max_det_var.set(model_settings['max_detections'])
                    if 'class_filter' in model_settings:
                        self.class_filter_var.set(model_settings['class_filter'])
                    if 'half_precision' in model_settings:
                        self.half_precision_var.set(model_settings['half_precision'])
                    if 'device' in model_settings:
                        self.device_var.set(model_settings['device'])
                    # Note: model path is not automatically loaded for safety
                
                # Apply output settings
                if 'output' in settings:
                    output_settings = settings['output']
                    if 'save_frames' in output_settings:
                        self.save_frames_var.set(output_settings['save_frames'])
                    if 'show_info' in output_settings:
                        self.show_info_var.set(output_settings['show_info'])
                    if 'save_csv' in output_settings:
                        self.save_csv_var.set(output_settings['save_csv'])
                    if 'interval' in output_settings:
                        self.interval_var.set(output_settings['interval'])
                    if 'quality' in output_settings:
                        self.quality_var.set(output_settings['quality'])
                
                # Apply performance settings
                if 'performance' in settings:
                    perf_settings = settings['performance']
                    if 'frame_skip' in perf_settings:
                        self.frame_skip_var.set(perf_settings['frame_skip'])
                
                self.status_label.config(text=f"Settings imported from {os.path.basename(file_path)}")
            except Exception as e:
                messagebox.showerror("Import Error", f"Failed to import settings:\n{str(e)}")
    
    def update_video(self):
        if self.is_camera_active and self.cap and self.cap.isOpened():
            try:
                ret, frame = self.cap.read()
                if ret and frame is not None and frame.size > 0:
                    # Calculate FPS
                    self.curr_time = time.time()
                    if self.prev_time > 0:
                        time_diff = self.curr_time - self.prev_time
                        if time_diff > 0:
                            current_fps = 1.0 / time_diff
                            self.fps_history.append(current_fps)
                            self.fps = sum(self.fps_history) / len(self.fps_history) if self.fps_history else 0
                    self.prev_time = self.curr_time
                    
                    # Add frame to processing queue if not full
                    if self.frame_queue.qsize() < 2:
                        try:
                            self.frame_queue.put_nowait(frame)
                        except queue.Full:
                            pass
                    
                    # Get processed frame if available
                    annotated_frame = frame
                    detection_count = 0
                    inference_time = 0
                    
                    with self.processing_lock:
                        if self.processed_frame is not None:
                            annotated_frame, detection_count, inference_time = self.processed_frame
                            self.processed_frame = None
                    
                    # Update detection count
                    if detection_count > 0:
                        self.detection_count += detection_count
                    
                    # Update performance labels
                    self.fps_label.config(text=f"FPS: {self.fps:.1f} | Inference: {inference_time:.1f}ms | Detections: {self.detection_count}")
                    self.avg_fps_label.config(text=f"{np.mean(list(self.fps_history)):.1f}" if self.fps_history else "0.0")
                    self.avg_inference_label.config(text=f"{self.avg_inference_time:.1f}ms")
                    self.total_detections_label.config(text=f"{self.detection_count}")
                    
                    # Convert frame to PIL Image
                    rgb_image = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(rgb_image)
                    
                    # Resize image to fit label while maintaining aspect ratio
                    label_width = self.video_label.winfo_width()
                    label_height = self.video_label.winfo_height()
                    
                    if label_width > 1 and label_height > 1:
                        img_width, img_height = pil_image.size
                        ratio = min(label_width / img_width, label_height / img_height)
                        new_width = int(img_width * ratio)
                        new_height = int(img_height * ratio)
                        
                        pil_image = pil_image.resize((new_width, new_height), Image.LANCZOS)
                    
                    # Convert to ImageTk
                    tk_image = ImageTk.PhotoImage(pil_image)
                    
                    # Update label
                    self.video_label.config(image=tk_image, text="")
                    self.video_label.image = tk_image
                else:
                    self.video_label.config(text="Failed to capture frame", image="")
            except Exception as e:
                print(f"Error processing frame: {e}")
                self.video_label.config(text="Error processing frame", image="")
        else:
            if not self.is_camera_active:
                self.video_label.config(text="Webcam feed will appear here", image="")
        
        # Schedule the next update
        self.root.after(10, self.update_video)  # Reduced from 20ms to 10ms for smoother updates

    def __del__(self):
        self.stop_processing = True
        if self.cap:
            self.cap.release()

if __name__ == "__main__":
    root = tk.Tk()
    app = WebcamApp(root)
    
    # Handle application close
    def on_closing():
        app.stop_processing = True
        if app.cap:
            app.cap.release()
        root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    root.mainloop()