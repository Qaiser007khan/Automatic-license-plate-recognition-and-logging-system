import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import threading
import os
from pathlib import Path
from improved_unified_lpr import LicensePlateRecognizer
import json

# Predefined cameras (Put your rtsp streams here)
CAMERAS = [
    #{"name": "Camera 1", "url": "rtsp://user:password@}
    
]

class LPRGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Multi-Camera License Plate Recognition")
        self.root.geometry("1100x850")
        self.root.configure(bg="#f0f2f5")
        
        self.recognizers = {}  # Dictionary to hold multiple recognizers
        self.processing_threads = {}
        self.is_running = False
        self.camera_vars = {}  # Checkboxes for camera selection
        
        self.create_widgets()
        self.style_widgets()
        
    def create_widgets(self):
        # Title
        title = tk.Label(self.root, text="Multi-Camera License Plate Recognition", 
                        font=("Segoe UI", 24, "bold"), fg="#2c3e50", bg="#f0f2f5")
        title.pack(pady=20)
        
        # Main frame with scrollbar
        main_canvas = tk.Canvas(self.root, bg="#f0f2f5")
        scrollbar = tk.Scrollbar(self.root, orient="vertical", command=main_canvas.yview)
        scrollable_frame = tk.Frame(main_canvas, bg="#f0f2f5")
        
        scrollable_frame.bind(
            "<Configure>",
            lambda e: main_canvas.configure(scrollregion=main_canvas.bbox("all"))
        )
        
        main_canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
        main_canvas.configure(yscrollcommand=scrollbar.set)
        
        main_canvas.pack(side="left", fill="both", expand=True, padx=20)
        scrollbar.pack(side="right", fill="y")
        
        # === 1. Video Source ===
        source_frame = self.create_section(scrollable_frame, "Video Source")
        self.source_var = tk.StringVar(value="multi_camera")
        
        tk.Radiobutton(source_frame, text="Single Webcam", variable=self.source_var, value="webcam", 
                      bg="#ffffff", font=("Segoe UI", 10), command=self.toggle_source).pack(anchor="w", padx=20, pady=2)
        tk.Radiobutton(source_frame, text="Single Video File", variable=self.source_var, value="file", 
                      bg="#ffffff", font=("Segoe UI", 10), command=self.toggle_source).pack(anchor="w", padx=20, pady=2)
        tk.Radiobutton(source_frame, text="Single RTSP Stream", variable=self.source_var, value="rtsp", 
                      bg="#ffffff", font=("Segoe UI", 10), command=self.toggle_source).pack(anchor="w", padx=20, pady=2)
        tk.Radiobutton(source_frame, text="Multiple Cameras (RTSP)", variable=self.source_var, value="multi_camera", 
                      bg="#ffffff", font=("Segoe UI", 10, "bold"), command=self.toggle_source).pack(anchor="w", padx=20, pady=2)
        
        self.source_entry = tk.Entry(source_frame, width=60, font=("Segoe UI", 10))
        self.source_entry.pack(pady=5, padx=20)
        self.source_entry.insert(0, "0")
        
        browse_btn = tk.Button(source_frame, text="Browse Video", command=self.browse_file,
                              bg="#3498db", fg="white", font=("Segoe UI", 9, "bold"))
        browse_btn.pack(pady=5)
        
        # === 2. Camera Selection ===
        self.camera_frame = self.create_section(scrollable_frame, "Select Cameras (Multiple Selection)")
        
        # Select/Deselect All buttons
        btn_frame = tk.Frame(self.camera_frame, bg="#ffffff")
        btn_frame.pack(fill="x", padx=20, pady=5)
        tk.Button(btn_frame, text="Select All", command=self.select_all_cameras,
                 bg="#27ae60", fg="white", font=("Segoe UI", 9)).pack(side="left", padx=5)
        tk.Button(btn_frame, text="Deselect All", command=self.deselect_all_cameras,
                 bg="#e74c3c", fg="white", font=("Segoe UI", 9)).pack(side="left", padx=5)
        
        # Camera checkboxes in a grid
        cam_grid = tk.Frame(self.camera_frame, bg="#ffffff")
        cam_grid.pack(fill="x", padx=20, pady=5)
        
        for idx, cam in enumerate(CAMERAS):
            var = tk.BooleanVar(value=False)
            self.camera_vars[cam['name']] = var
            cb = tk.Checkbutton(cam_grid, text=f"{cam['name']} ({cam['url'].split('@')[1].split(':')[0]})", 
                              variable=var, bg="#ffffff", font=("Segoe UI", 9))
            row = idx // 3
            col = idx % 3
            cb.grid(row=row, column=col, sticky="w", padx=10, pady=2)
        
        # === 3. Model Selection ===
        model_frame = self.create_section(scrollable_frame, "Model Selection (TensorRT / PyTorch / ONNX)")
        
        # Vehicle Model
        vehicle_frame = tk.Frame(model_frame, bg="#ffffff")
        vehicle_frame.pack(fill="x", pady=5)
        tk.Label(vehicle_frame, text="Vehicle Model:", bg="#ffffff", font=("Segoe UI", 10)).pack(side="left", padx=20)
        self.vehicle_model_path = tk.StringVar()
        vehicle_entry = tk.Entry(vehicle_frame, textvariable=self.vehicle_model_path, width=45)
        vehicle_entry.pack(side="left", padx=5)
        tk.Button(vehicle_frame, text="Browse", command=lambda: self.browse_model(self.vehicle_model_path)).pack(side="left", padx=5)
        
        # License Plate Model
        plate_frame = tk.Frame(model_frame, bg="#ffffff")
        plate_frame.pack(fill="x", pady=5)
        tk.Label(plate_frame, text="License Plate Model:", bg="#ffffff", font=("Segoe UI", 10)).pack(side="left", padx=20)
        self.plate_model_path = tk.StringVar()
        plate_entry = tk.Entry(plate_frame, textvariable=self.plate_model_path, width=45)
        plate_entry.pack(side="left", padx=5)
        tk.Button(plate_frame, text="Browse", command=lambda: self.browse_model(self.plate_model_path)).pack(side="left", padx=5)
        
        # === 4. Detection Settings ===
        det_frame = self.create_section(scrollable_frame, "Detection Settings")
        tk.Label(det_frame, text="Confidence Threshold:", bg="#ffffff", 
                font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=(10,2))
        self.conf_scale = tk.Scale(det_frame, from_=0.0, to=1.0, resolution=0.05, orient="horizontal",
                                  bg="#ffffff", troughcolor="#bdc3c7", highlightthickness=0)
        self.conf_scale.set(0.3)
        self.conf_scale.pack(fill="x", padx=20, pady=2)
        
        tk.Label(det_frame, text="Skip Frames (0 = all):", bg="#ffffff", 
                font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=(10,2))
        self.skip_var = tk.IntVar(value=1)
        skip_entry = tk.Entry(det_frame, textvariable=self.skip_var, width=10, font=("Segoe UI", 10))
        skip_entry.pack(anchor="w", padx=20, pady=2)
        
        # === 5. OCR Settings ===
        ocr_frame = self.create_section(scrollable_frame, "OCR Settings")
        self.ocr_engine_var = tk.StringVar(value="easyocr")
        tk.Radiobutton(ocr_frame, text="EasyOCR", variable=self.ocr_engine_var, value="easyocr",
                      bg="#ffffff", font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=2)
        tk.Radiobutton(ocr_frame, text="PaddleOCR", variable=self.ocr_engine_var, value="paddleocr",
                      bg="#ffffff", font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=2)
        
        self.gpu_ocr_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ocr_frame, text="Use GPU for OCR", variable=self.gpu_ocr_var,
                      bg="#ffffff", font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=5)
        
        self.enhanced_pp_var = tk.BooleanVar(value=False)
        tk.Checkbutton(ocr_frame, text="Enhanced Preprocessing (slower)", variable=self.enhanced_pp_var,
                      bg="#ffffff", font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=5)
        
        self.batch_ocr_var = tk.BooleanVar(value=True)
        tk.Checkbutton(ocr_frame, text="Batch OCR (faster)", variable=self.batch_ocr_var,
                      bg="#ffffff", font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=5)
        
        # === 6. Output Options ===
        out_frame = self.create_section(scrollable_frame, "Output Options")
        self.save_video_var = tk.BooleanVar(value=False)
        tk.Checkbutton(out_frame, text="Save Output Video(s)", variable=self.save_video_var,
                      bg="#ffffff", font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=5)
        
        self.video_path_var = tk.StringVar(value="output_cam")
        video_entry_frame = tk.Frame(out_frame, bg="#ffffff")
        video_entry_frame.pack(fill="x", padx=20, pady=5)
        tk.Label(video_entry_frame, text="Output prefix:", bg="#ffffff", font=("Segoe UI", 9)).pack(side="left")
        video_entry = tk.Entry(video_entry_frame, textvariable=self.video_path_var, width=40)
        video_entry.pack(side="left", padx=5)
        tk.Label(video_entry_frame, text="(will append _CAM1.mp4, etc.)", bg="#ffffff", 
                font=("Segoe UI", 8), fg="#7f8c8d").pack(side="left", padx=5)
        
        self.interpolate_var = tk.BooleanVar(value=True)
        tk.Checkbutton(out_frame, text="Interpolate Missing Frames", variable=self.interpolate_var,
                      bg="#ffffff", font=("Segoe UI", 10)).pack(anchor="w", padx=20, pady=5)
        
        # === 7. Control Buttons ===
        btn_frame = tk.Frame(self.root, bg="#f0f2f5")
        btn_frame.pack(pady=15)
        
        self.start_btn = tk.Button(btn_frame, text="START RECOGNITION", command=self.start_processing,
                                  bg="#27ae60", fg="white", font=("Segoe UI", 12, "bold"), width=22, height=2)
        self.start_btn.pack(side="left", padx=10)
        
        self.stop_btn = tk.Button(btn_frame, text="STOP ALL", command=self.stop_processing,
                                 bg="#c0392b", fg="white", font=("Segoe UI", 12, "bold"), width=10, height=2, state="disabled")
        self.stop_btn.pack(side="left", padx=10)
        
        # === 8. Status Display ===
        status_frame = tk.Frame(self.root, bg="#f0f2f5")
        status_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        tk.Label(status_frame, text="Processing Status:", bg="#f0f2f5", 
                font=("Segoe UI", 10, "bold")).pack(anchor="w")
        
        self.status_text = scrolledtext.ScrolledText(status_frame, height=6, font=("Consolas", 9),
                                                     bg="#2c3e50", fg="#ecf0f1", state="disabled")
        self.status_text.pack(fill="both", expand=True, pady=5)
        
        # === 9. Status Bar ===
        self.status_var = tk.StringVar(value="Ready - Select cameras and start!")
        status_bar = tk.Label(self.root, textvariable=self.status_var, relief="sunken", 
                             anchor="w", bg="#ecf0f1", font=("Segoe UI", 9))
        status_bar.pack(side="bottom", fill="x")
        
        # Initial toggle
        self.toggle_source()
        
    def create_section(self, parent, title):
        frame = tk.LabelFrame(parent, text=f" {title} ", font=("Segoe UI", 11, "bold"), 
                             bg="#ffffff", fg="#2c3e50", padx=10, pady=10)
        frame.pack(fill="x", pady=8)
        return frame
        
    def style_widgets(self):
        style = ttk.Style()
        style.theme_use('clam')
    
    def toggle_source(self):
        """Show/hide camera selection based on source type"""
        if self.source_var.get() == "multi_camera":
            self.camera_frame.pack(fill="x", pady=8, after=self.camera_frame.master.children[list(self.camera_frame.master.children.keys())[0]])
            self.source_entry.config(state="disabled")
        else:
            self.camera_frame.pack_forget()
            self.source_entry.config(state="normal")
    
    def select_all_cameras(self):
        for var in self.camera_vars.values():
            var.set(True)
    
    def deselect_all_cameras(self):
        for var in self.camera_vars.values():
            var.set(False)
        
    def browse_file(self):
        if self.source_var.get() == "file":
            file = filedialog.askopenfilename(
                title="Select Video File",
                filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"), ("All files", "*.*")]
            )
            if file:
                self.source_entry.delete(0, tk.END)
                self.source_entry.insert(0, file)
        elif self.source_var.get() == "rtsp":
            url = tk.simpledialog.askstring("RTSP URL", "Enter RTSP stream URL:")
            if url:
                self.source_entry.delete(0, tk.END)
                self.source_entry.insert(0, url)
    
    def browse_model(self, var):
        file = filedialog.askopenfilename(
            title="Select Model File",
            filetypes=[("Models", "*.pt *.engine *.onnx"), ("All", "*.*")]
        )
        if file:
            var.set(file)
    
    def log_status(self, message):
        """Add message to status text widget"""
        self.status_text.config(state="normal")
        self.status_text.insert(tk.END, f"{message}\n")
        self.status_text.see(tk.END)
        self.status_text.config(state="disabled")
    
    def start_processing(self):
        if self.is_running:
            return
        
        if self.source_var.get() == "multi_camera":
            # Get selected cameras
            selected_cameras = [cam for cam in CAMERAS if self.camera_vars[cam['name']].get()]
            
            if not selected_cameras:
                messagebox.showerror("Error", "Please select at least one camera")
                return
            
            # Start processing for each selected camera
            self.is_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_var.set(f"Processing {len(selected_cameras)} camera(s)...")
            
            for cam in selected_cameras:
                self.start_camera_processing(cam)
        else:
            # Single source processing
            source = self.source_entry.get().strip()
            if not source:
                messagebox.showerror("Error", "Please specify a valid source")
                return
            
            self.is_running = True
            self.start_btn.config(state="disabled")
            self.stop_btn.config(state="normal")
            self.status_var.set("Processing...")
            
            cam_info = {"name": "Single Source", "url": source if self.source_var.get() != "webcam" else "webcam"}
            self.start_camera_processing(cam_info)
    
    def start_camera_processing(self, cam_info):
        """Start processing for a single camera"""
        try:
            cam_name = cam_info['name']
            cam_url = cam_info['url']
            
            # Determine source
            if self.source_var.get() == "webcam":
                source = "webcam"
            else:
                source = cam_url
            
            # Create output path
            output_path = None
            if self.save_video_var.get():
                base_path = self.video_path_var.get()
                output_path = f"{base_path}_{cam_name.replace(' ', '_')}.mp4"
            
            self.log_status(f"[{cam_name}] Initializing...")
            
            recognizer = LicensePlateRecognizer(
                source=source,
                live_view=True,
                save_video=self.save_video_var.get(),
                output_path=output_path,
                skip_frames=self.skip_var.get(),
                conf_threshold=self.conf_scale.get(),
                resize_display=True,
                coco_model_path=self.vehicle_model_path.get().strip() or None,
                license_plate_model_path=self.plate_model_path.get().strip() or None,
                use_gpu_ocr=self.gpu_ocr_var.get(),
                ocr_engine=self.ocr_engine_var.get(),
                batch_ocr=self.batch_ocr_var.get(),
                use_enhanced_preprocessing=self.enhanced_pp_var.get()
            )
            
            self.recognizers[cam_name] = recognizer
            
            # Start processing thread
            thread = threading.Thread(
                target=self.run_camera_processing, 
                args=(cam_name, recognizer),
                daemon=True
            )
            self.processing_threads[cam_name] = thread
            thread.start()
            
            self.log_status(f"[{cam_name}] Started successfully")
            
        except Exception as e:
            self.log_status(f"[{cam_name}] ERROR: {str(e)}")
            messagebox.showerror("Error", f"Failed to start {cam_name}: {str(e)}")
    
    def run_camera_processing(self, cam_name, recognizer):
        """Run processing for a single camera"""
        try:
            self.log_status(f"[{cam_name}] Processing...")
            recognizer.run()
            
            # Save results
            csv_path = f"results_{cam_name.replace(' ', '_')}.csv"
            recognizer.save_results(
                csv_path=csv_path,
                interpolate=self.interpolate_var.get()
            )
            
            self.log_status(f"[{cam_name}] Completed! Results saved to {csv_path}")
            
        except Exception as e:
            self.log_status(f"[{cam_name}] Processing error: {str(e)}")
        finally:
            # Check if all threads are complete
            self.check_all_complete()
    
    def check_all_complete(self):
        """Check if all processing threads have completed"""
        active_threads = [t for t in self.processing_threads.values() if t.is_alive()]
        if not active_threads:
            self.root.after(0, self.processing_complete)
    
    def processing_complete(self):
        self.status_var.set("All processing complete!")
        messagebox.showinfo("Success", "License plate recognition completed for all cameras!\nResults saved to individual CSV files.")
        self.reset_ui()
    
    def stop_processing(self):
        self.is_running = False
        self.status_var.set("Stopping all cameras...")
        self.log_status("Stopping all processing...")
        
        for cam_name, recognizer in self.recognizers.items():
            try:
                recognizer.cap.release()
                self.log_status(f"[{cam_name}] Stopped")
            except:
                pass
        
        self.reset_ui()
    
    def reset_ui(self):
        self.is_running = False
        self.recognizers = {}
        self.processing_threads = {}
        self.start_btn.config(state="normal")
        self.stop_btn.config(state="disabled")
        self.status_var.set("Ready")

# Launch GUI
if __name__ == "__main__":
    root = tk.Tk()
    app = LPRGUI(root)
    root.mainloop()