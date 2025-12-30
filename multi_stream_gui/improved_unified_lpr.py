from ultralytics import YOLO
import cv2
import numpy as np
import torch
import time
import csv
import argparse
import os
from pathlib import Path
from scipy.interpolate import interp1d
from sort.sort import Sort
from util import get_car, read_license_plate, initialize_ocr, read_license_plates_batch
from collections import Counter
from difflib import SequenceMatcher

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def draw_border(img, top_left, bottom_right, color=(0, 255, 0), thickness=10, 
                line_length_x=200, line_length_y=200):
    """Draw decorative corner borders around bounding boxes"""
    x1, y1 = top_left
    x2, y2 = bottom_right

    cv2.line(img, (x1, y1), (x1, y1 + line_length_y), color, thickness)
    cv2.line(img, (x1, y1), (x1 + line_length_x, y1), color, thickness)
    cv2.line(img, (x1, y2), (x1, y2 - line_length_y), color, thickness)
    cv2.line(img, (x1, y2), (x1 + line_length_x, y2), color, thickness)
    cv2.line(img, (x2, y1), (x2 - line_length_x, y1), color, thickness)
    cv2.line(img, (x2, y1), (x2, y1 + line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2, y2 - line_length_y), color, thickness)
    cv2.line(img, (x2, y2), (x2 - line_length_x, y2), color, thickness)

    return img


def preprocess_license_plate(license_plate_crop):
    """
    Enhanced pre-processing pipeline for better OCR
    Returns list of processed images to try
    """
    if license_plate_crop.size == 0:
        return []
    
    processed_images = []
    
    # Convert to grayscale
    gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
    
    # 1. CLAHE (Contrast Limited Adaptive Histogram Equalization)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(gray)
    processed_images.append(enhanced)
    
    # 2. Denoising
    denoised = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)
    processed_images.append(denoised)
    
    # 3. Sharpening
    kernel = np.array([[-1,-1,-1],
                       [-1, 9,-1],
                       [-1,-1,-1]])
    sharpened = cv2.filter2D(gray, -1, kernel)
    processed_images.append(sharpened)
    
    # 4. Otsu's thresholding
    _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    processed_images.append(otsu)
    
    # 5. Adaptive threshold
    adaptive = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
        cv2.THRESH_BINARY, 11, 2
    )
    processed_images.append(adaptive)
    
    # 6. Morphological operations
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    morph = cv2.morphologyEx(otsu, cv2.MORPH_CLOSE, kernel)
    processed_images.append(morph)
    
    # 7. Try inverted versions
    inverted_versions = [cv2.bitwise_not(img) for img in processed_images[:3]]
    processed_images.extend(inverted_versions)
    
    return processed_images


def read_license_plate_enhanced(license_plate_crop):
    """
    Try OCR on multiple pre-processed versions
    Return best result
    """
    processed_images = preprocess_license_plate(license_plate_crop)
    
    results = []
    for img in processed_images:
        text, score = read_license_plate(img)
        if text is not None and len(text) > 0:
            results.append((text, score))
    
    if not results:
        return None, None
    
    # Return result with highest confidence
    best_result = max(results, key=lambda x: x[1])
    return best_result


def interpolate_bounding_boxes(data):
    """Interpolate missing bounding boxes between frames"""
    frame_numbers = np.array([int(row['frame_nmr']) for row in data])
    car_ids = np.array([int(float(row['car_id'])) for row in data])
    car_bboxes = np.array([list(map(float, row['car_bbox'][1:-1].split())) for row in data])
    license_plate_bboxes = np.array([list(map(float, row['license_plate_bbox'][1:-1].split())) for row in data])

    interpolated_data = []
    unique_car_ids = np.unique(car_ids)
    
    for car_id in unique_car_ids:
        frame_numbers_ = [p['frame_nmr'] for p in data if int(float(p['car_id'])) == int(float(car_id))]
        
        car_mask = car_ids == car_id
        car_frame_numbers = frame_numbers[car_mask]
        car_bboxes_interpolated = []
        license_plate_bboxes_interpolated = []

        first_frame_number = car_frame_numbers[0]

        for i in range(len(car_bboxes[car_mask])):
            frame_number = car_frame_numbers[i]
            car_bbox = car_bboxes[car_mask][i]
            license_plate_bbox = license_plate_bboxes[car_mask][i]

            if i > 0:
                prev_frame_number = car_frame_numbers[i-1]
                prev_car_bbox = car_bboxes_interpolated[-1]
                prev_license_plate_bbox = license_plate_bboxes_interpolated[-1]

                if frame_number - prev_frame_number > 1:
                    frames_gap = frame_number - prev_frame_number
                    x = np.array([prev_frame_number, frame_number])
                    x_new = np.linspace(prev_frame_number, frame_number, num=frames_gap, endpoint=False)
                    interp_func = interp1d(x, np.vstack((prev_car_bbox, car_bbox)), axis=0, kind='linear')
                    interpolated_car_bboxes = interp_func(x_new)
                    interp_func = interp1d(x, np.vstack((prev_license_plate_bbox, license_plate_bbox)), axis=0, kind='linear')
                    interpolated_license_plate_bboxes = interp_func(x_new)

                    car_bboxes_interpolated.extend(interpolated_car_bboxes[1:])
                    license_plate_bboxes_interpolated.extend(interpolated_license_plate_bboxes[1:])

            car_bboxes_interpolated.append(car_bbox)
            license_plate_bboxes_interpolated.append(license_plate_bbox)

        for i in range(len(car_bboxes_interpolated)):
            frame_number = first_frame_number + i
            row = {
                'frame_nmr': str(frame_number),
                'car_id': str(car_id),
                'car_bbox': ' '.join(map(str, car_bboxes_interpolated[i])),
                'license_plate_bbox': ' '.join(map(str, license_plate_bboxes_interpolated[i]))
            }

            if str(frame_number) not in frame_numbers_:
                row['license_plate_bbox_score'] = '0'
                row['license_number'] = '0'
                row['license_number_score'] = '0'
            else:
                original_row = [p for p in data if int(p['frame_nmr']) == frame_number and 
                              int(float(p['car_id'])) == int(float(car_id))][0]
                row['license_plate_bbox_score'] = original_row.get('license_plate_bbox_score', '0')
                row['license_number'] = original_row.get('license_number', '0')
                row['license_number_score'] = original_row.get('license_number_score', '0')

            interpolated_data.append(row)

    return interpolated_data


def write_csv(results, output_path):
    """Write results to CSV file"""
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 
                        'license_plate_bbox_score', 'license_number', 'license_number_score'])
        
        for frame_nmr, frame_data in results.items():
            for car_id, data in frame_data.items():
                writer.writerow([
                    frame_nmr,
                    car_id,
                    ' '.join(map(str, data['car']['bbox'])),
                    ' '.join(map(str, data['license_plate']['bbox'])),
                    data['license_plate']['bbox_score'],
                    data['license_plate']['text'],
                    data['license_plate']['text_score']
                ])


# ============================================================================
# MAIN LICENSE PLATE RECOGNIZER CLASS
# ============================================================================

class LicensePlateRecognizer:
    def __init__(self, source, live_view=True, save_video=False, output_path='output.mp4', 
                 skip_frames=0, conf_threshold=0.5, resize_display=True,
                 coco_model_path=None, license_plate_model_path=None,
                 use_gpu_ocr=True, ocr_engine='easyocr', batch_ocr=False,
                 use_enhanced_preprocessing=True):
        self.source = source
        self.live_view = live_view
        self.save_video = save_video
        self.output_path = output_path
        self.skip_frames = skip_frames
        self.conf_threshold = conf_threshold
        self.resize_display = resize_display
        self.coco_model_path = coco_model_path
        self.license_plate_model_path = license_plate_model_path
        self.batch_ocr = batch_ocr
        self.use_enhanced_preprocessing = use_enhanced_preprocessing
        self.results = {}
        self.mot_tracker = Sort()
        self.vehicles = [2, 3, 5, 7]  # car, motorcycle, bus, truck
        self.license_plate_cache = {}
        
        # Multi-frame aggregation storage
        self.plate_readings_per_car = {}  # {car_id: [(text, score, frame), ...]}
        
        # Timing statistics
        self.total_inference_time = 0.0
        self.total_ocr_time = 0.0
        self.ocr_count = 0
        
        # Setup device
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self._print_device_info()
        
        # Initialize OCR (GPU-accelerated)
        self._initialize_ocr(use_gpu_ocr, ocr_engine)
        
        # Load models
        self._load_models()
        
        # Setup video capture
        self._setup_capture()
    
    def _print_device_info(self):
        """Print device information"""
        print("=" * 70)
        print("License Plate Recognition System - ENHANCED VERSION")
        print("=" * 70)
        print(f"\n✓ Device: {self.device.upper()}")
        
        if self.device == 'cuda':
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA Version: {torch.version.cuda}")
            print(f"  Available Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        else:
            print("  ⚠️  Running on CPU - Processing will be slower")
        
        print(f"\n✓ Enhanced Pre-processing: {'ENABLED' if self.use_enhanced_preprocessing else 'DISABLED'}")
        print(f"✓ Multi-frame Aggregation: ENABLED")
    
    def _initialize_ocr(self, use_gpu, ocr_engine):
        """Initialize GPU-accelerated OCR"""
        print("\n" + "-" * 70)
        print("Initializing OCR Engine...")
        print("-" * 70)
        
        try:
            initialize_ocr(use_gpu=use_gpu, ocr_engine=ocr_engine)
            self.ocr_enabled = True
            if self.batch_ocr:
                print("  ✓ Batch OCR processing enabled")
        except Exception as e:
            print(f"  ✗ OCR initialization failed: {e}")
            print("  Continuing without OCR...")
            self.ocr_enabled = False
    
    def _load_models(self):
        """Load YOLO models (supports both .pt and .engine formats)"""
        print("\n" + "-" * 70)
        print("Loading YOLO Models...")
        print("-" * 70)
        
        try:
            print("  Loading COCO model (YOLOv8n)...", end=" ")
            if self.coco_model_path:
                model_path = self.coco_model_path
            else:
                if os.path.exists('yolov8n_fp16.engine'):
                    model_path = 'yolov8n_fp16.engine'
                elif os.path.exists('yolov8n.engine'):
                    model_path = 'yolov8n.engine'
                else:
                    model_path = 'yolov8n.pt'
            
            self.coco_model = YOLO(model_path)
            if model_path.endswith('.pt'):
                self.coco_model.to(self.device)
            model_type = "TensorRT" if model_path.endswith('.engine') else "PyTorch"
            print(f"✓ ({model_type})")
        except Exception as e:
            print(f"✗\n  Error: {e}")
            exit()
        
        try:
            print("  Loading License Plate Detector...", end=" ")
            if self.license_plate_model_path:
                model_path = self.license_plate_model_path
            else:
                if os.path.exists('license_plate_detector_fp16.engine'):
                    model_path = 'license_plate_detector_fp16.engine'
                elif os.path.exists('license_plate_detector.engine'):
                    model_path = 'license_plate_detector.engine'
                else:
                    model_path = 'license_plate_detector.pt'
            
            self.license_plate_detector = YOLO(model_path)
            if model_path.endswith('.pt'):
                self.license_plate_detector.to(self.device)
            model_type = "TensorRT" if model_path.endswith('.engine') else "PyTorch"
            print(f"✓ ({model_type})")
        except Exception as e:
            print(f"✗\n  Error: {e}")
            exit()
    
    def _setup_capture(self):
        """Setup video capture based on source type"""
        print("\n" + "-" * 70)
        print("Setting Up Video Source...")
        print("-" * 70)
        
        self.has_gui_support = self._check_gui_support()
        if self.live_view and not self.has_gui_support:
            print("  ⚠️  OpenCV GUI not available - disabling live view")
            self.live_view = False
        
        if self.source.lower() == 'webcam':
            self.cap = cv2.VideoCapture(0)
            self.source_type = 'webcam'
            print("  ✓ Webcam initialized")
        elif self.source.startswith('rtsp://') or self.source.startswith('http://'):
            self.cap = cv2.VideoCapture(self.source)
            self.source_type = 'stream'
            print(f"  ✓ RTSP/HTTP stream initialized: {self.source}")
        else:
            self.cap = cv2.VideoCapture(self.source)
            self.source_type = 'video'
            print(f"  ✓ Video file loaded: {self.source}")
        
        if not self.cap.isOpened():
            print(f"  ✗ Error: Could not open video source '{self.source}'")
            exit()
        
        self.fps = int(self.cap.get(cv2.CAP_PROP_FPS)) or 30
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"    Resolution: {self.width}x{self.height}")
        print(f"    FPS: {self.fps}")
        if self.source_type == 'video':
            print(f"    Total Frames: {self.total_frames}")
            print(f"    Duration: {self.total_frames/self.fps:.2f} seconds")
        
        if self.save_video:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            self.video_writer = cv2.VideoWriter(self.output_path, fourcc, self.fps, 
                                               (self.width, self.height))
    
    def _check_gui_support(self):
        """Check if OpenCV has GUI support"""
        try:
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            cv2.imshow('test', test_img)
            cv2.destroyAllWindows()
            return True
        except:
            return False
    
    def aggregate_license_plate_text(self, car_id):
        """
        Strategy 1: Multi-frame text aggregation
        Aggregate multiple readings for a car and return best text
        """
        if car_id not in self.plate_readings_per_car or len(self.plate_readings_per_car[car_id]) == 0:
            return None, 0.0
        
        readings = self.plate_readings_per_car[car_id]
        
        # Filter by confidence threshold
        good_readings = [r for r in readings if r[1] > 0.6]
        if not good_readings:
            good_readings = readings
        
        # Method 1: Simple voting (most common)
        texts = [r[0] for r in good_readings]
        text_counts = Counter(texts)
        most_common_text, count = text_counts.most_common(1)[0]
        
        # Method 2: Clustering similar texts
        clusters = []
        for text in texts:
            matched = False
            for cluster in clusters:
                similarity = SequenceMatcher(None, text, cluster[0]).ratio()
                if similarity > 0.8:  # 80% similar
                    cluster.append(text)
                    matched = True
                    break
            if not matched:
                clusters.append([text])
        
        # Get largest cluster
        if clusters:
            largest_cluster = max(clusters, key=len)
            cluster_text_counts = Counter(largest_cluster)
            best_text = cluster_text_counts.most_common(1)[0][0]
        else:
            best_text = most_common_text
        
        # Find best score for this text
        matching_readings = [r for r in good_readings if r[0] == best_text]
        if matching_readings:
            best_score = max(matching_readings, key=lambda x: x[1])[1]
        else:
            best_score = good_readings[0][1] if good_readings else 0.0
        
        return best_text, best_score
    
    def process_frame(self, frame, frame_nmr):
        """Process a single frame for license plate detection"""
        self.results[frame_nmr] = {}
        
        inf_start = time.time()
        
        # Detect vehicles
        detections = self.coco_model(frame, device=self.device, verbose=False, 
                                     conf=self.conf_threshold, iou=0.5, imgsz=640)[0]
        detections_ = []
        
        for detection in detections.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = detection
            if int(class_id) in self.vehicles:
                detections_.append([x1, y1, x2, y2, score])
        
        # Track vehicles
        if len(detections_) > 0:
            track_ids = self.mot_tracker.update(np.asarray(detections_))
        else:
            track_ids = np.empty((0, 5))
        
        # Detect license plates
        license_plates = self.license_plate_detector(frame, device=self.device, verbose=False,
                                                     conf=self.conf_threshold, iou=0.5, imgsz=640)[0]
        
        self.total_inference_time += (time.time() - inf_start)
        
        plates_detected = 0
        
        # Process license plates
        for license_plate in license_plates.boxes.data.tolist():
            x1, y1, x2, y2, score, class_id = license_plate
            
            xcar1, ycar1, xcar2, ycar2, car_id = get_car(license_plate, track_ids)
            
            if car_id != -1:
                y1_int, y2_int = max(0, int(y1)), min(frame.shape[0], int(y2))
                x1_int, x2_int = max(0, int(x1)), min(frame.shape[1], int(x2))
                
                license_plate_crop = frame[y1_int:y2_int, x1_int:x2_int, :]
                
                if license_plate_crop.size == 0:
                    continue
                
                # OCR with enhanced preprocessing
                if self.ocr_enabled:
                    ocr_start = time.time()
                    
                    if self.use_enhanced_preprocessing:
                        license_plate_text, license_plate_text_score = read_license_plate_enhanced(
                            license_plate_crop
                        )
                    else:
                        # Standard preprocessing
                        license_plate_crop_gray = cv2.cvtColor(license_plate_crop, cv2.COLOR_BGR2GRAY)
                        _, license_plate_crop_thresh = cv2.threshold(
                            license_plate_crop_gray, 64, 255, cv2.THRESH_BINARY_INV
                        )
                        license_plate_text, license_plate_text_score = read_license_plate(
                            license_plate_crop_thresh
                        )
                    
                    self.total_ocr_time += (time.time() - ocr_start)
                    self.ocr_count += 1
                else:
                    license_plate_text, license_plate_text_score = "OCR_DISABLED", 0.0
                
                if license_plate_text is not None:
                    # Store reading for multi-frame aggregation
                    if car_id not in self.plate_readings_per_car:
                        self.plate_readings_per_car[car_id] = []
                    self.plate_readings_per_car[car_id].append(
                        (license_plate_text, license_plate_text_score, frame_nmr)
                    )
                    
                    self.results[frame_nmr][car_id] = {
                        'car': {'bbox': [xcar1, ycar1, xcar2, ycar2]},
                        'license_plate': {
                            'bbox': [x1, y1, x2, y2],
                            'text': license_plate_text,
                            'bbox_score': score,
                            'text_score': license_plate_text_score
                        }
                    }
                    plates_detected += 1
                    
                    # Update cache with best reading
                    if car_id not in self.license_plate_cache or \
                       license_plate_text_score > self.license_plate_cache[car_id]['score']:
                        crop_width = int((x2 - x1) * 150 / max(1, (y2 - y1)))
                        license_crop_resized = cv2.resize(license_plate_crop, (crop_width, 150))
                        self.license_plate_cache[car_id] = {
                            'text': license_plate_text,
                            'score': license_plate_text_score,
                            'crop': license_crop_resized
                        }
        
        return len(detections_), plates_detected
    
    def visualize_frame(self, frame, frame_nmr):
        """Add visualizations to frame - LICENSE PLATE IN CENTER OF VEHICLE"""
        if frame_nmr not in self.results:
            return frame
        
        vis_frame = frame.copy()
        
        for car_id, data in self.results[frame_nmr].items():
            # Draw car bounding box
            xcar1, ycar1, xcar2, ycar2 = map(int, data['car']['bbox'])
            draw_border(vis_frame, (xcar1, ycar1), (xcar2, ycar2), 
                       (0, 255, 0), 15, line_length_x=100, line_length_y=100)
            
            # Draw license plate bounding box
            x1, y1, x2, y2 = map(int, data['license_plate']['bbox'])
            cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 0, 255), 8)
            
            # Get aggregated license plate text (best from all frames)
            aggregated_text, aggregated_score = self.aggregate_license_plate_text(car_id)
            
            # Draw license plate crop and text if available
            if car_id in self.license_plate_cache:
                license_crop = self.license_plate_cache[car_id]['crop']
                # Use aggregated text if available, otherwise use cached text
                license_text = aggregated_text if aggregated_text else self.license_plate_cache[car_id]['text']
                
                H, W, _ = license_crop.shape
                
                try:
                    # Calculate center of vehicle
                    car_center_y = int((ycar1 + ycar2) / 2)
                    car_center_x = int((xcar1 + xcar2) / 2)
                    
                    # Position license plate crop at CENTER of vehicle
                    y_start = max(0, car_center_y - H // 2)
                    y_end = min(vis_frame.shape[0], car_center_y + H // 2)
                    x_start = max(0, car_center_x - W // 2)
                    x_end = min(vis_frame.shape[1], car_center_x + W // 2)
                    
                    # Adjust if bounds exceeded
                    actual_h = y_end - y_start
                    actual_w = x_end - x_start
                    
                    if actual_h > 0 and actual_w > 0:
                        # Insert license plate crop with semi-transparent background
                        overlay = vis_frame.copy()
                        
                        # Draw dark background
                        padding = 15
                        cv2.rectangle(overlay,
                                    (x_start - padding, y_start - padding - 60),
                                    (x_end + padding, y_end + padding),
                                    (0, 0, 0),
                                    -1)
                        vis_frame = cv2.addWeighted(overlay, 0.7, vis_frame, 0.3, 0)
                        
                        # Insert cropped license plate
                        crop_h = min(H, actual_h)
                        crop_w = min(W, actual_w)
                        vis_frame[y_start:y_start+crop_h, x_start:x_start+crop_w] = \
                            license_crop[:crop_h, :crop_w]
                        
                        # Draw text ABOVE the plate crop
                        (text_width, text_height), _ = cv2.getTextSize(
                            license_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                        
                        text_x = max(10, car_center_x - text_width // 2)
                        text_y = max(text_height + 10, y_start - 20)
                        
                        # Ensure text is within frame
                        if text_x + text_width > vis_frame.shape[1]:
                            text_x = vis_frame.shape[1] - text_width - 10
                        
                        # Draw text with shadow for better visibility
                        cv2.putText(vis_frame, license_text, (text_x + 2, text_y + 2),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 5)
                        cv2.putText(vis_frame, license_text, (text_x, text_y),
                                   cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                        
                        # Draw confidence indicator
                        if aggregated_score:
                            conf_text = f"{aggregated_score:.2f}"
                            cv2.putText(vis_frame, conf_text, 
                                      (x_start, y_end + 25),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                except:
                    pass
        
        return vis_frame
    
    def run(self):
        """Main processing loop"""
        print("\n" + "-" * 70)
        print("Processing...")
        print("-" * 70)
        
        frame_nmr = -1
        processed_frames = 0
        start_time = time.time()
        total_vehicles = 0
        total_plates = 0
        
        try:
            while True:
                ret, frame = self.cap.read()
                
                if not ret or frame is None:
                    if self.source_type == 'video':
                        break
                    else:
                        continue
                
                frame_nmr += 1
                
                # Skip frames if specified
                if self.skip_frames > 0 and frame_nmr % (self.skip_frames + 1) != 0:
                    continue
                
                processed_frames += 1
                
                # Process frame
                vehicles_count, plates_count = self.process_frame(frame, frame_nmr)
                total_vehicles += vehicles_count
                total_plates += plates_count
                
                # Visualize if needed
                if self.live_view or self.save_video:
                    vis_frame = self.visualize_frame(frame, frame_nmr)
                    
                    # Add info overlay
                    elapsed = time.time() - start_time
                    fps_processing = processed_frames / elapsed if elapsed > 0 else 0
                    avg_ocr_time = (self.total_ocr_time / self.ocr_count * 1000) if self.ocr_count > 0 else 0
                    info_text = f"Frame: {frame_nmr} | FPS: {fps_processing:.1f} | Vehicles: {vehicles_count} | Plates: {plates_count}"
                    ocr_text = f"OCR: {avg_ocr_time:.1f}ms avg | Enhanced: {'ON' if self.use_enhanced_preprocessing else 'OFF'}"
                    
                    cv2.putText(vis_frame, info_text, (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    cv2.putText(vis_frame, ocr_text, (10, 60), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                    
                    if self.save_video:
                        self.video_writer.write(vis_frame)
                    
                    if self.live_view:
                        try:
                            if self.resize_display:
                                display_frame = cv2.resize(vis_frame, (1280, 720))
                            else:
                                display_frame = vis_frame
                            cv2.imshow('License Plate Recognition', display_frame)
                            
                            key = cv2.waitKey(1) & 0xFF
                            if key == ord('q'):
                                print("\n  User interrupted processing")
                                break
                        except cv2.error:
                            print("\n  ⚠️  GUI display error - disabling live view")
                            self.live_view = False
                
                # Progress indicator every 30 frames
                if processed_frames % 30 == 0:
                    elapsed = time.time() - start_time
                    fps_processing = processed_frames / elapsed if elapsed > 0 else 0
                    
                    if self.source_type == 'video':
                        progress = (frame_nmr / self.total_frames) * 100
                        eta = (self.total_frames - frame_nmr) / fps_processing if fps_processing > 0 else 0
                        print(f"  Frame {frame_nmr}/{self.total_frames} ({progress:.1f}%) | "
                              f"Speed: {fps_processing:.1f} FPS | ETA: {eta:.1f}s")
                    else:
                        print(f"  Frame {frame_nmr} | Speed: {fps_processing:.1f} FPS | "
                              f"Vehicles: {vehicles_count} | Plates: {plates_count}")
        
        except KeyboardInterrupt:
            print("\n  Processing interrupted by user")
        
        finally:
            self._cleanup()
            self._print_summary(processed_frames, time.time() - start_time, total_vehicles, total_plates)
    
    def _cleanup(self):
        """Clean up resources"""
        self.cap.release()
        if self.save_video:
            self.video_writer.release()
        if self.live_view:
            try:
                cv2.destroyAllWindows()
            except:
                pass
    
    def _print_summary(self, total_frames, total_time, total_vehicles, total_plates):
        """Print processing summary"""
        avg_fps = total_frames / total_time if total_time > 0 else 0
        avg_ocr_time = (self.total_ocr_time / self.ocr_count * 1000) if self.ocr_count > 0 else 0
        
        print("\n" + "=" * 70)
        print("Processing Complete!")
        print("=" * 70)
        print(f"\nStatistics:")
        print(f"  Total Frames Processed: {total_frames}")
        print(f"  Processing Time: {total_time:.2f} seconds")
        print(f"  Average FPS: {avg_fps:.2f}")
        print(f"  Total Vehicles Detected: {total_vehicles}")
        print(f"  Total License Plates Read: {total_plates}")
        print(f"  Device Used: {self.device.upper()}")
        
        # Multi-frame aggregation stats
        print(f"\nMulti-frame Aggregation Stats:")
        total_cars_tracked = len(self.plate_readings_per_car)
        total_readings = sum(len(readings) for readings in self.plate_readings_per_car.values())
        avg_readings_per_car = total_readings / total_cars_tracked if total_cars_tracked > 0 else 0
        print(f"  Total Cars Tracked: {total_cars_tracked}")
        print(f"  Average Readings per Car: {avg_readings_per_car:.1f}")
        
        # Performance breakdown
        print(f"\nPerformance Breakdown:")
        print(f"  YOLO Inference Time: {self.total_inference_time:.2f}s ({self.total_inference_time/total_time*100:.1f}%)")
        print(f"  OCR Processing Time: {self.total_ocr_time:.2f}s ({self.total_ocr_time/total_time*100:.1f}%)")
        print(f"  Average OCR Time: {avg_ocr_time:.2f}ms per plate")
        print(f"  Other Processing: {(total_time - self.total_inference_time - self.total_ocr_time):.2f}s")
        
        if self.device == 'cuda':
            speedup = avg_fps / self.fps if self.fps > 0 else 0
            print(f"  Processing Speed: {speedup:.2f}x real-time")
    
    def apply_aggregation_to_results(self):
        """
        Apply multi-frame aggregation to all results
        Updates each frame's license plate text with aggregated best text
        """
        print("\n" + "-" * 70)
        print("Applying Multi-frame Aggregation...")
        print("-" * 70)
        
        aggregation_improvements = 0
        
        for car_id in self.plate_readings_per_car.keys():
            best_text, best_score = self.aggregate_license_plate_text(car_id)
            
            if best_text:
                # Update all frames for this car with the aggregated text
                for frame_nmr in self.results.keys():
                    if car_id in self.results[frame_nmr]:
                        original_text = self.results[frame_nmr][car_id]['license_plate']['text']
                        if original_text != best_text:
                            aggregation_improvements += 1
                        
                        self.results[frame_nmr][car_id]['license_plate']['text'] = best_text
                        self.results[frame_nmr][car_id]['license_plate']['aggregated'] = True
                        self.results[frame_nmr][car_id]['license_plate']['aggregated_score'] = best_score
        
        print(f"  ✓ Aggregation applied to {len(self.plate_readings_per_car)} vehicles")
        print(f"  ✓ Improved {aggregation_improvements} readings")
    
    def save_results(self, csv_path='results.csv', interpolate=True):
        """Save results to CSV with optional interpolation"""
        print("\n" + "-" * 70)
        print("Saving Results...")
        print("-" * 70)
        
        # Apply aggregation before saving
        self.apply_aggregation_to_results()
        
        # Save raw results
        write_csv(self.results, csv_path)
        print(f"  ✓ Results saved to: {csv_path}")
        
        # Interpolate and save
        if interpolate and len(self.results) > 0:
            try:
                with open(csv_path, 'r') as f:
                    reader = csv.DictReader(f)
                    data = list(reader)
                
                if len(data) > 0:
                    interpolated_data = interpolate_bounding_boxes(data)
                    
                    interpolated_path = csv_path.replace('.csv', '_interpolated.csv')
                    header = ['frame_nmr', 'car_id', 'car_bbox', 'license_plate_bbox', 
                             'license_plate_bbox_score', 'license_number', 'license_number_score']
                    
                    with open(interpolated_path, 'w', newline='') as f:
                        writer = csv.DictWriter(f, fieldnames=header)
                        writer.writeheader()
                        writer.writerows(interpolated_data)
                    
                    print(f"  ✓ Interpolated results saved to: {interpolated_path}")
            except Exception as e:
                print(f"  ⚠ Could not interpolate results: {e}")


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='GPU-Accelerated License Plate Recognition System - ENHANCED',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with all enhancements
  python improved_unified_lpr.py --source sample.mp4
  
  # Disable enhanced preprocessing (faster but less accurate)
  python improved_unified_lpr.py --source sample.mp4 --no-enhanced-preprocessing
  
  # Use PaddleOCR with enhancements
  python improved_unified_lpr.py --source sample.mp4 --ocr-engine paddleocr
  
  # Use TensorRT models for maximum speed
  python improved_unified_lpr.py --source sample.mp4 --coco-model yolov8n_fp16.engine --license-model license_plate_detector_fp16.engine
  
  # Process video without display and save output
  python improved_unified_lpr.py --source sample.mp4 --no-display --save-video
        """
    )
    
    parser.add_argument('--source', type=str, default='webcam',
                       help='Video source: "webcam", video file path, or RTSP URL')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable live video display')
    parser.add_argument('--save-video', action='store_true',
                       help='Save output video with annotations')
    parser.add_argument('--output-video', type=str, default='output_enhanced.mp4',
                       help='Output video file path')
    parser.add_argument('--output-csv', type=str, default='results_enhanced.csv',
                       help='Output CSV file path')
    parser.add_argument('--no-interpolate', action='store_true',
                       help='Disable bounding box interpolation')
    parser.add_argument('--skip-frames', type=int, default=0,
                       help='Process every Nth frame (0=process all frames)')
    parser.add_argument('--conf', type=float, default=0.25,
                       help='Confidence threshold for detections (0.0-1.0)')
    parser.add_argument('--no-resize-display', action='store_true',
                       help='Display video at original resolution')
    parser.add_argument('--coco-model', type=str, default=None,
                       help='Path to COCO/vehicle detection model (.pt or .engine)')
    parser.add_argument('--license-model', type=str, default=None,
                       help='Path to license plate detection model (.pt or .engine)')
    
    # OCR-related arguments
    parser.add_argument('--no-gpu-ocr', action='store_true',
                       help='Disable GPU acceleration for OCR (use CPU)')
    parser.add_argument('--ocr-engine', type=str, default='easyocr',
                       choices=['easyocr', 'paddleocr'],
                       help='OCR engine to use')
    parser.add_argument('--batch-ocr', action='store_true',
                       help='Enable batch OCR processing')
    parser.add_argument('--no-enhanced-preprocessing', action='store_true',
                       help='Disable enhanced preprocessing pipeline')
    
    args = parser.parse_args()
    
    # Print configuration
    print("\n" + "=" * 70)
    print("Configuration")
    print("=" * 70)
    print(f"Source: {args.source}")
    print(f"OCR Engine: {args.ocr_engine.upper()}")
    print(f"GPU OCR: {'Enabled' if not args.no_gpu_ocr else 'Disabled'}")
    print(f"Batch OCR: {'Enabled' if args.batch_ocr else 'Disabled'}")
    print(f"Enhanced Preprocessing: {'Enabled' if not args.no_enhanced_preprocessing else 'Disabled'}")
    print(f"Multi-frame Aggregation: Enabled")
    if args.skip_frames > 0:
        print(f"Frame Skipping: Process every {args.skip_frames + 1} frames")
    print("=" * 70)
    
    # Initialize recognizer
    recognizer = LicensePlateRecognizer(
        source=args.source,
        live_view=not args.no_display,
        save_video=args.save_video,
        output_path=args.output_video,
        skip_frames=args.skip_frames,
        conf_threshold=args.conf,
        resize_display=not args.no_resize_display,
        coco_model_path=args.coco_model,
        license_plate_model_path=args.license_model,
        use_gpu_ocr=not args.no_gpu_ocr,
        ocr_engine=args.ocr_engine,
        batch_ocr=args.batch_ocr,
        use_enhanced_preprocessing=not args.no_enhanced_preprocessing
    )
    
    # Run processing
    recognizer.run()
    
    # Save results
    recognizer.save_results(
        csv_path=args.output_csv,
        interpolate=not args.no_interpolate
    )
    
    print("\n" + "=" * 70)


if __name__ == '__main__':
    main()