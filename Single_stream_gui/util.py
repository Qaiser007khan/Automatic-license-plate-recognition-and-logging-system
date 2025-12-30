"""
Utility functions for License Plate Recognition
Now with GPU-accelerated OCR support
"""

import string
import easyocr
import numpy as np

# Global OCR reader instance (initialized once)
_ocr_reader = None
_ocr_type = None


def initialize_ocr(use_gpu=True, ocr_engine='easyocr'):
    """
    Initialize OCR reader globally (call once at startup)
    
    Args:
        use_gpu: Whether to use GPU acceleration
        ocr_engine: 'easyocr' or 'paddleocr'
    
    Returns:
        OCR reader instance
    """
    global _ocr_reader, _ocr_type
    
    print(f"\nInitializing {ocr_engine.upper()} with GPU: {use_gpu}")
    
    if ocr_engine == 'easyocr':
        try:
            # EasyOCR with GPU support
            _ocr_reader = easyocr.Reader(
                ['en'], 
                gpu=use_gpu,
                verbose=False
            )
            _ocr_type = 'easyocr'
            print(f"✓ EasyOCR initialized (GPU: {use_gpu})")
        except Exception as e:
            print(f"✗ EasyOCR GPU initialization failed: {e}")
            print("  Falling back to CPU...")
            _ocr_reader = easyocr.Reader(['en'], gpu=False, verbose=False)
            _ocr_type = 'easyocr'
    
    elif ocr_engine == 'paddleocr':
        try:
            from paddleocr import PaddleOCR
            import logging
            
            # Suppress PaddleOCR logging
            logging.getLogger('ppocr').setLevel(logging.ERROR)
            
            # PaddleOCR with GPU support
            _ocr_reader = PaddleOCR(
                use_angle_cls=True,
                lang='en',
                use_gpu=use_gpu,
                show_log=False  # This controls internal logging
            )
            _ocr_type = 'paddleocr'
            print(f"✓ PaddleOCR initialized (GPU: {use_gpu})")
        except TypeError as e:
            # Handle parameter name issues with different PaddleOCR versions
            try:
                from paddleocr import PaddleOCR
                import logging
                logging.getLogger('ppocr').setLevel(logging.ERROR)
                
                _ocr_reader = PaddleOCR(
                    use_angle_cls=True,
                    lang='en',
                    use_gpu=use_gpu
                    # Remove show_log parameter for older versions
                )
                _ocr_type = 'paddleocr'
                print(f"✓ PaddleOCR initialized (GPU: {use_gpu})")
            except Exception as e2:
                print(f"✗ PaddleOCR initialization failed: {e2}")
                print("  Falling back to EasyOCR...")
                _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
                _ocr_type = 'easyocr'
        except ImportError:
            print("✗ PaddleOCR not installed. Install with: pip install paddleocr paddlepaddle-gpu")
            print("  Falling back to EasyOCR...")
            _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            _ocr_type = 'easyocr'
        except Exception as e:
            print(f"✗ PaddleOCR initialization failed: {e}")
            print("  Falling back to EasyOCR...")
            _ocr_reader = easyocr.Reader(['en'], gpu=use_gpu, verbose=False)
            _ocr_type = 'easyocr'
    
    return _ocr_reader


def get_ocr_reader():
    """Get the global OCR reader instance"""
    global _ocr_reader, _ocr_type
    if _ocr_reader is None:
        # Initialize with default settings if not already done
        initialize_ocr(use_gpu=True, ocr_engine='easyocr')
    return _ocr_reader, _ocr_type


def read_license_plate(license_plate_crop):
    """
    Read license plate text using GPU-accelerated OCR
    
    Args:
        license_plate_crop: Preprocessed license plate image (grayscale/thresholded)
    
    Returns:
        tuple: (license_plate_text, confidence_score)
    """
    reader, ocr_type = get_ocr_reader()
    
    # Dictionary for character mapping
    dict_char_to_int = {'O': '0',
                        'I': '1',
                        'J': '3',
                        'A': '4',
                        'G': '6',
                        'S': '5'}

    dict_int_to_char = {'0': 'O',
                        '1': 'I',
                        '3': 'J',
                        '4': 'A',
                        '6': 'G',
                        '5': 'S'}

    try:
        if ocr_type == 'easyocr':
            # EasyOCR
            detections = reader.readtext(license_plate_crop)
            
            if not detections:
                return None, None
            
            # Get text with highest confidence
            bbox, text, score = max(detections, key=lambda x: x[2])
            text = text.upper().replace(' ', '')
            
        elif ocr_type == 'paddleocr':
            # PaddleOCR
            result = reader.ocr(license_plate_crop, cls=True)
            
            if not result or not result[0]:
                return None, None
            
            # PaddleOCR returns list of [bbox, (text, confidence)]
            text_results = [(line[1][0], line[1][1]) for line in result[0]]
            
            if not text_results:
                return None, None
            
            # Get text with highest confidence
            text, score = max(text_results, key=lambda x: x[1])
            text = text.upper().replace(' ', '')
        
        else:
            return None, None
        
        # Format license plate text
        if license_plate_complies_format(text):
            return format_license(text, dict_char_to_int, dict_int_to_char), score
        
        return text, score
        
    except Exception as e:
        print(f"OCR Error: {e}")
        return None, None


def license_plate_complies_format(text):
    """
    Check if the license plate text complies with the required format.
    Format: At least 2 letters and numbers
    
    Args:
        text (str): License plate text.
    
    Returns:
        bool: True if format complies, False otherwise.
    """
    if len(text) < 5 or len(text) > 10:
        return False
    
    # Check if it has both letters and numbers
    has_letter = any(c.isalpha() for c in text)
    has_number = any(c.isdigit() for c in text)
    
    return has_letter and has_number


def format_license(text, dict_char_to_int, dict_int_to_char):
    """
    Format the license plate text by converting characters using the provided mappings.
    
    Args:
        text (str): License plate text.
        dict_char_to_int (dict): Mapping of characters to integers.
        dict_int_to_char (dict): Mapping of integers to characters.
    
    Returns:
        str: Formatted license plate text.
    """
    license_plate_ = ''
    mapping = {0: dict_int_to_char, 1: dict_int_to_char, 4: dict_int_to_char,
               5: dict_int_to_char, 6: dict_int_to_char,
               2: dict_char_to_int, 3: dict_char_to_int}
    
    for j in range(len(text)):
        if text[j] in mapping.get(j, {}):
            license_plate_ += mapping[j][text[j]]
        else:
            license_plate_ += text[j]
    
    return license_plate_


def get_car(license_plate, vehicle_track_ids):
    """
    Retrieve the vehicle coordinates and ID based on the license plate coordinates.
    
    Args:
        license_plate (tuple): Tuple containing information about the license plate.
        vehicle_track_ids (list): List of vehicle track IDs and their coordinates.
    
    Returns:
        tuple: Vehicle coordinates and ID.
    """
    x1, y1, x2, y2, score, class_id = license_plate
    
    foundIt = False
    for j in range(len(vehicle_track_ids)):
        xcar1, ycar1, xcar2, ycar2, car_id = vehicle_track_ids[j]
        
        if x1 > xcar1 and y1 > ycar1 and x2 < xcar2 and y2 < ycar2:
            car_indx = j
            foundIt = True
            break
    
    if foundIt:
        return vehicle_track_ids[car_indx]
    
    return -1, -1, -1, -1, -1


# Batch processing support for multiple license plates
def read_license_plates_batch(license_plate_crops):
    """
    Read multiple license plates in a batch (more efficient for GPU)
    
    Args:
        license_plate_crops: List of preprocessed license plate images
    
    Returns:
        list: List of tuples (text, score) for each plate
    """
    if not license_plate_crops:
        return []
    
    reader, ocr_type = get_ocr_reader()
    results = []
    
    try:
        if ocr_type == 'easyocr':
            # EasyOCR doesn't support batch processing well, process individually
            for crop in license_plate_crops:
                text, score = read_license_plate(crop)
                results.append((text, score))
        
        elif ocr_type == 'paddleocr':
            # PaddleOCR can process multiple images
            ocr_results = reader.ocr_batch(license_plate_crops, cls=True)
            
            dict_char_to_int = {'O': '0', 'I': '1', 'J': '3', 'A': '4', 'G': '6', 'S': '5'}
            dict_int_to_char = {'0': 'O', '1': 'I', '3': 'J', '4': 'A', '6': 'G', '5': 'S'}
            
            for result in ocr_results:
                if not result or not result[0]:
                    results.append((None, None))
                    continue
                
                text_results = [(line[1][0], line[1][1]) for line in result[0]]
                
                if not text_results:
                    results.append((None, None))
                    continue
                
                text, score = max(text_results, key=lambda x: x[1])
                text = text.upper().replace(' ', '')
                
                if license_plate_complies_format(text):
                    text = format_license(text, dict_char_to_int, dict_int_to_char)
                
                results.append((text, score))
        
        return results
        
    except Exception as e:
        print(f"Batch OCR Error: {e}")
        # Fallback to individual processing
        return [read_license_plate(crop) for crop in license_plate_crops]