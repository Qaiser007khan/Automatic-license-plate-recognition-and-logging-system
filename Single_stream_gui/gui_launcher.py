"""
License Plate Recognition GUI Launcher
Simple launcher script with dependency checking
"""

import sys
import os

def check_dependencies():
    """Check if all required dependencies are installed"""
    missing = []
    
    dependencies = [
        ('tkinter', 'tkinter (usually comes with Python)'),
        ('cv2', 'opencv-python'),
        ('numpy', 'numpy'),
        ('torch', 'torch'),
        ('ultralytics', 'ultralytics'),
    ]
    
    print("Checking dependencies...")
    for module, package in dependencies:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ✗ {package} - NOT FOUND")
            missing.append(package)
    
    if missing:
        print("\n❌ Missing dependencies!")
        print("\nInstall missing packages:")
        print(f"  pip install {' '.join(missing)}")
        return False
    
    print("\n✓ All dependencies found!")
    return True


def check_files():
    """Check if required files exist"""
    required_files = [
        ('gui_lpr.py', 'GUI application'),
        ('unified_lpr.py', 'Main processing script'),
        ('util.py', 'Utility functions'),
    ]
    
    print("\nChecking required files...")
    missing = []
    
    for filename, description in required_files:
        if os.path.exists(filename):
            print(f"  ✓ {filename} ({description})")
        else:
            print(f"  ✗ {filename} ({description}) - NOT FOUND")
            missing.append(filename)
    
    if missing:
        print("\n❌ Missing required files!")
        print("\nMake sure these files are in the same directory:")
        for f in missing:
            print(f"  - {f}")
        return False
    
    print("\n✓ All required files found!")
    return True


def main():
    """Main launcher function"""
    print("=" * 70)
    print("License Plate Recognition System - GUI Launcher")
    print("=" * 70)
    print()
    
    # Check dependencies
    if not check_dependencies():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Check files
    if not check_files():
        input("\nPress Enter to exit...")
        sys.exit(1)
    
    # Launch GUI
    print("\n" + "=" * 70)
    print("Launching GUI...")
    print("=" * 70)
    print()
    
    try:
        import gui_lpr
        gui_lpr.main()
    except Exception as e:
        print(f"\n❌ Error launching GUI: {e}")
        import traceback
        traceback.print_exc()
        input("\nPress Enter to exit...")
        sys.exit(1)


if __name__ == "__main__":
    main()