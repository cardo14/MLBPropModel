#!/usr/bin/env python3
"""
Setup script to initialize the MLB Prop Betting project structure
This script ensures all directories exist and files are in their proper locations
"""

import os
import sys
import shutil
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('setup_project')

def create_directory_structure(base_dir):
    """Create the required directory structure"""
    directories = [
        'data',
        'legacy_scripts',
        'scripts',
        'src/api',
        'src/data',
        'src/models',
        'src/utils',
    ]
    
    for directory in directories:
        dir_path = os.path.join(base_dir, directory)
        if not os.path.exists(dir_path):
            logger.info(f"Creating directory: {dir_path}")
            os.makedirs(dir_path, exist_ok=True)
        else:
            logger.info(f"Directory already exists: {dir_path}")

def create_init_files(base_dir):
    """Create __init__.py files in all Python package directories"""
    python_dirs = [
        'src',
        'src/api',
        'src/data',
        'src/models',
        'src/utils',
    ]
    
    for directory in python_dirs:
        init_file = os.path.join(base_dir, directory, '__init__.py')
        if not os.path.exists(init_file):
            logger.info(f"Creating __init__.py in {directory}")
            with open(init_file, 'w') as f:
                f.write('# This file makes the directory a Python package\n')
        else:
            logger.info(f"__init__.py already exists in {directory}")

def clean_pycache(base_dir):
    """Remove __pycache__ directories"""
    for root, dirs, files in os.walk(base_dir):
        if '__pycache__' in dirs:
            pycache_dir = os.path.join(root, '__pycache__')
            logger.info(f"Removing __pycache__ directory: {pycache_dir}")
            shutil.rmtree(pycache_dir)

def main():
    """Main function to set up the project structure"""
    # Get the base directory (project root)
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    logger.info(f"Setting up project in: {base_dir}")
    
    # Create directory structure
    create_directory_structure(base_dir)
    
    # Create __init__.py files
    create_init_files(base_dir)
    
    # Clean __pycache__ directories
    clean_pycache(base_dir)
    
    logger.info("Project setup complete!")

if __name__ == "__main__":
    main()
