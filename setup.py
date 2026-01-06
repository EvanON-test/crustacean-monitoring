"""
Setup script for Crustacean Monitoring System.

This setup.py enables installation of the crustacean package and provides
command-line entry points for running the pipelines.

Installation Options:

1. Docker (Recommended for Jetson):
    docker-compose build
    docker-compose run realtime
    docker-compose run offline

2. Native Installation:
    pip install -e .                    # Development install
    pip install -e ".[dev]"             # With development dependencies
    pip install -e ".[jetson]"          # With Jetson-specific dependencies
    pip install -e ".[dev,jetson]"      # With all extras

After native installation, you can run:
    crustacean-offline --video-dir ./videos
    crustacean-realtime --display
    crustacean-monitor --video-dir ./videos --output metrics.csv
"""

from setuptools import setup, find_packages
from pathlib import Path

# Package metadata
NAME = "crustacean-monitoring"
VERSION = "2.0.0"
AUTHOR = "Crustacean Monitoring Team"
AUTHOR_EMAIL = ""
DESCRIPTION = "Real-time computer vision pipeline for detecting and analyzing crustaceans on edge devices"
URL = "https://github.com/EvanON-test/crustacean-monitoring"
LICENSE = "MIT"

# Read the README file for long description
readme_file = Path(__file__).parent / "README.md"
long_description = ""
if readme_file.exists():
    long_description = readme_file.read_text(encoding="utf-8")

# Read requirements from requirements.txt
requirements_file = Path(__file__).parent / "requirements.txt"
install_requires = []
if requirements_file.exists():
    with open(requirements_file) as f:
        for line in f:
            line = line.strip()
            # Skip empty lines, comments, and -r includes
            if line and not line.startswith("#") and not line.startswith("-"):
                install_requires.append(line)

# Development dependencies
dev_requires = [
    "pytest>=7.4.0",
    "pytest-cov>=4.1.0",
    "pytest-timeout>=2.1.0",
    "black>=23.7.0",
    "flake8>=6.1.0",
    "mypy>=1.5.0",
    "isort>=5.12.0",
]

# Jetson-specific dependencies
jetson_requires = [
    "jetson-stats>=4.2.0",
]

# Raspberry Pi dependencies
pi_requires = [
    "gpiozero>=1.6.0",
]

setup(
    name=NAME,
    version=VERSION,
    author=AUTHOR,
    author_email=AUTHOR_EMAIL,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=URL,
    license=LICENSE,
    
    # Package discovery
    packages=find_packages(exclude=[
        "tests",
        "tests.*",
        "docs",
        "scripts",
        "processing",
        "benchmark",
        "realtime_frames",
    ]),
    
    # Package data (config files, etc.)
    package_data={
        "crustacean": [
            "py.typed",  # PEP 561 marker for type hints
        ],
    },
    
    # Include non-Python files specified in MANIFEST.in
    include_package_data=True,
    
    # Python version requirement
    python_requires=">=3.9",
    
    # Dependencies
    install_requires=install_requires,
    
    # Optional dependencies
    extras_require={
        "dev": dev_requires,
        "jetson": jetson_requires,
        "pi": pi_requires,
        "all": dev_requires + jetson_requires + pi_requires,
    },
    
    # Command-line entry points
    entry_points={
        "console_scripts": [
            # Main pipeline commands
            "crustacean-offline=scripts.run_offline:main",
            "crustacean-realtime=scripts.run_realtime:main",
            "crustacean-monitor=scripts.run_monitoring:main",
        ],
    },
    
    # PyPI classifiers
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Multimedia :: Video :: Capture",
        "License :: OSI Approved :: MIT License",
        "Operating System :: POSIX :: Linux",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: GPU :: NVIDIA CUDA",
        "Typing :: Typed",
    ],
    
    # Keywords for PyPI search
    keywords=[
        "computer-vision",
        "deep-learning",
        "object-detection",
        "keypoint-detection",
        "crustacean",
        "marine-biology",
        "edge-computing",
        "jetson-nano",
        "tflite",
        "real-time",
    ],
    
    # Project URLs
    project_urls={
        "Bug Reports": f"{URL}/issues",
        "Source": URL,
        "Documentation": f"{URL}#readme",
    },
    
    # Don't zip the package (needed for some resources)
    zip_safe=False,
)
