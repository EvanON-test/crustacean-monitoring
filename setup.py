"""
Setup script for Crustacean Monitoring System.

This is a minimal setup.py for development installation.
A more comprehensive version will be created in Phase 4.
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
readme_file = Path(__file__).parent / "README.md"
long_description = readme_file.read_text(encoding="utf-8") if readme_file.exists() else ""

# Read requirements
requirements_file = Path(__file__).parent / "requirements.txt"
requirements = []
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [
            line.strip()
            for line in f
            if line.strip() and not line.startswith("#") and not line.startswith("-")
        ]

setup(
    name="crustacean-monitoring",
    version="2.0.0",
    author="Crustacean Monitoring Team",
    description="Real-time computer vision pipeline for detecting and analyzing crustaceans on edge devices",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/EvanON-test/crustacean-monitoring",
    packages=find_packages(exclude=["tests", "tests.*"]),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.7.0",
            "flake8>=6.1.0",
            "mypy>=1.5.0",
        ],
        "jetson": [
            "jetson-stats==4.3.2",
        ],
    },
    entry_points={
        "console_scripts": [
            # Entry points will be added in Phase 4
        ],
    },
    include_package_data=True,
    zip_safe=False,
)
