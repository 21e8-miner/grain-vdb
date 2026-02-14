#!/usr/bin/env python3
"""
GrainVDB v2.0 - Breakthrough Edition
Setup script for Python package
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
readme_path = Path(__file__).parent / "README.md"
long_description = readme_path.read_text() if readme_path.exists() else ""

setup(
    name="grainvdb",
    version="2.0.0",
    author="GrainVDB Team",
    author_email="support@grainvdb.dev",
    description="High-Performance Vector Search Engine for Apple Silicon",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/grainvdb/grain-vdb",
    packages=find_packages(),
    package_data={
        "grainvdb": [
            "libgrainvdb.dylib",
            "gv_kernel.metallib",
        ],
    },
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
        "Operating System :: MacOS :: MacOS 12",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Programming Language :: Objective C++",
        "Programming Language :: Metal",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.21.0",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-benchmark>=4.0.0",
            "black>=22.0.0",
            "mypy>=0.950",
        ],
        "benchmark": [
            "matplotlib>=3.5.0",
            "pandas>=1.3.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "grainvdb-benchmark=benchmark:main",
        ],
    },
    keywords=[
        "vector search",
        "similarity search",
        "ann",
        "approximate nearest neighbor",
        "hnsw",
        "metal",
        "gpu",
        "apple silicon",
        "rag",
        "embeddings",
    ],
    project_urls={
        "Bug Reports": "https://github.com/grainvdb/grain-vdb/issues",
        "Source": "https://github.com/grainvdb/grain-vdb",
        "Documentation": "https://grainvdb.readthedocs.io",
    },
)
