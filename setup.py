from setuptools import setup, find_packages

setup(
    name="grainvdb",
    version="2.0.0",
    description="Apple Silicon-Native Embedded Vector Store for Local-First RAG",
    author="GrainVDB Contributors",
    license="MIT",
    packages=find_packages(),
    package_data={"grainvdb": ["*.dylib", "*.metallib"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "numpy>=1.20.0",
    ],
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3",
        "Topic :: Database",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    entry_points={
        "console_scripts": [
            "grainvdb = grainvdb.cli:main",
            "agent-memory = grainvdb.cli:agent_memory_main",
        ],
    },
)
