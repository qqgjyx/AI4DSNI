"""Setup script for AI4DSNI package."""

from pathlib import Path
from setuptools import setup, find_packages

# Read README
readme_path = Path(__file__).parent / "README.md"
if readme_path.exists():
    long_description = readme_path.read_text(encoding="utf-8")
else:
    long_description = "AI4DSNI: AI for Deep-Sea Niche Identification"

# Read requirements
requirements_path = Path(__file__).parent / "requirements.txt"
if requirements_path.exists():
    requirements = [
        line.strip()
        for line in requirements_path.read_text().splitlines()
        if line.strip() and not line.startswith("#")
    ]
else:
    requirements = [
        "torch>=2.0.0",
        "pytorch-lightning>=2.1.0",
        "hydra-core>=1.3.0",
        "numpy>=1.24.0",
        "pandas>=2.0.0",
        "scikit-learn>=1.3.0",
    ]

setup(
    name="ai4dsni",
    version="0.1.0",
    author="AI4DSNI Team",
    description="AI for Deep-Sea Niche Identification",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/your-org/ai4dsni",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.4.0",
            "pytest-cov>=4.1.0",
            "black>=23.0.0",
            "isort>=5.12.0",
            "mypy>=1.5.0",
        ],
        "wandb": [
            "wandb>=0.15.0",
        ],
    },
    entry_points={
        "console_scripts": [
            "ai4dsni-train=scripts.run:main",
        ],
    },
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
)
