import os
from setuptools import setup


def get_readme():
    """
    Convenience function to read in the contents of the README.md file
    such that it can be conveniently included in the "long_description".
    """
    return open(os.path.join(os.path.dirname(__file__), 'README.md')).read()


setup(
    name="akademy",
    version="0.1.1",
    url="https://github.com/alphazwest/akademy",
    author="Zack West",
    author_email="alphazwest@gmail.com",
    description="akademy: A Reinforcement Learning Framework",
    license="BSD-3-Clause",
    keywords=[
        "reinforcement learning", "quantitative trading", "fintech",
        "trading bot", "algorithmic trading", "finance", "automated trading",
        "neural networks", "artificial intelligence", "machine learning"
    ],
    packages=[
        'tests',
        'akademy',
        'akademy.common',
        'akademy.models',
        'akademy.models.agents',
        'akademy.models.envs',
        'akademy.models.base_models',
        'akademy.models.networks'
    ],
    long_description=get_readme(),

    # see here: https://pypi.org/classifiers/
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Operating System :: OS Independent",
        "Natural Language :: English",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",

        # optional GPU usage, but 11.7 CUDA required for GPU via PyTorch
        "Environment :: GPU :: NVIDIA CUDA :: 11.7"
    ],
    # note: PyTorch allows 3.7 for Windows, but requires >=3.7.6 for Linux
    #       but formally recommends >=3.8.1 when building from source.
    #       see here: https://pypi.org/project/torch/ (check version == 1.13.1)
    python_requires=">=3.7.6",
    install_requires=[
        "pandas==1.5.2",
        "torch==1.13.1",
        "gymnasium==0.27.0"
    ],
    provides=["akademy"]
)
