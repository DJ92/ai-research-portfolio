from setuptools import setup, find_packages

setup(
    name="post-training-methods",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
)
