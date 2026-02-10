from setuptools import setup, find_packages

setup(
    name="parameter-efficient-tuning",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "numpy>=1.24.0",
    ],
)
