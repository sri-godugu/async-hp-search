from setuptools import setup, find_packages

setup(
    name='async-hp-search',
    version='0.1.0',
    packages=find_packages(),
    python_requires='>=3.10',
    install_requires=['trio>=0.25.0', 'pyyaml>=6.0.1'],
)
