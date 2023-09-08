from setuptools import setup, find_packages

setup(
    name="lts",
    version="0.1",
    authors="Ibrahim Hroob",
    package_dir={"": "src"},
    description="long term stable points filter",
    packages=find_packages(where="src"),
    install_requires=[
        "numpy>=1.20.1",
        "PyYAML>=6.0",
        "tqdm>=4.62.3",
        "torch",
        "ninja",
    ],
)
