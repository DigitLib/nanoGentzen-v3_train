#!/usr/bin/env python3
"""
setup.py for nanoGentzen package.
"""
from setuptools import setup, find_packages

setup(
    name="nanogentzen",
    version="3.0.0",
    description="Saul Kripke (1965) Epistemic Neural Prover with Span Pointer Sequent Attention and Dyckhoff G4ip Calculus",
    long_description=open("README.md", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    license="MIT",
    packages=find_packages(),
    package_data={"nanogentzen": ["vocab.json", "config.json", "*.json"]},
    include_package_data=True,
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "safetensors>=0.4.0",
        "tqdm>=4.65.0",
    ],
)

