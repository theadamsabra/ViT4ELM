import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="vit4elm",
    version="1.3.3",
    author="Adam Sabra",
    description="Vision Transformers for Exotic Lattice Models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=setuptools.find_packages(),
    install_requires=[
        'scikit-learn',
        'numpy',
        'matplotlib',
        'pandas',
        'transformers',
        'datasets',
        'torch',
        'torchvision',
        'pillow',
        'twine'
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
    python_requires=">=3.8"
)