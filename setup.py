from setuptools import setup, find_packages

setup(
    name="clera",
    version="0.1.0",
    description="Cellular Latent Equations Representation and Analysis",
    author="Vasu Swaroop",  # Assuming user name based on path, generic otherwise
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "numpy",
        "scipy",
        "tensorflow<2.0",  # Explicitly marking the legacy requirement
        "matplotlib",
        "networkx",
        "scikit-learn",
    ],
    python_requires=">=3.6",
)
