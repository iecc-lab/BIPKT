from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    packages=find_packages(),
    python_requires=">=3.5",
    include_package_data=True,
    install_requires=['numpy>=1.17.2','pandas>=1.1.5','scikit-learn','torch>=1.7.0','wandb>=0.12.9','entmax'],
)
