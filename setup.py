from setuptools import setup, find_packages

setup(
    name="RetinaNetTransformer",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,  # Include non-code files like JSON
    install_requires=[
        "numpy",
        "pandas",
        "scikit-learn",
        "vipas"  # Replace with the actual version if needed
    ],
    entry_points={
        "console_scripts": [
            "processor=processor:main",  # Optional CLI entry point
        ]
    },
    package_data={
        "": ["utils/*.json", "utils/*.txt"],  # Include JSON and text files
    },
    author="Your Name",
    author_email="your.email@example.com",
    description="Package for RetinaNet Transformer",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ],
)
