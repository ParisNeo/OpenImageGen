from setuptools import setup, find_packages

# Read the README.md file with UTF-8 encoding
try:
    with open("README.md", "r", encoding="utf-8") as f:
        long_description = f.read()
except FileNotFoundError:
    long_description = "Open source image generation API using diffusion models."


# Read requirements from requirements.txt
try:
    with open("requirements.txt", "r", encoding="utf-8") as f:
        install_requires = f.read().splitlines()
except FileNotFoundError:
    # Fallback requirements if file is missing
    install_requires=[
        "fastapi>=0.100.0",
        "uvicorn[standard]>=0.20.0",
        "pydantic>=2.0.0",
        "torch>=2.0.0",
        "diffusers>=0.26.0",
        "transformers>=4.30.0",
        "accelerate>=0.25.0",
        "Pillow>=9.0.0",
        "toml>=0.10.0",
        "pipmaster>=0.5.4" # Optional, for dependency checks in main.py
    ]

setup(
    name="openimagegen",
    version="0.1.0", # Start with initial version
    description="Open source image generation API using diffusion models",
    long_description=long_description,
    long_description_content_type="text/markdown",
    author="ParisNeo",
    author_email="parisneo_ai@gmail.com",
    url="https://https://github.com/ParisNeo/OpenImageGen",
    packages=find_packages(),
    include_package_data=True, # Include static files etc.
    install_requires=install_requires,
    entry_points={
        "console_scripts": [
            "openimagegen=openimagegen.cli:main" # Use cli.py as the entry point
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Multimedia :: Graphics :: Graphics Conversion", # Relevant topic
        "Framework :: FastAPI",
    ],
    python_requires=">=3.11",
)
