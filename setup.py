from setuptools import setup, find_packages

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name="geoclip",
    version="1.1.0",
    packages=find_packages(),
    description="",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    url="https://github.com/VicenteVivan/geo-clip",
    author="Vicente Vivanco",
    author_email="vicente.vivancocepeda@ucf.edu",
    license="MIT",
    install_requires=requirements,
    include_package_data=True,
    python_requires=">=3.6",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
 