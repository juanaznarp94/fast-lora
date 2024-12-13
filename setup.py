from setuptools import setup, find_packages

with open("requirements.txt") as f:
    requirements = f.read().splitlines()

setup(
    name="FAST-LoRa",
    version="0.1",
    include_package_data=True,
    python_requires=">=3.8",
    packages=find_packages(),
    setup_requires=["setuptools-git-versioning"],
    install_requires=requirements,
    author="Fabian MARGREITER",
    author_email="fabian.margreiter@gmail.com",
    url="https://github.com/EmmArrGee/FAST-LoRa",
    description="FAST-LoRa - Fast Analytical Simulation Toolkit for LoRa Networks",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    license="GPLv3",
    classifiers=[
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    version_config={
        "dirty_template": "{tag}",
    },
)
