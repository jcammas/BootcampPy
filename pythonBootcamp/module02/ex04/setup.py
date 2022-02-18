#!/usr/bin/env python3
import setuptools

setuptools.setup(
    name='my_minipack',
    version="1.0.0",
    author='jcammas',
    author_email='jcammas@student.42.fr',
    description="How to create a package in python.",
    long_description="ah ouais ouais",
    long_description_content_type="text/markdown",
    url="None",
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    license="GPLv3",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Students",
        "Topic :: Education",
        "Topic :: HowTo",
        "Topic :: Package",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3 :: Only",
    ],
    python_requires='>=3.7',
)
