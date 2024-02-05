import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="came-pytorch",
    license='MIT',
    version="0.1.3",
    author="Yang Luo",
    author_email="yangluo@comp.nus.edu.sg",
    description="CAME Optimizer - Pytorch Version",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yangluo7/CAME/",
    packages=setuptools.find_packages(),
    keywords=[
        'artificial intelligence',
        'deep learning',
        'optimizers',
        'memory efficient'
    ],
    install_requires=[
        'torch>=1.6'
    ],
    classifiers=[
        "Programming Language :: Python :: 3.6",
        "License :: OSI Approved :: MIT License",
        'Intended Audience :: Developers',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
    python_requires='>=3.6',
)
