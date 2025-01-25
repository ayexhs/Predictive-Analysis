from setuptools import setup, find_packages

setup(
    name='Topsis-Shreya-102217130',
    version='1.0.0',
    packages=find_packages(),
    install_requires=[
        'pandas',
        'numpy'
    ],
    author='Shreya',
    author_email='shreya.anshi04@gmail.com',
    description='A Python package for implementing the TOPSIS decision-making method.',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    url='https://github.com/shreya/Topsis-Shreya-102217130',
    classifiers=[
        'Programming Language :: Python :: 3',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
    ],
    entry_points={
        'console_scripts': [
            'topsis=102217130:topsis',
        ],
    },
)
