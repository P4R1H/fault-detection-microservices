"""
Setup configuration for fault-detection-microservices project.

Install in development mode:
    pip install -e .

Install with dependencies:
    pip install -e .[dev]
"""

from setuptools import setup, find_packages

# Read requirements
with open('requirements.txt') as f:
    requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]

# Development dependencies
dev_requirements = [
    'pytest>=7.2.0',
    'pytest-cov>=4.0.0',
    'black>=22.10.0',
    'pylint>=2.15.0',
    'mypy>=0.991',
    'jupyter>=1.0.0',
]

setup(
    name='fault-detection-microservices',
    version='0.2.0',  # Phase 2 complete
    description='Multimodal Root Cause Analysis for Microservices',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Parth Gupta, Pratyush Jain, Vipul Kumar Chauhan',
    author_email='',
    url='https://github.com/P4R1H/fault-detection-microservices',
    packages=find_packages(where='.', exclude=['tests', 'experiments', 'outputs']),
    package_dir={'': '.'},
    python_requires='>=3.8',
    install_requires=requirements,
    extras_require={
        'dev': dev_requirements,
    },
    entry_points={
        'console_scripts': [
            'fd-download=scripts.download_dataset:main',
            'fd-verify=scripts.verify_dataset:main',
            'fd-eda=scripts.eda_analysis:main',
        ],
    },
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
    ],
    keywords='AIOps, microservices, root-cause-analysis, anomaly-detection, multimodal',
)
