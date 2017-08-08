from distutils.core import setup

setup(
    name='pcog',
    version='0.1dev',
    packages=['pcog',],
    license='Creative Commons Attribution-Noncommercial-Share Alike license',
    long_description=open('README.md').read(),
    install_requires=[],
    entry_points={
        'console_scripts': [
            'pcog = pcog.__main__:main',
        ],
    }
)
