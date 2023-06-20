from setuptools import setup, find_packages

setup(name='starworld_tunnel_mpc',
      version='1.0',
      packages=find_packages(),
      install_requires=[
          'pyyaml',
          'numpy',
          'scipy',
          'matplotlib',
          'shapely',
          'casadi',
          'opengen',
          'opencv-python'
      ]
)
