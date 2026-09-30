from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

setup(
    packages=[],
    license_files=['vendor/LICENSE.ceres'],
    ext_modules=[Pybind11Extension(
        'geosunpre_native', ['depth_guard.cpp', 'vendor/cost_function.cc'],
        depends=['joint_costs.h', 'vendor/ceres/cost_function.h'],
        include_dirs=['vendor'], cxx_std=17,
    )],
    cmdclass={'build_ext': build_ext},
)
