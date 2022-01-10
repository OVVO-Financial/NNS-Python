# -*- coding: utf-8 -*-
# https://raw.githubusercontent.com/pypa/sampleproject/master/setup.py
# Always prefer setuptools over distutils
import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()
# Get the long description from the README file
long_description = (here / "README.md").read_text(encoding="utf-8")

# Arguments marked as "Required" below must be included for upload to PyPI.
# Fields marked as "Optional" may be commented out.
setup(
    name="NNS",
    version="0.1.3",
    description="Nonlinear nonparametric statistics using partial moments",
    long_description=long_description,
    long_description_content_type="text/markdown",  # Optional (see note above)
    author="Fred Viole, Roberto Spadim",
    author_email="ovvo.financial.systems@gmail.com",
    classifiers=[  # Optional
        # How mature is this project? Common values are
        #   3 - Alpha
        #   4 - Beta
        #   5 - Production/Stable
        "Development Status :: 3 - Alpha",
        # Indicate who your project is intended for
        # TODO: 'Intended Audience :: Developers',
        # Pick your license as you wish
        # TODO: 'License :: OSI Approved :: MIT License',
        # Specify the Python versions you support here. In particular, ensure
        # that you indicate you support Python 3. These classifiers are *not*
        # checked by 'pip install'. See instead 'python_requires' below.
        "Programming Language :: Python :: 3.7",
    ],
    keywords="Nonlinear nonparametric regression classification clustering",
    # When your source code is in a subdirectory under the project root, e.g.
    # `src/`, it is necessary to specify the `package_dir` argument.
    # package_dir={'': 'src'},  # Optional
    # You can just specify package directories manually here if your project is
    # simple. Or you can use find_packages().
    #
    # Alternatively, if you just want to distribute a single Python file, use
    # the `py_modules` argument instead as follows, which will expect a file
    # called `my_module.py` to exist:
    #
    #   py_modules=["my_module"],
    #
    # packages=find_packages(where='src'),
    packages=find_packages(),
    # Specify which Python versions you support. In contrast to the
    # 'Programming Language' classifiers above, 'pip install' will check this
    # and refuse to install the project if the version does not match. See
    # https://packaging.python.org/guides/distributing-packages-using-setuptools/#python-requires
    python_requires=">=3.7",
    # This field lists other packages that your project depends on to run.
    # Any package you put here will be installed by pip when your project is
    # installed, so they must be valid existing projects.
    #
    # For an analysis of "install_requires" vs pip's requirements files see:
    # https://packaging.python.org/en/latest/requirements.html
    install_requires=[
        "numpy>=1.19.0",
        "pandas>=1.0.5",
        "scikit_learn>=0.23.1",
        "KDEpy>=1.1.0",
    ],
    extras_require={
        "test": ["pytest-cov==2.10.0", "pytest==5.4.3"],
    },
    url="https://github.com/OVVO-Financial/NNS-Python/",
    # https://medium.com/@joel.barmettler/how-to-upload-your-python-package-to-pypi-65edc5fe9c56
    # build:  python setup.py sdist
    # upload: twine upload dist/*
    download_url="https://github.com/OVVO-Financial/NNS-Python/archive/refs/tags/v_013.tar.gz",
)
