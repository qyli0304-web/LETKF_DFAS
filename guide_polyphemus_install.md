### Dependencies

- Installation steps:

  `Ubuntu 18.04 LTS` or `Ubuntu 20.04 LTS`

  Install `build-essential` which provides `gcc` and `g++`

  ```bash
  sudo apt update
  sudo apt install build-essential
  gcc --version
  g++ --version
  ```

  Then install `gfortran`

  ```bash
  sudo apt install gfortran
  gfortran --version
  ```

  

  Install `Python 2`. We recommend managing multiple Python versions with `conda`.
  First download Linux `Anaconda3` from a mirror below, then create a `Python 2.7`
  environment with `conda`.

  ```shell
  # install anaconda3
  # https://mirrors.tuna.tsinghua.edu.cn/anaconda/archive/
  bash Anaconda3-2023.09-Linux-x86_64.sh
  # install python 2 by using conda
  conda create --name py27 python=2.7
  # activate the new env
  conda activate py27
  ```

- Required external libraries and Python packages

  - Blitz++ (C++ library)
  - Blas/Lapack
  - NewRan: C++ random number generator (bundled; no manual install needed)
  - NetCDF
  - NumPy
  - Matplotlib
  - SciPy
  - WGRIB

  ```shell
  # ubuntu
  # Blitz++
  # sudo apt install libblitz0-dev libblitz0v5
  # Check Ubuntu version:
  lsb_release -a
  # Blitz versions compatible with 18.04 and 20.04:
  # http://fr.archive.ubuntu.com/ubuntu/pool/universe/b/blitz++/libblitz0-dev_1.0.2+ds-2_amd64.deb
  # http://fr.archive.ubuntu.com/ubuntu/pool/universe/b/blitz++/libblitz0v5_1.0.2+ds-2_amd64.deb
  # Then move to the directory where the files were downloaded
  cd ~/Downloads/
  sudo dpkg -i libblitz0v5_1.0.2+ds-2_amd64.deb
  sudo dpkg -i libblitz0-dev_1.0.2+ds-2_amd64.deb
  # Blas/Lapack
  sudo apt install libblas-dev liblapack-dev
  # netcdf, netcdf_c++
  sudo apt install libnetcdf-cxx-legacy-dev
  sudo apt install libnetcdf-dev
  # numpy matplotlib scipy
  conda activate py27
  conda install numpy matplotlib scipy
  # ipython
  conda activate py27
  conda install ipython
  ```

  If you encounter issues installing `ipython`, see the appendix below.

- Parallel computing



### Build and Installation

- After installing libraries and compilers, download the source code from the official site,
  then extract it to a fixed directory using one of the following commands (depending on the archive type):

  ```shell
  # https://cerea.enpc.fr/polyphemus/
  tar xvf Polyphemus.tar
  tar zxvf Polyphemus.tgz
  tar zxvf Polyphemus.tar.gz
  tar jxvf Polyphemus.tar.bz2
  ```

- Build atmopy

  ```shell
  cd Polyphemus-[version]/include/atmopy/talos
  ../../../utils/scons.py
  # or use following command if possible
  # scons
  ```

  Because `/Polyphemus-[version]/utils/scons.py` is used frequently, you can add it to your
  environment and use `scons` as an alias for `scons.py`. Add `~/Polyphemus-[version]/utils`
  to your `PATH` so you can invoke `scons` directly. For example:

  First edit your `.bashrc` (or equivalent shell rc). Example using `gedit`:

  ```shell
  # vi, vim, gedit ..
  gedit .bashrc
  ```

  Then append the following lines to the end of the file (assuming `Polyphemus-[version]`
  is installed under `home/[username]/`):

  ```bash
  alias scons="scons.py"
  export PATH="~/Polyphemus-[version]/utils:$PATH"
  ```

  Additionally, to import the project's Python modules during simulations, add
  `Polyphemus-[version]/include` to your `PYTHONPATH` in a similar manner:

  ```bash
  export PYTHONPATH="~/Polyphemus-[version]/include:$PYTHONPATH"
  ```

  The installation is now complete; you can proceed to build the project.