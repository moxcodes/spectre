#!/bin/env sh

# Distributed under the MIT License.
# See LICENSE.txt for details.

# Load system modules
spectre_load_sys_modules() {
    module load gcc/9.2.0
    module load mkl/18.1
    module load gsl/2.4
    module load hdf5/1.10.1
    module load boost/1_68_0-gcc730
    module load cmake/3.18.0
    module load python3/3.8.5
}

# Unload system modules
spectre_unload_sys_modules() {
    module unload boost/1_68_0-gcc730
    module unload hdf5/1.10.1
    module unload gsl/2.4
    module unload mkl/18.1
    module unload gcc/9.2.0
    module unload cmake/3.18.0
    module unload python3/3.8.5
}


spectre_setup_modules() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi

    local start_dir=`pwd`
    dep_dir=`realpath $1`
    if [ $# != 1 ]; then
        echo "You must pass one argument to spectre_setup_modules, which"
        echo "is the directory where you want the dependencies to be built."
        return 1
    fi
    mkdir -p $dep_dir
    cd $dep_dir
    mkdir -p $dep_dir/modules

    spectre_load_sys_modules

    if [ -f catch/include/catch.hpp ]; then
        echo "Catch is already installed"
    else
        echo "Installing catch..."
        mkdir -p $dep_dir/catch/include
        cd $dep_dir/catch/include
        wget https://github.com/catchorg/Catch2/releases/download/v2.13.0/catch.hpp -O catch.hpp
        echo "Installed Catch into $dep_dir/catch"
cat >$dep_dir/modules/catch <<EOF
#%Module1.0
prepend-path CPATH "$dep_dir/catch/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/catch/"
EOF
    fi
    cd $dep_dir

    if [ -f blaze/include/blaze/Blaze.h ]; then
        echo "Blaze is already installed"
    else
        echo "Installing Blaze..."
        mkdir -p $dep_dir/blaze/
        cd $dep_dir/blaze/
        wget https://bitbucket.org/blaze-lib/blaze/downloads/blaze-3.8.tar.gz -O blaze.tar.gz
        tar -xzf blaze.tar.gz
        mv blaze-* include
        ln -s include/blaze-3.8/blaze include/blaze
        echo "Installed Blaze into $dep_dir/blaze"
        cat >$dep_dir/modules/blaze <<EOF
#%Module1.0
prepend-path CPATH "$dep_dir/blaze/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/blaze/"
EOF
    fi
    cd $dep_dir


    if [ -f brigand/include/brigand/brigand.hpp ]; then
        echo "Brigand is already installed"
    else
        echo "Installing Brigand..."
        rm -rf $dep_dir/brigand
        git clone https://github.com/edouarda/brigand.git
        echo "Installed Brigand into $dep_dir/brigand"
        cat >$dep_dir/modules/brigand <<EOF
#%Module1.0
prepend-path CPATH "$dep_dir/brigand/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/brigand/"
EOF
    fi
    cd $dep_dir

    if [ -f libxsmm/lib/libxsmm.a ]; then
        echo "LIBXSMM is already installed"
    else
        echo "Installing LIBXSMM..."
        rm -rf $dep_dir/libxsmm
        wget https://github.com/hfp/libxsmm/archive/1.16.1.tar.gz -O libxsmm.tar.gz
        tar -xzf libxsmm.tar.gz
        mv libxsmm-* libxsmm
        cd libxsmm
        make CXX=g++ CC=gcc FC=gfortran -j4
        cd $dep_dir
        rm libxsmm.tar.gz
        echo "Installed LIBXSMM into $dep_dir/libxsmm"
        cat >$dep_dir/modules/libxsmm <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/libxsmm/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/libxsmm/lib"
prepend-path CPATH "$dep_dir/libxsmm/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/libxsmm/"
EOF
    fi
    cd $dep_dir

    if [ -f $dep_dir/yaml-cpp/lib/libyaml-cpp.a ]; then
        echo "yaml-cpp is already installed"
    else
        echo "Installing yaml-cpp..."
        wget https://github.com/jbeder/yaml-cpp/archive/yaml-cpp-0.6.2.tar.gz -O yaml-cpp.tar.gz
        tar -xzf yaml-cpp.tar.gz
        mv yaml-cpp-* yaml-cpp-build
        cd $dep_dir/yaml-cpp-build
        mkdir build
        cd build
        cmake -D CMAKE_BUILD_TYPE=Release -D YAML_CPP_BUILD_TESTS=OFF \
              -D CMAKE_C_COMPILER=gcc -D CMAKE_CXX_COMPILER=g++ \
              -D YAML_CPP_BUILD_CONTRIB=OFF \
              -D YAML_CPP_BUILD_TOOLS=ON \
              -D CMAKE_INSTALL_PREFIX=$dep_dir/yaml-cpp ..
        make -j4
        make install
        cd $dep_dir
        rm -r yaml-cpp-build
        rm -r yaml-cpp.tar.gz
        echo "Installed yaml-cpp into $dep_dir/yaml-cpp"
        cat >$dep_dir/modules/yaml-cpp <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/yaml-cpp/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/yaml-cpp/lib"
prepend-path CPATH "$dep_dir/yaml-cpp/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/yaml-cpp/"
EOF
    fi
    cd $dep_dir

    if [ -f $dep_dir/libsharp/lib/libsharp.a ]; then
        echo "libsharp is already installed"
    else
        echo "Installing libsharp..."
        wget https://github.com/Libsharp/libsharp/archive/v1.0.0.tar.gz -O libsharp.tar.gz
        tar -xzf libsharp.tar.gz
        mv libsharp-* libsharp_build
        cd $dep_dir/libsharp_build
        autoconf
        ./configure --prefix=$dep_dir/libsharp --disable-openmp
        make -j4
        mv ./auto $dep_dir/libsharp
        cd $dep_dir
        rm -r libsharp_build
        rm libsharp.tar.gz
        echo "Installed libsharp into $dep_dir/libsharp"
        cat >$dep_dir/modules/libsharp <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/libsharp/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/libsharp/lib"
prepend-path CPATH "$dep_dir/libsharp/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/libsharp/"
EOF
    fi
    cd $dep_dir

    # Set up Charm++ because that can be difficult
    if [ -f $dep_dir/charm/verbs-linux-x86_64-smp/lib/libck.a ]; then
        echo "Charm++ is already installed"
    else
        echo "Installing Charm++..."
        wget https://github.com/UIUC-PPL/charm/archive/v6.10.2.tar.gz
        tar xzf v6.10.2.tar.gz
        mv charm-6.10.2 charm
        cd $dep_dir/charm
        ./build charm++ verbs-linux-x86_64-smp --with-production -j4
        ./build LIBS verbs-linux-x86_64-smp --with-production -j4
        cd $dep_dir
        rm v6.10.2.tar.gz
        echo "Installed Charm++ into $dep_dir/charm"
        cat >$dep_dir/modules/charm <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/charm/verbs-linux-x86_64-smp/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/charm/verbs-linux-x86_64-smp/lib"
prepend-path CPATH "$dep_dir/charm/verbs-linux-x86_64-smp/include"
prepend-path CMAKE_PREFIX_PATH "$dep_dir/charm/verbs-linux-x86_64-smp/"
prepend-path PATH "$dep_dir/charm/verbs-linux-x86_64-smp/bin"
setenv CHARM_VERSION 6.10.2
setenv CHARM_HOME $dep_dir/charm/verbs-linux-x86_64-smp
setenv CHARM_ROOT $dep_dir/charm/verbs-linux-x86_64-smp
EOF
    fi
    cd $dep_dir

    if [ -f $dep_dir/modules/spectre_boost ]; then
        echo "Boost is already set up"
    else
        cat >$dep_dir/modules/spectre_boost <<EOF
#%Module1.0
prepend-path CPATH "$BOOST_HOME/include"
setenv BOOST_ROOT "$BOOST_HOME"
EOF
    fi
    cd $dep_dir

    if [ -f $dep_dir/modules/spectre_gsl ]; then
        echo "GSL is already set up"
    else
        # This hard-coded gsl path is not easily maintainable, so if and when
        # Caltech HPC assigns an accessible environment variable for the loaded
        # GSL module, we should use that in favor of the hard-coded path
        cat >$dep_dir/modules/spectre_gsl <<EOF
#%Module1.0
setenv GSL_HOME "/software/gsl/2.4"
setenv GSL_ROOT "/software/gsl/2.4"
EOF
    fi
    cd $dep_dir

    if [ -f $dep_dir/jemalloc/lib/libjemalloc.a ]; then
        echo "jemalloc is already set up"
    else
        echo "Installing jemalloc..."
        wget https://github.com/jemalloc/jemalloc/releases/download/5.2.1/jemalloc-5.2.1.tar.bz2
        tar -xjf jemalloc-5.2.1.tar.bz2
        mv jemalloc-5.2.1 jemalloc
        cd $dep_dir/jemalloc
        ./autogen.sh
        make
        cd $dep_dir
        rm jemalloc-5.2.1.tar.bz2
        echo "Installed jemalloc into $dep_dir/jemalloc"
        cat >$dep_dir/modules/jemalloc <<EOF
#%Module1.0
prepend-path LIBRARY_PATH "$dep_dir/jemalloc/lib"
prepend-path LD_LIBRARY_PATH "$dep_dir/jemalloc/lib"
prepend-path CPATH "$dep_dir/jemalloc/include"
prepend-path PATH "$dep_dir/jemalloc/bin"
prepend-path LD_RUN_PATH "$dep_dir/jemalloc/lib"
setenv JEMALLOC_HOME $dep_dir/jemalloc/
prepend-path CMAKE_PREFIX_PATH "$dep_dir/jemalloc/"
EOF
    fi

    cd $start_dir

    spectre_unload_sys_modules

    printf "\n\nIMPORTANT!!!\nIn order to be able to use these modules you\n"
    echo "must run:"
    echo "  module use $dep_dir/modules"
    echo "You will need to do this every time you compile SpECTRE, so you may"
    echo "want to add it to your ~/.bashrc."
}

spectre_unload_modules() {
    module unload charm
    module unload yaml-cpp
    module unload spectre_boost
    module unload spectre_gsl
    module unload jemalloc
    module unload libxsmm
    module unload libsharp
    module unload catch
    module unload brigand
    module unload blaze_37

    spectre_unload_sys_modules
}

spectre_load_modules() {
    spectre_load_sys_modules

    module load blaze_37
    module load brigand
    module load catch
    module load libsharp
    module load libxsmm
    module load jemalloc
    module load spectre_boost
    module load spectre_gsl
    module load yaml-cpp
    module load charm
}

spectre_run_cmake() {
    if [ -z ${SPECTRE_HOME} ]; then
        echo "You must set SPECTRE_HOME to the cloned SpECTRE directory"
        return 1
    fi
    spectre_load_modules
    cmake -D CHARM_ROOT=$CHARM_ROOT \
          -D CMAKE_BUILD_TYPE=Release \
          -D CMAKE_Fortran_COMPILER=gfortran \
          -D MEMORY_ALLOCATOR=JEMALLOC \
          -D BUILD_PYTHON_BINDINGS=off \
          -D Python_EXECUTABLE=`which python3` \
          "$@" \
          $SPECTRE_HOME
}
