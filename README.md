# Build Overview
There are two main complexities to this build. 

1. We require python-2.7s libraries
2. We need to build an old version of AI-toolbox because the current version requires C++ 17. We also require three C++ libraries to use AI-toolbox
3. We need to copy two shared libraries from AI-toolbox to PCog

# Prerequisits
This guide requires an ubuntu 16.04 box and python-2.7 installed. It also assumes that you have installed the QCog javacode base and the unity3d test bed.

# Dependencies

This command should install all packages required to build and run PCog. 

```
sudo apt-get update
sudo apt-get install libeigen3-dev liblpsolve55-dev python-tk graphviz python-pygraphviz libboost-all-dev build-essential
```

# Building AI-toolbox
In this section we build AI-toolbox so that we can use its shared libraries to run PCog. 

First clone AI-toolbox

```
git clone https://github.com/Ivan1931/AI-Toolbox.git
```

Now build the libraries:

```
cd AI-toolbox
git checkout origin/c++14-build
mkdir build
cd build
cmake ..
make
```
Now you should have the shared libraries located in `build` directory. 

# Setting up PCog
First clone PCog

```
git clone https://github.com/Ivan1931/pcog.git
```

Copy the build libraries from AI-toolbox to the deps directory in PCog.

```
cd pcog
cp /path/to/ai-toolbox/build/*.so pcog/deps
```

If all steps have been setup you can run PCog. 

Install the necessary python libraries:

```
pip install -r .
```

Finally, run PCog

```
python -m pcog.__main__
```

# Notes
To run the agent you first have to start the test bed, then start PCog, then start the java agent with the following command:

```
gradle run -Ppcog
```
