# Multiscale Urban Growth Modeling

*Project description should go here once repository is public.*

## Running Simulations

In this section we document configuration steps necessary to use `simulation_script.py` to generate samples from our working model. 

### Necessary Resources

1. Computer with ~16GB of memory or more. Must be *nix (Linux or OS X); no tests have been run on Windows. Best if a server. 
2. A **clean** installation of Python, preferably as a virtual environment. Instructions for installing and configuring a virtual environment follow. 
3. This repository, accessed using git clone (instructions below). 

### Configuration Steps

#### 1. Clone this repository.

On your computer or server, navigate in the terminal to a folder of your choice. Clone this repository by typing `git clone https://github.com/adrianalbert/city-simulator.git`. Sample workflow in your terminal: 
```{bash}
$ cd ~/mydir
$ git clone https://github.com/adrianalbert/city-simulator.git 
```
There is now a complete copy of this software in the `mydir/city-simulator` folder. It may be that you need to later download updated versions of the software. To update the software, run the lines below: 
```{bash}
$ cd ~/mydir/city-simulator
$ git pull origin master
```
#### 2. Create a Virtual Python Environment

This is probably the most finicky part of these instructions. Your computer likely already has an installation of Python, potentially with some packages. In order to ensure reproducibility and reliability, we need to construct a "sanitized" Python environment that we can fully control -- particularly with respect to the packages installed and their versions. The lines below will construct such a clean installation (called `venv`, short for Virtual ENVironment) and download all the relevant packages from scratch. Note that you need to have cloned the repository from the previous step in order to correctly pull down the packages. 

```{bash}
$ cd ~/mydir/city-simulator       
$ virtualenv --no-site-packages venv     # create the virtual environment
$ source venv/bin/activate               # activate the virtual environment
(venv) $ pip install -r requirements.txt # install all required packages 
```

#### 3. Use the Simulator

You are now ready to run simulations. For detailed instructions, see the introduction to the file `simulation_script.py`. In broad outline, you need to:

##### Prepare

1. Generate `M0.txt` and `geo.txt` as matrices saved in plaintext to represent your desired initial condition and geographic mask. 
2. Generate the list of parameter values you want to simulate. 
3. Open `simulation_script.py` and adjust any internal parameters you like, such as the number of iterations and whether or not to print visualizations. 

##### Run Script

```
$ source venv/bin/activate # activate the virtual environment
(venv) $ python simulation_script.py
```

##### Inspect Output

The output of the script will be saved as a set of `.txt` files in the `simulator_files/output` folder. If you asked the script to make visualizations as well, they will be saved in the `simulator_files/viz` folder. 

## Don't Hesitate to Ask! 

This simulator is a first-draft in terms of your user experience. Some annoying aspects (e.g. the virtual environment) are unavoidable, but other aspects of your user experience (e.g. additional options etc) can potentially be improved. Let me know how! 
