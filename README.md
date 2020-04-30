# Solving the EMI equations using finite element methods

This repository contains source codes and an environment used to produce
results of the FEM chapter in the EMI book **EMI: CELL BASED MATHEMATICAL MODEL OF EXCITABLE CELLS**.

## Installation
### Obtaining the image
The environment is provided as a docker container built on top of FEniCS
2017.2.0 (which uses Python 2). The container can be built locally by

```
docker build --no-cache -t emi_book_fem .
```
from the root directory of this repository. You would then run the container
as

```
docker run -it -v $(pwd):/home/fenics/shared emi_book_fem
```

Alternatively, pull and run the image built in the cloud

```
docker run -it -v $(pwd):/home/fenics/shared mirok/emi-book-fem
```

### Testing
Having lauched the container navigate to the source folder and launch
the test shell script

```
# You are inside docker shell
# fenics@268200ea18c2:~$ pwd
# /home/fenics

cd emi-book-fem
cd emi
sh test.sh
```

If `You are all set message appears` everything is good to go.

## Usage
Results of the chapter can be obtained by running convergence studies of
the different formulations. Below the single-dimensional primal formulation
is used

```
fenics@268200ea18c2:~/emi-book-fem/emi$ python check_sanity.py fem/prime_single.py -ncases 4 -param_kappa 1 -param_eps 1E-2
```

with 4 refinements, conductivity value 1, and time step parameter 0.01. A succesfull
run results in a result file `sanity_fem.prime_single_standard_kappa1_eps0.01.txt`
located in `/home/fenics/emi-book-fem/emi/results`. The file contains details of
the refinement study in particular the columns of mesh sizes (h) and different errors
as specified in the `setup_error_monitor` function of `prime_single.py` [module](https://github.com/MiroK/emi-book-fem/blob/master/emi/fem/prime_single.py#L85).
In the example below a header file of the result script can be seen.

```
# ('plot', 0), ('ncases', 1), ('save_dir', './results'), ('case0', 0), ('problem', 'fem/prime_single.py'), ('norm', 'standard')
# kappa1_eps1
# ndofs h e[|u|_1] r[|u|_1] e[|u|_0] r[|u|_0] e[|v|_0] r[|v|_0] subspaces
```

Therefore, extracting the first and third columns from the file a convergence of the 
global potential in the (broken) $H^1$ norm can be plotted.

## Troubleshooting
Please use the GitHub issue tracker for reporting issues, discussing code
contributions or requesting assistance.


## Citing
This code is based on several other packages in addition to [FEniCS](https://fenicsproject.org/citing/). 

1. [**FEniCS_ii**](https://github.com/MiroK/fenics_ii) is used to perform assembly of the multiscale varitional forms
2. [**cbc.block**](https://bitbucket.org/fenics-apps/cbc.block/src/master/) is used to represent the discrete operators

These can be cited as

1. _Kuchta, Miroslav. "Assembly of multiscale linear PDE operators." arXiv preprint arXiv:1912.09319 (2019)._
2. _Mardal, Kent-Andre, and Joachim Berdal Haga. "Block preconditioning of systems of PDEs." Automated solution of differential equations by the finite element method. Springer, Berlin, Heidelberg, 2012. 643-655._
