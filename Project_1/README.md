# FYS-STK4155 Project 1

This is the repo contains all the code used to generate the results for our report. 

`A Comparison of Linear Regression Techniques as Applied to Runge’s
Function`

Where we use and compare different linear regression techniques to study Runge’s function.


## Structure

Our repository has the following structure:

<!-- GPT generated section -->
```bash
├── src/            # Source files
    └── parts/      # Data generation
├── doc/            # Documents
    └── Figures/    # Figures for the paper
├── main.py         # Main file
└── README.md       # Project readme
```
<!-- GPT generated section end -->

- `src/` houses the source code for the project (gradient decent, regression, etc.)
- `parts/` imports the code from src and generates the data and figures used in the paper in `doc/figures/`.
- `doc/` contains the paper in .pdf format and the figures used for it in `figures/`
- `main.py` is where we import and call the files from `parts/` and generate all the results.
- `readme.md` is this file.

## Installation

Clone the repository:
```bash
git clone https://github.com/KimGraatrud/FYS-STK4155-Projects.git
cd FYS-STK4155-Projects/Project_1/
```

Note: Before the code is run, ensure you have a `./data/` and a `./figures/` directory as that is where the output of `parts/` will be placed.

## Dependencies
- Numpy
- Matplotlib
- scikit-learn

To install with pip:
```bash
pip install numpy matplotlib scikit-learn
```
or equivalently in `Anaconda`.
---

## Generation of results


