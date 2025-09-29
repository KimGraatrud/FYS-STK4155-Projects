# FYS-STK4155 Project 1

This is the repo contains all the code used to generate the results for our report.


## Structure

Our repository has the following structure:

<!-- GPT generated section -->
```bash
├── src/            # Source files
    └── parts/      # Data generation
├── doc/            # Documents
    └── figures/    # Figures for the paper
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
git clone https://github.com/username/repo-name.git
cd repo-name
```


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

