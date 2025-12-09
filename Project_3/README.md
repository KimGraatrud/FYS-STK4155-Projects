# FYS-STK4155 Project 3

Group members:
Kim Graatrud, Simon Silverstein, and Nicholas Andrés Tran Rodriguez.

This is the repo contains all the code used to generate the results for our report. **It does not provide the dataset used**, this is due to the large filesize. The dataset can be found [here](https://www.kaggle.com/datasets/samithsachidanandan/human-face-emotions).

`TBD title`

Where we used Scikit-learn's and Pytorch's implementations of trees, gradient boosting, and CNNs respectively to study how they preform on image recognition of human face emotions. 


## Structure

Our repository has the following structure:

<!-- GPT generated section -->
```bash
├── src/            # Source files
├── parts/          # Data generation
├── doc/            # Documents
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
cd FYS-STK4155-Projects/Project_3/
```

Note: Before the code is run, ensure you have a `./Data/` and a `./figures/` directory as that is where the output of `parts/` will be placed. Unfortunatly the data also contains some invalid entries, meaning that the following commands has to be run:
- `rm ./Data/*/*jpg*`
- `rm ./Data/*/*[A-z]*.*`
- `find Data -name '*[[:alpha:]]*.*'`

for a UNIX shell. Find the equivalent for the shell you are using. These commands takes care of the majority, if not all, of the invalid entries.

## Dependencies

Almost everything is included in `requirements.txt`. `texlive` and `pdflatex` may also be required for the fonts used in the plots. If this is an issue, comment out the offending line in `style.mplstyle`.

To install with pip:
```bash
pip install -r requirements.txt
```
or equivalently in `Anaconda`.

## Generation of results

To generate the results used in the paper, just import the files from `parts/` then run them. Or use the function provided in `main.py`, the functions have to be called from `this` directory, or else the relative paths will fail.