## Explanation of each file

- `__init__.py` is just an empty file so python recognises that it can be imported.
- `parts/` contians the files which import the ones from this directory and uses them to compute the results for the exploratory paper.
- `utils.py` contain misc. helper functions and variables, like our random seed.
- `regression.py` contains the analytical solutions for the OLS and Ridge regression methods.
- `ml.py` contains the gradient descent (GD) class which houses all the gradient descent methods we used.
- `resampling.py` contains the class for the bootstrap and $k$-fold CV methods we have used.