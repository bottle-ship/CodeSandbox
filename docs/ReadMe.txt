myst-parser
sphinx-copybutton
sphinx-rtd-theme
sphinx-nested-apidoc
numpydoc

sphinx-apidoc -f -o .\source\api ..\learning_kit\ ..\learning_kit\test* ..\learning_kit\*\test*
make html
https://github.com/mabuchilab/QNET/blob/develop/docs/conf.py