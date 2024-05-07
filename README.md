# Template project Python Data Science using poetry

!!!
this is a project template, you can use this template clicking on the green button "use this template" on the top right of the page, this will create your repo with this structure 
!!!



install poetry [here](https://python-poetry.org/docs/#installing-with-the-official-installer)


## create the virtual environment and install the dependencies

```bash
poetry install
```

## activate the virtual environment

```bash
poetry shell
```

now you created an isolated enviroment, everything you install will not affect your system python installation, you can experiment freely and delete the .venv folder to start over. 
(if you cannot find the .env folder )


## install a new package

```bash
poetry add <package_name>
```
what you usually do with ```pip install pandas``` is now ```poetry add pandas```

## using .env file

if you have to handle sensitive data put it in the ```.env ```file, it will be ignored by git and you can access it with ```os.getenv('VARIABLE_NAME')```


## suggestion 
open the folder with vscode and you should see 
```#%%``` lines, these are the cell division, you can run a cell with ```shift+enter``` or pressing the play button on the left of the cell.



if something breaks reclone the repo and start over, it's a good practice to have a clean environment to work with.

```bash