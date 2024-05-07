#%%
import pandas as pd

# you should use dotenv if you manage tokens, passwords, api keys, private infos etc
# look at the .env.example file and delete the .example extension
from dotenv import load_dotenv
import os
load_dotenv()
print(os.getenv("SUPER_SECRET_KEY"))


# %% 
# you can use a utils file to store functions and main (this) to call them
from utils import change_my_name # import the function from utils.py
change_my_name(10) # call the function
# %%
