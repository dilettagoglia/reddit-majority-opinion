#%%
import pandas as pd
from utils import Entropy 
import warnings
warnings.filterwarnings("ignore")

e = Entropy()
#e.entropy_in_time_plot()
e.user_nodelist(entropy_param=False)
# %%
