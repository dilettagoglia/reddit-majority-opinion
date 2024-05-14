#%%
import pandas as pd
import warnings
warnings.filterwarnings("ignore")
from utils import Entropy 

e = Entropy()
#e.entropy_in_time_plot()
e.user_nodelist(entropy_param=False)
# %%
