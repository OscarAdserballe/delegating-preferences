import sys, gzip, json, glob, re
from pathlib import Path
import numpy as np, pandas as pd
ROOT=Path("/home/user/masters-thesis/bayesian-teaching-replication-and-economic-extension")
sys.path.insert(0,str(ROOT/"src"))
from flight_prefs import paper_data, prompts
from flight_prefs.agents.bayesian_assistant import BayesianAssistant
from flight_prefs.agents.base import DialogueState, PastRound
D=paper_data.load(ROOT/"data/paper")
H=np.stack([m.phi for m in D.heldout])  # 100,3,4
T=D.thetas
df=pd.read_parquet(ROOT/"results/main/decisions.parquet")
users=sorted(df.user_id.unique())
held=df[df.kind=="heldout"].copy()
LLM=["qwen25-7b","gemma4-31b","gemma4-31b-think"]
FEAT=["departure","duration","stops","price"]
pd.set_option("display.width",220); pd.set_option("display.max_columns",30)
