import os
os.environ['EVAL_MODE']='x4_formation'
import cpe_x4_run   # patches the panel loader
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cpe_climate_eval.py')).read())
