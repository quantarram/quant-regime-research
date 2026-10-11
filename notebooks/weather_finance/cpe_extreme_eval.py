import os
os.environ['EVAL_MODE']='extreme_formation'
import cpe_extreme_run   # patches the panel loader to the extreme-event set
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cpe_climate_eval.py')).read())
