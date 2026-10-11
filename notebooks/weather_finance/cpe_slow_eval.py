import os
os.environ['EVAL_MODE']='slow_formation'
import cpe_slow_run   # patches the panel loader to the batch-3 set
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cpe_climate_eval.py')).read())
