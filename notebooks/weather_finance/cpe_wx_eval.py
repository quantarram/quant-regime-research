import os
os.environ['EVAL_MODE']='wx_formation'
import cpe_wx_run
exec(open(os.path.join(os.path.dirname(os.path.abspath(__file__)),'cpe_climate_eval.py')).read())
