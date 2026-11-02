import time, sys
from kessler_mpc.engine_compat import import_engine
import_engine()
from kesslergame import Scenario, TrainerEnvironment
from kessler_mpc.controller import KesslerMPCController
seed=int(sys.argv[1]) if len(sys.argv)>1 else 1
T=float(sys.argv[2]) if len(sys.argv)>2 else 30
ctrl=KesslerMPCController(debug=True)
sc=Scenario(name="s",num_asteroids=6,seed=seed,time_limit=T,map_size=(1000,800))
t=time.time()
score,_=TrainerEnvironment().run(scenario=sc,controllers=[ctrl])
tm=score.teams[0]
import numpy as np
print("stop",score.stop_reason,"sim_t",round(score.sim_time,1),"hit",tm.asteroids_hit,"/",tm.total_asteroids,"shots",tm.shots_fired,"acc",round(tm.accuracy,2),"deaths",tm.deaths,"lives",tm.lives_remaining)
st=np.array(ctrl.step_times)*1000
print("ms/step mean %.1f p95 %.1f max %.1f  errors %d  wall %.1fs"%(st.mean(),np.percentile(st,95),st.max(),ctrl.error_count,time.time()-t))
