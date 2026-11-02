import sys, numpy as np
from kessler_mpc.engine_compat import import_engine
import_engine()
from kesslergame import Scenario, TrainerEnvironment
from kessler_mpc.controller import KesslerMPCController
from kessler_mpc.world import read_world

class Diag(KesslerMPCController):
    hist=[]; deaths=[]; prev_lives=None
    def actions(self, ss, gs):
        out=super().actions(ss,gs)
        w=read_world(ss,gs); p=self.last_plan
        if self.prev_lives is not None and w.lives<self.prev_lives:
            self.deaths.append((w.time, list(self.hist[-25:])))
        self.prev_lives=w.lives
        d=np.linalg.norm(w.ast_pos-w.pos,axis=1)-w.ast_rad-w.params.radius
        self.hist.append((round(w.time,2), w.n_ast, round(float(d.min()),1) if len(d) else None,
            None if p is None else (p.best_collides,p.n_safe,p.thrust_idx,p.turn_mode), out[2], w.respawning, round(w.speed)))
        return out
seed=int(sys.argv[1]); 
c=Diag(debug=True)
sc=Scenario(name="s",num_asteroids=6,seed=seed,time_limit=45,map_size=(1000,800))
TrainerEnvironment().run(scenario=sc,controllers=[c])
for t,h in c.deaths:
    print("DEATH at",round(t,2)); 
    for x in h[-14:]: print("  ",x)
