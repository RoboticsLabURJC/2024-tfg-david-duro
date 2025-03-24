import random
from itertools import combinations
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.sstypes import (
    Mission,
    EndlessMission,
    Route,
    Scenario,
    Traffic,
    Trip,
    TrafficActor,
    TrapEntryTactic,
)
import time

normal = TrafficActor(name="car", depart_speed=0)

traffic = {}
traffic["0"] = Traffic(
    engine="SUMO",
    trips=[
        Trip(
            vehicle_name=f"car_{i+1}",
            route=Route(
                begin=("gneE3", 0, (i+2) * 12),
                end=("gneE3", 0, (i+2) * 13)
            ),
            depart=0,
            actor=normal
        )
        for i in range(2)
    ],
    flows=[]
)

# random.seed(time.time())
# begin_pose = random.randint(28, 30)
route = Route(begin=("gneE3", 1, 30), end=("gneE3", 1, 100))
ego_missions = [
    Mission(
        route=route,
        entry_tactic=TrapEntryTactic(
            start_time=0
        ),
    )
]

gen_scenario(
    scenario=Scenario(
        traffic=traffic,
        ego_missions=ego_missions,
    ),
    output_dir=Path(__file__).parent,
)

