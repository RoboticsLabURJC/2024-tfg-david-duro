# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NON-INFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

import random
from itertools import combinations
from pathlib import Path

from smarts.sstudio import gen_scenario
from smarts.sstudio.sstypes import (
    Flow,
    Mission,
    Route,
    Scenario,
    Traffic,
    Trip,
    TrafficActor,
    TrapEntryTactic,
)

normal = TrafficActor(name="car", depart_speed=0)
# truck = TrafficActor(name="bus")


route_opt = [
    (0, 0),
    (1, 1),
    (2, 2),
]
x_cars = 20

# Traffic combinations = 3C2 + 3C3 = 3 + 1 = 4
# Repeated traffic combinations = 4 * 100 = 400
min_flows = 2
max_flows = 3
route_comb = [
    com
    for elems in range(min_flows, max_flows + 1)
    for com in combinations(route_opt, elems)
] * 100

traffic = {}
for name, routes in enumerate(route_comb):
    traffic[str(name)] = Traffic(
        flows=[
            Flow(
                route=Route(
                    begin=("gneE3", 1, (i+2) * 10),
                    end=("gneE3", 1, "max"),
                ),
                rate=60,
                begin=0,
                end=40,
                actors={normal: 1},
            )
            for i in range(2)
            # Flow(
            #     route=Route(
            #         begin=("gneE3", 2, 7),
            #         end=("gneE3", 2, "max"),
            #     ),
            #     rate=60,
            #     begin=0,
            #     end=50,
            #     actors={normal: 1},
            # ),
            # Flow(
            #     route=Route(
            #         begin=("gneE3", 2, 10),
            #         end=("gneE3", 2, "max"),
            #     ),
            #     rate=60,
            #     begin=0,
            #     end=50,
            #     actors={normal: 1},
            # ),
            # Flow(
            #     route=Route(
            #         begin=("gneE3", 2, 16),
            #         end=("gneE3", 2, "max"),
            #     ),
            #     rate=60,
            #     begin=0,
            #     end=50,
            #     actors={normal: 1},
            # )
        ],
        trips=[
            Trip(
                vehicle_name=f"car_{i+1}",
                route=Route(
                    begin=("gneE3", 0, (i+1) * 8),
                    end=("gneE3", 0, "max")
                ),
                depart=0,
                actor=normal
            )
            for i in range(x_cars)
        ]
        # trips=[
        #     Trip(
        #         vehicle_name=f"car_1",
        #         route=Route(
        #             begin=("gneE3", 0, 2),
        #             end=("gneE3", 0, "max")
        #         ),
        #         depart=0,
        #         actor=normal
        #     ),
        #     Trip(
        #         vehicle_name=f"car_2",
        #         route=Route(
        #             begin=("gneE3", 0, 5),
        #             end=("gneE3", 0, "max")
        #         ),
        #         depart=0,
        #         actor=normal
        #     ),
        #     Trip(
        #         vehicle_name=f"car_3",
        #         route=Route(
        #             begin=("gneE3", 0, 8),
        #             end=("gneE3", 0, "max")
        #         ),
        #         depart=0,
        #         actor=normal
        #     ),
        #     Trip(
        #         vehicle_name=f"car_4",
        #         route=Route(
        #             begin=("gneE3", 0, 11),
        #             end=("gneE3", 0, "max")
        #         ),
        #         depart=0,
        #         actor=normal
        #     ),
        #     Trip(
        #         vehicle_name=f"car_5",
        #         route=Route(
        #             begin=("gneE3", 0, 16),
        #             end=("gneE3", 0, "max")
        #         ),
        #         depart=0,
        #         actor=normal
        #     )
        # ]
    )

    

route = Route(begin=("gneE3", 2, 15), end=("gneE3", 1, 25))
# route = Route(begin=("gneE3", 1, 24), end=("gneE3", 1, 50))
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

