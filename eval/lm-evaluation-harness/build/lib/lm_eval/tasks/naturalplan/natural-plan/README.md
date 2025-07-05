# NATURAL PLAN

Natural Plan is a realistic planning benchmark in natural language containing 3 key tasks: Trip Planning, Meeting Planning, and Calendar Scheduling. This repo contains the data for inference, and evaluation code on model responses.

## Installation
Create a python virtual env

```
python -m venv natural_plan

source natural_plan/bin/activate
```

and install the dependencies

```
pip install absl-py
```

## Usage

Prepare the data after model inferences, and run evaluation code:

```
python evaluate_trip_planning.py
python evaluate_meeting_planning.py
python evaluate_calendar_scheduling.py
```

## Citing this work

```latex
@article{zheng2024naturalplanbenchmarkingllms,
      title={NATURAL PLAN: Benchmarking LLMs on Natural Language Planning},
      author={Huaixiu Steven Zheng and Swaroop Mishra and Hugh Zhang and Xinyun Chen and Minmin Chen and Azade Nova and Le Hou and Heng-Tze Cheng and Quoc V. Le and Ed H. Chi and Denny Zhou},
      year={2024},
      eprint={2406.04520},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2406.04520},
}
```

## License and disclaimer

Copyright 2024 DeepMind Technologies Limited

All software is licensed under the Apache License, Version 2.0 (Apache 2.0);
you may not use this file except in compliance with the Apache 2.0 license.
You may obtain a copy of the Apache 2.0 license at:
https://www.apache.org/licenses/LICENSE-2.0

All other materials are licensed under the Creative Commons Attribution 4.0
International License (CC-BY). You may obtain a copy of the CC-BY license at:
https://creativecommons.org/licenses/by/4.0/legalcode

Unless required by applicable law or agreed to in writing, all software and
materials distributed here under the Apache 2.0 or CC-BY licenses are
distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND,
either express or implied. See the licenses for the specific language governing
permissions and limitations under those licenses.

This is not an official Google product.
