# NATURAL PLAN TEST DATA

There are 3 tasks with test data: Trip Planning, Meeting Planning and Calendar Scheduling. Data is stored in json format, and each example contains the following fields:
### Trip Planning
- cities: cities to visit in the golden_plan
- durations: duration spent in each city in the golden_plan
- num_cities: total number of cities to visit in the golden_plan
- prompt_0shot: 0-shot prompt
- prompt_5shot: 5-shot prompt
- golden_plan: ground-truth solution
- pred_5shot_pro: 5-shot model response from Gemini 1.5 Pro

### Meeting Planning
- num_people: number of people to meet
- constraints: constraints to satisfy
- dist_matrix: traveling time between locations
- prompt_0shot: 0-shot prompt
- prompt_5shot: 5-shot prompt
- golden_plan: ground-truth solution
- pred_5shot_pro: 5-shot model response from Gemini 1.5 Pro

### Calendar Scheduling
- num_days: total number of days
- num_people: total number of people
- duration: number of meetings
- prompt_0shot: 0-shot prompt
- prompt_5shot: 5-shot prompt
- golden_plan: ground-truth solution
- pred_5shot_pro: 5-shot model response from Gemini 1.5 Pro

## Usage

Do model inference on the 5shot-prompt: `prompt_5shot`, and run evaluation on the model response.
