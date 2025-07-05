# Copyright 2024 Google LLC

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     https://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Eval Script for Calendar Scheduling."""

import json
import re

from absl import app
from absl import flags


_DATA_PATH = flags.DEFINE_string(
    'data_path',
    'data/calendar_scheduling.json',
    'path to the data file containing model responses in json format.',
)


def hour_to_num(hr_str):
  return float(hr_str.split(':')[0]) + (
      0.5 if hr_str.split(':')[1] == '30' else 0.0
  )


def _parse_response(response: str):
  """Parse the response.

  Returns a parsed suggested meeting time in (day, start_hour, end_hour).

  Args:
    response: Raw response from the model.

  Returns:
    A tuple of (day, start_hour, end_hour).
  """
  time_strs = re.findall(r'[A-Za-z]+, [0-9]+:[0-9]+ - [0-9]+:[0-9]+', response)
  if not time_strs:
    return '', -1, -1
  # If multiple matches are found, return the first one.
  time_str = time_strs[0]
  day, hour_str = (
      time_str.split(',')[0].strip(),
      time_str.split(',')[1].strip(),
  )
  start_hour, end_hour = (
      hour_str.split('-')[0].strip(),
      hour_str.split('-')[1].strip(),
  )
  return day, hour_to_num(start_hour), hour_to_num(end_hour)


def compute_solve_rate(responses: list[str], solutions: list[str]):
  """Computes solve rate by comparing model responses to golden solutions.

  Args:
    responses: A list of model responses.
    solutions: The corresponding list of golden solutions for the same tasks.

  Returns:
    A scalr solve rate.
  """
  solved_count = 0

  for r, s in zip(responses, solutions):
    r_day, r_start_hour, r_end_hour = _parse_response(r)
    s_day, s_start_hour, s_end_hour = _parse_response(s)
    if (
        r_day == s_day
        and r_start_hour == s_start_hour
        and r_end_hour == s_end_hour
    ):
      solved_count += 1
  return float(solved_count) / len(responses)


def main(_):
  with open(_DATA_PATH.value) as f:
    data = json.load(f)

  num_people, num_days, responses, solutions = [], [], [], []
  for _, item in data.items():
    num_people.append(item['num_people'])
    num_days.append(item['num_days'])
    responses.append(item['pred_5shot_pro'])
    solutions.append(item['golden_plan'])

  overall_solved_rate = compute_solve_rate(responses, solutions)
  print(
      f'Overall solve rate of {len(responses)} samples: {overall_solved_rate}'
  )
  for num_p, num_d in sorted(
      list(set(zip(num_people, num_days))), key=lambda tup: (tup[0], tup[1])
  ):
    included_responses = [
        r
        for r, p, d in zip(responses, num_people, num_days)
        if p == num_p and d == num_d
    ]
    included_solutions = [
        s
        for s, p, d in zip(solutions, num_people, num_days)
        if p == num_p and d == num_d
    ]
    solved_rate = compute_solve_rate(included_responses, included_solutions)
    print(
        f'Solve rate of {num_p} people and {num_d} days of'
        f' {len(included_responses)} samples: {solved_rate}'
    )


if __name__ == '__main__':
  app.run(main)
