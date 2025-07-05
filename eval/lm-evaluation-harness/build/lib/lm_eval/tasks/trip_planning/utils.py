# lm_eval/tasks/trip_planning/utils.py

import re
import datasets
import json
def process_docs(dataset: datasets.Dataset) -> datasets.Dataset:
    """
    对原始数据进行预处理，确保每个样本包含任务所需的字段。
    假定输入样本包含如下字段：
      - prompt_0shot, prompt_5shot
      - cities: 例如 "Helsinki**Barcelona**Florence"
      - durations: 例如 "5**5**6"
      - golden_plan: ground-truth 解决方案
      - pred_5shot_pro: 模型生成的 5-shot 回答
    这里我们直接保留相关字段，如果需要进一步处理可在此进行。
    """
    def _process_doc(doc):
        # 此处可以进一步转换，比如对 cities/durations 字符串做修剪、格式检查等
        out =  {
            'id': doc['id'],
            "prompt_0shot": doc.get("prompt_0shot", ""),
            "prompt_5shot": doc.get("prompt_5shot", ""),
            "cities": doc["cities"],
            "durations": doc["durations"],
            "golden_plan": doc.get("golden_plan", ""),
            "pred_5shot_pro": doc.get("pred_5shot_pro", ""),
        }
        return out
    return dataset.map(_process_doc)


def parse_response(response: str):
  """Parse the response.

  Returns a parsed plan in a list of (city, stay_days) tuples.

  Args:
    response: Raw response from the model.

  Returns:
    Structured plan after parsing.
  """
  pattern_visit = r'\d+-\d+'
  pattern_flight = r'.*Day (\d+).*from (\w+) to (\w+)'
  pattern_days = r'European cities for (\d+) days'

  days, flights, flight_days = [], [], []
  total_days = None
  for piece in response.split('\n'):
    days_match = re.findall(pattern_days, piece)
    if days_match:
      total_days = int(days_match[0])

    visit_match = re.findall(pattern_visit, piece)
    if visit_match:
      days.append(visit_match[0])
      end_day = int(visit_match[0].split('-')[1])
      # Reach the end of the plan, stop to avoid parsing alternative plans.
      if end_day == total_days:
        break
    flight_match = re.findall(pattern_flight, piece)
    if flight_match:
      flights.append(flight_match[0])

  visit_cities, parsed_plan = [], []
  for flight_day, begin_city, end_city in flights:
    flight_days.append(int(flight_day))
    if not visit_cities:
      visit_cities.append(begin_city)
      visit_cities.append(end_city)
    else:
      visit_cities.append(end_city)

  if not days or not flights or not visit_cities:
    return []
  last_day = int(days[-1].split('-')[1])
  flight_days = [1] + flight_days + [last_day]
  for i, visit_city in enumerate(visit_cities):
    city_stay = flight_days[i + 1] - flight_days[i] + 1
    parsed_plan.append((visit_city, city_stay))

  return parsed_plan


def compute_example_score(cities: str, durations: str, parsed_plan):
    """
    根据解析后的计划和 ground-truth 中提供的城市与停留天数计算 exact-match (0/1) 分数。
    参数：
      cities: 城市字符串，格式为 "city1**city2**city3"
      durations: 停留天数字符串，格式为 "1**2**3"
      parsed_plan: 由 parse_response 返回的列表，每个元素为 (city, stay_days)
    返回：
      1.0 代表完全匹配，0.0 代表不匹配
    """
    stays = [x for x in cities.split('**') if x]
    days = [int(x) for x in durations.split('**') if x]
    num_stays = min(len(stays), len(parsed_plan))
    num_match = 0
    for i in range(num_stays):
        if stays[i] == parsed_plan[i][0] and days[i] == parsed_plan[i][1]:
            num_match += 1
        else:
            break
    hard_score = 0.0 if num_match / len(stays) < 1.0 else 1.0
    return hard_score


def trip_planning_metric(references, predictions, **kwargs):
    """
    自定义的评估函数，计算样本级别的 exact-match 准确率。
    参数：
      references: list，每个元素是一个字典或者 JSON 字符串，包含 "cities" 和 "durations"
      predictions: list，每个元素是模型生成的响应文本
      **kwargs: 其它额外参数
    返回：
      一个浮点数表示平均准确率
    """
    scores = []
    
    for gold, pred in zip(references, predictions):
        # 如果 gold 是字符串，则尝试解析为字典
        if isinstance(gold, str):
            try:
                gold = json.loads(gold)
            except Exception as e:
                # 如果转换失败，这里打印错误并继续（或者抛出异常）
                print("Error parsing gold as JSON:", gold, e)
                continue
        # 使用 parse_response 解析模型生成的文本
        parsed = parse_response(pred)
        # 使用 gold 的 "cities" 和 "durations" 计算评分
        print("results: ",gold["cities"], gold["durations"], "\n Output:",pred, "\n Parsed_Output:" ,parsed)
        score = compute_example_score(gold["cities"], gold["durations"], parsed)
        scores.append(score)
    return sum(scores) / len(scores) if scores else 0.0

