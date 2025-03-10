import ast
import sys
import csv
import json
import math
import logging
import numpy as np
from tqdm import tqdm
from typing import List, Dict, Tuple
from dataclasses import dataclass
from scipy.stats import pearsonr
from collections import Counter


field_size_limit = sys.maxsize
while True:
    try:
        csv.field_size_limit(field_size_limit)
        break
    except OverflowError:
        field_size_limit = int(field_size_limit / 10)


class EvaluationLogger:
    """Handles logging operations to file and console"""
    def __init__(self, log_file='result.txt'):
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)

        file_handler = logging.FileHandler(log_file, mode='w')  
        file_formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(file_formatter)

        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter('%(message)s'))

        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def log(self, message: str):
        self.logger.info(message)


logger = EvaluationLogger()


@dataclass
class EvaluationConfig:
    """Configuration parameters for evaluation"""
    task_type: str = 'all'
    link_usefulness_file: str = ''
    ground_truth_file: str = ''
    evaluation_files: Dict[str, str] = None  # {method: file_path}


@dataclass
class EvaluationResult:
    """Container for evaluation metrics"""
    query_level_corr: float
    session_level_corr: float
    combined_corr: float


class DataLoader:
    """Handles data loading and preprocessing"""

    @staticmethod
    def load_evaluation_data(file_path: str, method: str, task_type: str) -> Tuple[List, List, List, List]:
        """Load evaluation results for different methods"""
        sat_turn = []
        sat = []
        sat_turn_dim_avg = []
        sat_turn_predict = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                task = row['task_type']
                if task_type != 'all' and task != task_type:
                    continue
                if method == 'GEVAL':
                    scores = ast.literal_eval(row['evaluation'])
                    sat_turn.append(scores[:-1])
                    sat.append(scores[-1])
                    sat_turn_dim_avg.append([])  # GEVAL 没有 dim 数据
                    sat_turn_predict.append(scores[:-1])  # GEVAL 的 query-level 预测就是 scores[:-1]
                elif method in ['URS', 'GAH']:
                    eval_data = ast.literal_eval(row['evaluation'])
                    eval_turn = []
                    eval_turn_dim_avg = []
                    eval_turn_predict = []
                    for k, v in eval_data.items():
                        values_list = list(v.values())
                        if k != '整个对话':
                            eval_turn.append(values_list)
                            eval_turn_predict.append(values_list[-1])
                            eval_turn_dim_avg.append(sum(values_list[:-1]) / len(values_list[:-1]))
                    sat_turn.append(eval_turn)
                    sat_turn_dim_avg.append(eval_turn_dim_avg)
                    sat_turn_predict.append(eval_turn_predict)
                    sat.append(values_list)
        return sat_turn_dim_avg, sat_turn_predict, sat_turn, sat

    @staticmethod
    def load_ground_truth(file_path: str) -> Tuple[List, List]:
        """Load ground truth satisfaction data"""
        rows = []
        with open(file_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
        sat_turns = [ast.literal_eval(row['sat_turn']) for row in rows]
        satisfactions = [int(row['satisfaction']) for row in rows]
        return sat_turns, satisfactions

    @staticmethod
    def load_link_usefulness_data(file_path: str, task_type: str) -> Tuple[List, List, List, List]:
        """Load link usefulness data"""
        sat_turn_link = []
        usefulness_sat_list = []
        all_link_usefulness = []
        all_satisfaction = []
        with open(file_path, 'r', encoding='utf-8') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                task = row['task_type']
                if task_type != 'all' and task != task_type:
                    continue
                sat_turn_list = ast.literal_eval(row['sat_turn'])
                useful_turn_list = json.loads(row['useful_split'])
                all_satisfaction.append(int(row['satisfaction']))
                sat_turn_blank = []
                usefulness_sat = []
                link_usefulness = []
                for index, sat in enumerate(sat_turn_list):
                    sat_turn_blank.append(sat)
                    sat_turn_link.append(sat)
                    use_dic = useful_turn_list[index]
                    use_list = []
                    if use_dic[0] != []:
                        for statement_link in use_dic:
                            if len(statement_link) < 1:
                                continue
                            statement_link_deal = statement_link
                            use_list.append(max(statement_link_deal))
                    else:
                        llm_usefulness = ast.literal_eval(row['llm-statement-useful'])
                        u = int(llm_usefulness[index])
                        use_list.append(u)
                    link_usefulness.append(use_list)
                    usefulness_sat_link = [
                        WeightingMethods.decreasing_weight(use_list),
                        WeightingMethods.increasing_weight(use_list),
                        WeightingMethods.euqal_weight(use_list),
                        WeightingMethods.middle_high_weight(use_list),
                        WeightingMethods.middle_low_weight(use_list)
                    ]
                    usefulness_sat.append(usefulness_sat_link)
                sat_turn_link.append(sat_turn_blank)
                usefulness_sat_list.append(usefulness_sat)
                all_link_usefulness.append(link_usefulness)
        return sat_turn_link, usefulness_sat_list, all_link_usefulness, all_satisfaction


class CorrelationAnalyzer:
    """Calculates various correlation metrics"""

    @staticmethod
    def calculate_pearson(pred: List[float], true: List[float]) -> Tuple[float, float]:
        """Calculate Pearson correlation coefficient"""
        return pearsonr(pred, true)

    @staticmethod
    def z_score_normalization(data: List[float]) -> List[float]:
        """Normalize data using Z-score"""
        mean = np.mean(data)
        std = np.std(data)
        return [(x - mean) / std for x in data]


class WeightingMethods:
    """Contains different weighting strategies"""

    @staticmethod
    def decreasing_weight(scores: List[float]) -> float:
        """Decreasing weights: 1/r"""
        weights = [1 / (i + 1) for i in range(len(scores))]
        sum_w = sum(weights)
        weights_deal = [w / sum_w for w in weights]
        return sum(s * w for s, w in zip(scores, weights_deal))

    @staticmethod
    def increasing_weight(scores: List[float]) -> float:
        """Increasing weights: r"""
        weights = [i + 1 for i in range(len(scores))]
        sum_w = sum(weights)
        weights_deal = [w / sum_w for w in weights]
        return sum(s * w for s, w in zip(scores, weights_deal))

    @staticmethod
    def euqal_weight(scores: List[float]) -> float:
        """euqal weighting"""
        return np.mean(scores)

    @staticmethod
    def middle_high_weight(scores: List[float]) -> float:
        """中间权重高: 前半段 w_r = r, 后半段 w_r = N + 1 - r"""
        N = len(scores)
        weights = [r + 1 if r < N / 2 else N - r for r in range(N)]
        sum_w = sum(weights)
        weights_deal = [w / sum_w for w in weights]
        return sum(s * w for s, w in zip(scores, weights_deal))

    @staticmethod
    def middle_low_weight(scores: List[float]) -> float:
        """中间权重低: 前半段 w_r = 1/r, 后半段 w_r = 1/(N + 1 - r)"""
        N = len(scores)
        weights = [1 / (r + 1) if r < N / 2 else 1 / (N - r) for r in range(N)]
        sum_w = sum(weights)
        weights_deal = [w / sum_w for w in weights]
        return sum(s * w for s, w in zip(scores, weights_deal))

    @staticmethod
    def find_mode_or_average(data: List[float]) -> float:
        counter = Counter(data)
        most_common = counter.most_common()
        max_count = most_common[0][1]
        modes = [item[0] for item in most_common if item[1] == max_count]

        if len(modes) > 1:
            return sum(modes) / len(modes)

        return modes[0]


def weighted_sum(scores, weights):
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]
    return sum(s * w for s, w in zip(scores, normalized_weights))


def sdcg_session(data, br=1.35, bq=1.05):
    result = []
    for d_l in data:
        r = []
        score = 0
        for d in d_l:
            if len(d) == 0:
                continue
            discounts = np.log2(np.arange(2, len(d) + 2)) / np.log2(br)
            r.append(weighted_sum(list(2**np.array(d, dtype=np.float32) - 1),list(discounts)))
        if len(r) == 0:
            result.append(-1)
            continue
        discount = []
        for k, rel in enumerate(r, start=1):  # k 从 1 开始
            discount.append(1/ (math.log(k+bq-1)/math.log(bq)))
            score += rel / (math.log(k+bq-1)/math.log(bq))
        result.append(score)
    return result


def srbp_session(data, b=0.0, p=0.05):
    result = []
    for d_l in data:
        r = []
        rbp_score = 0
        for d_ll in d_l:
            s = 0
            if len(d_ll) == 0:
                continue
            discount1 = []
            for n, d in enumerate(d_ll):
                discount1.append((b*p)**(n))
                s += d*(b*p)**(n)
            r.append(s)
        if len(r) == 0:
            result.append(-1)
            continue
        discount2 = []
        for k, rel in enumerate(r, start=1):
            discount2.append((((p-b*p)/(1-b*p)) ** (k-1)))
            rbp_score += rel * (((p-b*p)/(1-b*p)) ** (k-1))
        result.append(rbp_score)
    return result


def rsdcg_session(data, br=1.35, bq=1.05, lamda=3):
    result = []
    for d_l in data:
        r = []
        score = 0
        for d in d_l:
            if len(d) == 0:
                continue
            discounts = np.log2(np.arange(2, len(d) + 2)) / np.log2(br)
            r.append(weighted_sum(list(2**np.array(d, dtype=np.float32) - 1),list(discounts)))
        if len(r) == 0:
            result.append(-1)
            continue
        for k, rel in enumerate(r, start=1):
            score += rel * math.exp(-lamda*(len(r)-k)) / (math.log(k+bq-1)/math.log(bq))
        result.append(score)
    return result


def rsrbp_session(data, b=0.0, p=0.05, lamda=0.05):
    result = []
    for d_l in data:
        r = []
        rbp_score = 0
        for d_ll in d_l:
            s = 0
            if len(d_ll) == 0:
                continue
            discount1 = []
            for n, d in enumerate(d_ll):
                discount1.append((b*p)**(n))
                s += d*(b*p)**(n)
            r.append(s)
        if len(r) == 0:
            result.append(-1)
            continue
        for k, rel in enumerate(r, start=1):
            rbp_score += rel * (((p-b*p)/(1-b*p)) ** (k-1)) * math.exp(-lamda*(len(r)-k))
        result.append(rbp_score)
    return result


class BaseEvaluator:
    """Base class for evaluation methods"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.gt_turns, self.gt_sessions = DataLoader.load_ground_truth(config.ground_truth_file)


class QueryLevelEvaluator(BaseEvaluator):
    """Handles query-level evaluation for different methods"""

    def evaluate_method(self, method: str) -> Dict[str, EvaluationResult]:
        """Evaluate a single method"""
        # Load method-specific data
        sat_turn_dim_avg, sat_turn_predict, sat_turn, sat = DataLoader.load_evaluation_data(
            self.config.evaluation_files[method], method, self.config.task_type
        )

        flattened_gt_turns = [item for sublist in self.gt_turns for item in sublist]

        results = {}

        if method == 'GEVAL':
            flattened_turns = [item for sublist in sat_turn for item in sublist]
            turn_corr, _ = CorrelationAnalyzer.calculate_pearson(flattened_turns, flattened_gt_turns)
            session_corr, _ = CorrelationAnalyzer.calculate_pearson(sat, self.gt_sessions)
            results[method] = EvaluationResult(turn_corr, session_corr, 0.0)
        elif method in ['URS', 'GAH']:
            # 计算 dim 相关性
            flattened_dim_avg = [item for sublist in sat_turn_dim_avg for item in sublist]
            dim_turn_corr, _ = CorrelationAnalyzer.calculate_pearson(flattened_dim_avg, flattened_gt_turns)
            dim_session_corr, _ = CorrelationAnalyzer.calculate_pearson([sum(sublist) / len(sublist) if sublist else 0 for sublist in sat_turn_dim_avg], self.gt_sessions)
            results[f'{method}_dim'] = EvaluationResult(dim_turn_corr, dim_session_corr, 0.0)

            # 计算 pre 相关性
            flattened_predict = [item for sublist in sat_turn_predict for item in sublist]
            pre_turn_corr, _ = CorrelationAnalyzer.calculate_pearson(flattened_predict, flattened_gt_turns)
            pre_session_corr, _ = CorrelationAnalyzer.calculate_pearson([sum(sublist) / len(sublist) if sublist else 0 for sublist in sat_turn_predict], self.gt_sessions)
            results[f'{method}_pre'] = EvaluationResult(pre_turn_corr, pre_session_corr, 0.0)

        return results


class CombinedEvaluator(BaseEvaluator):
    """Handles combined evaluation with link usefulness"""

    def evaluate_combination(self, method: str, usefulness_sat_list, sat_turn_link, all_link_usefulness, all_satisfaction):
        """Evaluate method combined with link usefulness"""
        # Load evaluation data for the method
        sat_turn_dim_avg, sat_turn_predict, sat_turn, sat = DataLoader.load_evaluation_data(
            self.config.evaluation_files[method], method, self.config.task_type
        )

        flattened_gt_turns = [item for sublist in self.gt_turns for item in sublist]

        results = {}

        if method == 'GEVAL':
            flattened_turns = [item for sublist in sat_turn for item in sublist]
            turn_corr, _ = CorrelationAnalyzer.calculate_pearson(flattened_turns, flattened_gt_turns)
            session_corr, _ = CorrelationAnalyzer.calculate_pearson(sat, self.gt_sessions)

            # Combine prompt and link usefulness
            norm_turns = CorrelationAnalyzer.z_score_normalization(flattened_turns)
            combined_scores = []
            aggregation_methods = [
                "decrease", "increase", "euqal", "middle_high", "middle_low"
            ]
            for i, method_name in enumerate(aggregation_methods):
                flattened_link_usefulness = [score[i] for sublist in usefulness_sat_list for score in sublist]
                norm_link_usefulness = CorrelationAnalyzer.z_score_normalization(flattened_link_usefulness)
                combined_score = [(p + l) / 2 for p, l in zip(norm_turns, norm_link_usefulness)]
                combined_corr, _ = CorrelationAnalyzer.calculate_pearson(combined_score, flattened_gt_turns)
                logger.log(f'{method} combined with link usefulness ({method_name}): corr:{combined_corr}')
                combined_scores.append(combined_corr)

            results[method] = EvaluationResult(turn_corr, session_corr, max(combined_scores))
        elif method in ['URS', 'GAH']:
            # 处理 dim
            flattened_dim_avg = [item for sublist in sat_turn_dim_avg for item in sublist]
            dim_turn_corr, _ = CorrelationAnalyzer.calculate_pearson(flattened_dim_avg, flattened_gt_turns)
            dim_session_corr, _ = CorrelationAnalyzer.calculate_pearson([sum(sublist) / len(sublist) if sublist else 0 for sublist in sat_turn_dim_avg], self.gt_sessions)

            norm_dim_avg = CorrelationAnalyzer.z_score_normalization(flattened_dim_avg)
            dim_combined_scores = []
            aggregation_methods = [
                "decrease", "increase", "euqal", "middle_high", "middle_low"
            ]
            for i, method_name in enumerate(aggregation_methods):
                flattened_link_usefulness = [score[i] for sublist in usefulness_sat_list for score in sublist]
                norm_link_usefulness = CorrelationAnalyzer.z_score_normalization(flattened_link_usefulness)
                combined_score = [(p + l) / 2 for p, l in zip(norm_dim_avg, norm_link_usefulness)]
                combined_corr, _ = CorrelationAnalyzer.calculate_pearson(combined_score, flattened_gt_turns)
                logger.log(f'{method}_dim combined with link usefulness ({method_name}): corr:{combined_corr}')
                dim_combined_scores.append(combined_corr)

            results[f'{method}_dim'] = EvaluationResult(dim_turn_corr, dim_session_corr, max(dim_combined_scores))

            # 处理 pre
            flattened_predict = [item for sublist in sat_turn_predict for item in sublist]
            pre_turn_corr, _ = CorrelationAnalyzer.calculate_pearson(flattened_predict, flattened_gt_turns)
            pre_session_corr, _ = CorrelationAnalyzer.calculate_pearson([sum(sublist) / len(sublist) if sublist else 0 for sublist in sat_turn_predict], self.gt_sessions)

            norm_predict = CorrelationAnalyzer.z_score_normalization(flattened_predict)
            pre_combined_scores = []
            for i, method_name in enumerate(aggregation_methods):
                flattened_link_usefulness = [score[i] for sublist in usefulness_sat_list for score in sublist]
                norm_link_usefulness = CorrelationAnalyzer.z_score_normalization(flattened_link_usefulness)
                combined_score = [(p + l) / 2 for p, l in zip(norm_predict, norm_link_usefulness)]
                combined_corr, _ = CorrelationAnalyzer.calculate_pearson(combined_score, flattened_gt_turns)
                logger.log(f'{method}_pre combined with link usefulness ({method_name}): corr:{combined_corr}')
                pre_combined_scores.append(combined_corr)

            results[f'{method}_pre'] = EvaluationResult(pre_turn_corr, pre_session_corr, max(pre_combined_scores))

        return results


class LinkUsefulnessEvaluator:
    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.sat_turn_link, self.usefulness_sat_list, self.all_link_usefulness, self.all_satisfaction = DataLoader.load_link_usefulness_data(
            config.link_usefulness_file, config.task_type)

    def evaluate_query_level(self):
        logger.log("Query-level link usefulness correlations:")
        aggregation_methods = [
            "decrease", "increase", "euqal", "middle_high", "middle_low"
        ]
        flattened_gt_turns = [item for sublist in DataLoader.load_ground_truth(self.config.ground_truth_file)[0] for item in sublist]
        for i, method_name in enumerate(aggregation_methods):
            flattened_link_usefulness = [score[i] for sublist in self.usefulness_sat_list for score in sublist]
            correlation, p_value = CorrelationAnalyzer.calculate_pearson(flattened_gt_turns, flattened_link_usefulness)
            logger.log(f'{method_name}: corr:{correlation}, pvalue:{p_value}')

    def evaluate_session_level_end_to_end(self):
        logger.log('Link end to end:')
        sdcg_list = sdcg_session(self.all_link_usefulness)
        srbp_list = srbp_session(self.all_link_usefulness)
        rsdcg_list = rsdcg_session(self.all_link_usefulness)
        rsrbp_list = rsrbp_session(self.all_link_usefulness)

        valid_indices = [i for i, s in enumerate(sdcg_list) if s >= 0]
        sdcg_list_valid = [sdcg_list[i] for i in valid_indices]
        srbp_list_valid = [srbp_list[i] for i in valid_indices]
        rsdcg_list_valid = [rsdcg_list[i] for i in valid_indices]
        rsrbp_list_valid = [rsrbp_list[i] for i in valid_indices]
        sat_gt_valid = [self.all_satisfaction[i] for i in valid_indices]

        correlation, p_value = CorrelationAnalyzer.calculate_pearson(sdcg_list_valid, sat_gt_valid)
        logger.log(f'sdcg: corr:{correlation}, p_value:{p_value}')
        correlation, p_value = CorrelationAnalyzer.calculate_pearson(srbp_list_valid, sat_gt_valid)
        logger.log(f'srbp: corr:{correlation}, p_value:{p_value}')
        correlation, p_value = CorrelationAnalyzer.calculate_pearson(rsdcg_list_valid, sat_gt_valid)
        logger.log(f'rsdcg: corr:{correlation}, p_value:{p_value}')
        correlation, p_value = CorrelationAnalyzer.calculate_pearson(rsrbp_list_valid, sat_gt_valid)
        logger.log(f'rsrbp: corr:{correlation}, p_value:{p_value}')

        return sdcg_list_valid, srbp_list_valid, rsdcg_list_valid, rsrbp_list_valid


class EvaluationPipeline:
    """Orchestrates the complete evaluation process"""

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.query_evaluator = QueryLevelEvaluator(config)
        self.combined_evaluator = CombinedEvaluator(config)
        self.link_evaluator = LinkUsefulnessEvaluator(config)

    
    def run_full_evaluation(self):
        # Evaluate link usefulness
        self.link_evaluator.evaluate_query_level()
        sdcg_list, srbp_list, rsdcg_list, rsrbp_list = self.link_evaluator.evaluate_session_level_end_to_end()

        # Extract link usefulness query-level results
        link_query_pred = []
        for session in self.link_evaluator.usefulness_sat_list:
            session_query_scores = []
            for query in session:
                session_query_scores.append(WeightingMethods.decreasing_weight(query))
            link_query_pred.append(session_query_scores)

        # Define aggregation methods
        aggregation_methods = [
            WeightingMethods.decreasing_weight,
            WeightingMethods.increasing_weight,
            WeightingMethods.euqal_weight,
            WeightingMethods.middle_high_weight,
            WeightingMethods.middle_low_weight,
            max,
            min,
            WeightingMethods.find_mode_or_average
        ]
        method_names = [
            "decrease", "increase", "euqal", "middle_high", "middle_low", "max", "min", "mode_or_average"
        ]

        # Aggregate link usefulness query-level results to session-level (only once)
        logger.log('Link usefulness query-level aggregation to session-level:')
        for i, method_func in enumerate(aggregation_methods):
            link_aggregated = [method_func(session) for session in link_query_pred]
            link_corr, _ = CorrelationAnalyzer.calculate_pearson(link_aggregated, self.link_evaluator.all_satisfaction)
            logger.log(f'{method_names[i]}: corr: {link_corr:.3f}')

        # Evaluate individual methods and their combinations with link usefulness
        for method in ['GEVAL', 'URS', 'GAH']:
            logger.log(f"\nEvaluating {method} method:")
            sat_turn_dim_avg, sat_turn_predict, sat_turn, sat = DataLoader.load_evaluation_data(
                self.config.evaluation_files[method], method, self.config.task_type
            )
            query_results = self.query_evaluator.evaluate_method(method)
            for result_name, result in query_results.items():
                logger.log(f"{result_name} Query-level correlation: {result.query_level_corr:.3f}")
                logger.log(f"{result_name} Session-level correlation: {result.session_level_corr:.3f}")

            combined_results = self.combined_evaluator.evaluate_combination(
                method,
                self.link_evaluator.usefulness_sat_list,
                self.link_evaluator.sat_turn_link,
                self.link_evaluator.all_link_usefulness,
                self.link_evaluator.all_satisfaction
            )
            for result_name, result in combined_results.items():
                logger.log(f"{result_name} Combined correlation: {result.combined_corr:.3f}")

            for result_name, result in query_results.items():
                if method == 'GEVAL':
                    prompt_session_pred = sat
                    prompt_query_pred = sat_turn
                elif method in ['URS', 'GAH']:
                    if 'dim' in result_name:
                        prompt_session_pred = [sum(sublist) / len(sublist) if sublist else 0 for sublist in sat_turn_dim_avg]
                        prompt_query_pred = sat_turn_dim_avg
                    else:
                        prompt_session_pred = [sum(sublist) / len(sublist) if sublist else 0 for sublist in sat_turn_predict]
                        prompt_query_pred = sat_turn_predict

                logger.log(f'{result_name} link + prompt end to end:')
                self._link_prompt_end_to_end(
                    prompt_session_pred,
                    sdcg_list,
                    self.link_evaluator.all_satisfaction
                )

                # Flatten link and prompt query-level results for Z - score normalization
                flattened_link_query_pred = [score for session in link_query_pred for score in session]
                flattened_prompt_query_pred = [score for session in prompt_query_pred for score in session]

                # Z - score normalization for all link and prompt query-level results
                normalized_flattened_link = CorrelationAnalyzer.z_score_normalization(flattened_link_query_pred)
                normalized_flattened_prompt = CorrelationAnalyzer.z_score_normalization(flattened_prompt_query_pred)

                # Restore the structure of normalized results
                normalized_link_query_pred = []
                index = 0
                for session in link_query_pred:
                    session_length = len(session)
                    normalized_link_query_pred.append(normalized_flattened_link[index:index + session_length])
                    index += session_length

                normalized_prompt_query_pred = []
                index = 0
                for session in prompt_query_pred:
                    session_length = len(session)
                    normalized_prompt_query_pred.append(normalized_flattened_prompt[index:index + session_length])
                    index += session_length

                # Combine link usefulness and prompt query-level results
                combined_query_pred = []
                for prompt_session, link_session in zip(normalized_prompt_query_pred, normalized_link_query_pred):
                    combined_session = [(p + l) / 2 for p, l in zip(prompt_session, link_session)]
                    combined_query_pred.append(combined_session)

                # Aggregate prompt query-level results to session-level
                logger.log(f'{result_name} Prompt query-level aggregation:')
                for i, method_func in enumerate(aggregation_methods):
                    prompt_aggregated = [method_func(session) for session in prompt_query_pred]
                    prompt_corr, _ = CorrelationAnalyzer.calculate_pearson(prompt_aggregated, self.link_evaluator.all_satisfaction)
                    logger.log(f'{method_names[i]}: corr: {prompt_corr:.3f}')

                # Aggregate combined query-level results to session-level
                logger.log(f'{result_name} Link usefulness + prompt query-level aggregation:')
                for i, method_func in enumerate(aggregation_methods):
                    combined_aggregated = [method_func(session) for session in combined_query_pred]
                    combined_corr, _ = CorrelationAnalyzer.calculate_pearson(combined_aggregated, self.link_evaluator.all_satisfaction)
                    logger.log(f'{method_names[i]}: corr: {combined_corr:.3f}')

        # Ground truth analysis
        logger.log("\nGround Truth query-level to session-level:")
        self._true_turn_sat_analysis()

    def _link_prompt_end_to_end(self, sat_prompt, sat_ir, sat_gt):
        sat_prompt_nor = CorrelationAnalyzer.z_score_normalization(sat_prompt)
        sat_ir_nor = CorrelationAnalyzer.z_score_normalization(sat_ir)

        sat_combine = []
        for p, l in zip(sat_prompt_nor, sat_ir_nor):
            sat_combine.append((p + l) / 2)

        correlation, p_value = CorrelationAnalyzer.calculate_pearson(sat_combine, sat_gt)
        logger.log(f'corr:{correlation}, p_value:{p_value}')

    def _true_turn_sat_analysis(self):
        """Analyze true turn satisfaction"""
        sat_turn, sat = DataLoader.load_ground_truth(self.config.ground_truth_file)

        sat_turn_single = [[], [], [], [], [], [], [], []]
        aggregation_methods = [
            "decrease", "increase", "euqal", "middle_high", "middle_low", "max", "min", "mode_or_average"
        ]
        for s in sat_turn:
            sat_turn_single[0].append(WeightingMethods.decreasing_weight(s))
            sat_turn_single[1].append(WeightingMethods.increasing_weight(s))
            sat_turn_single[2].append(WeightingMethods.euqal_weight(s))
            sat_turn_single[3].append(WeightingMethods.middle_high_weight(s))
            sat_turn_single[4].append(WeightingMethods.middle_low_weight(s))
            sat_turn_single[5].append(max(s))
            sat_turn_single[6].append(min(s))
            sat_turn_single[7].append(WeightingMethods.find_mode_or_average(s))

        for i, method_name in enumerate(aggregation_methods):
            correlation, p_value = CorrelationAnalyzer.calculate_pearson(sat_turn_single[i], sat)
            logger.log(f'{method_name}:{correlation}')

        logger.log("-----------------------------------------------")


if __name__ == "__main__":
    # Configuration
    config = EvaluationConfig(
        evaluation_files={
            'GEVAL': '',
            'URS': '',
            'GAH': ''
        },
        link_usefulness_file='',
        ground_truth_file=''
    )

    # Execute pipeline
    try:
        pipeline = EvaluationPipeline(config)
        pipeline.run_full_evaluation()
        logger.log("\nEvaluation completed successfully")
    except Exception as e:
        logger.log(f"\nEvaluation failed: {str(e)}")
