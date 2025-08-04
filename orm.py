import os
import re
from typing import TYPE_CHECKING, Dict, List, Union, Any

import json

if TYPE_CHECKING:
    from swift.llm import InferRequest


class ORM:

    def __call__(self, **kwargs) -> List[float]:
        raise NotImplementedError


class ReactORM(ORM):

    @staticmethod
    def evaluate_action_reward(action_pred: list, action_ref: list, cand_list: list, ref_list: list):
        f1 = []
        for i in range(len(action_pred)):
            ref_action = action_ref[i]
            pred_action = action_pred[i]

            ref_input = ref_list[i]
            cand_input = cand_list[i]

            ref_is_json = False
            try:
                ref_input_json = json.loads(ref_input)
                ref_is_json = True
            except Exception:
                ref_input_json = ref_input

            cand_is_json = False
            try:
                cand_input_json = json.loads(cand_input)
                cand_is_json = True
            except Exception:
                cand_input_json = cand_input

            if ref_action != pred_action or (ref_is_json ^ cand_is_json):
                f1.append(0)
            elif not ref_is_json and not cand_is_json:
                rougel = ReactORM.evaluate_rougel([ref_input_json], [cand_input_json])
                if rougel is None or rougel < 10:
                    f1.append(0)
                elif 10 <= rougel < 20:
                    f1.append(0.1)
                else:
                    f1.append(1)
            else:
                if not isinstance(ref_input_json, dict) or not isinstance(cand_input_json, dict):
                    # This cannot be happen, but:
                    # line 62, in evaluate_action_reward
                    # for k, v in ref_input_json.items():
                    # AttributeError: 'str' object has no attribute 'items'
                    # print(f'>>>>>>ref_input_json: {ref_input_json}, cand_input_json: {cand_input_json}')
                    f1.append(0)
                    continue

                half_match = 0
                full_match = 0
                if ref_input_json == {}:
                    if cand_input_json == {}:
                        f1.append(1)
                    else:
                        f1.append(0)
                else:
                    for k, v in ref_input_json.items():
                        if k in cand_input_json.keys():
                            if cand_input_json[k] == v:
                                full_match += 1
                            else:
                                half_match += 1

                    recall = (0.5 * half_match + full_match) / (len(ref_input_json) + 1e-30)
                    precision = (0.5 * half_match + full_match) / (len(cand_input_json) + 1e-30)
                    try:
                        f1.append((2 * recall * precision) / (recall + precision))
                    except Exception:
                        f1.append(0.0)

        if f1[0] == 1.0:
            return True
        else:
            return False

    @staticmethod
    def parse_action(text):
        if 'Action Input:' in text:
            input_idx = text.rindex('Action Input:')
            action_input = text[input_idx + len('Action Input:'):].strip()
        else:
            action_input = '{}'

        if 'Action:' in text:
            action_idx = text.rindex('Action:')
            action = text[action_idx + len('Action:'):].strip()
            if 'Action Input:' in action:
                input_idx = action.index('Action Input:')
                action = action[:input_idx].strip()
        else:
            action = 'none'
        return action, action_input

    @staticmethod
    def parse_output(text):
        action, action_input = ReactORM.parse_action(text)
        return action, action_input

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], solution: List[str], **kwargs) -> List[float]:
        rewards = []
        if not isinstance(infer_requests[0], str):
            predictions = [request['messages'][-1]['content'] for request in infer_requests]
        else:
            predictions = infer_requests
        for prediction, ground_truth in zip(predictions, solution):
            if prediction.endswith('Observation:'):
                prediction = prediction[:prediction.index('Observation:')].strip()
            action_ref = []
            action_input_ref = []
            action_pred = []
            action_input_pred = []
            reference = ground_truth
            prediction = prediction.replace('<|endoftext|>', '').replace('<|im_end|>', '').strip()
            ref_action, ref_input = ReactORM.parse_output(reference)
            pred_action, pred_input = ReactORM.parse_output(prediction)
            action_ref.append(ref_action)
            action_input_ref.append(ref_input)
            if pred_action is None:
                action_pred.append('none')
            else:
                action_pred.append(pred_action)

            if pred_input is None:
                action_input_pred.append('{}')
            else:
                action_input_pred.append(pred_input)

            reward = ReactORM.evaluate_action_reward(action_pred, action_ref, action_input_pred, action_input_ref)
            rewards.append(float(reward))
        return rewards

    @staticmethod
    def evaluate_rougel(cand_list: list, ref_list: list):
        if len(ref_list) == 0:
            return None
        try:
            from rouge import Rouge
            rouge = Rouge()
            rouge_score = rouge.get_scores(hyps=cand_list, refs=ref_list, avg=True)
            rougel = rouge_score['rouge-l']['f']
            return rougel
        except Exception:
            return None


class MathORM(ORM):

    def __init__(self):
        from transformers.utils import strtobool
        self.use_opencompass = strtobool(os.environ.get('USE_OPENCOMPASS_EVALUATOR', 'False'))
        if self.use_opencompass:
            from opencompass.datasets.math import MATHEvaluator
            self.evaluator = MATHEvaluator()

    @staticmethod
    def check_terminate(answers: Union[str, List[str]]) -> List[bool]:
        if isinstance(answers, str):
            answers = [answers]
        results = []
        for answer in answers:
            results.append('\\boxed' in answer)
        return results

    @staticmethod
    def extract_boxed_result(text):
        pattern = r'\\boxed{([^}]*)}'
        match = re.search(pattern, text)
        if match:
            return match.group(1).strip()
        else:
            return text

    @staticmethod
    def clean_latex(latex_str):
        latex_str = re.sub(r'\\\(|\\\)|\\\[|\\]', '', latex_str)
        latex_str = latex_str.replace('}}', '}').replace('{', '').replace('}', '')
        return latex_str.strip()

    @staticmethod
    def parse_expression(latex_str):
        from sympy import simplify
        from sympy.parsing.latex import parse_latex
        try:
            expr = parse_latex(latex_str)
            return simplify(expr)
        except Exception:
            return None

    @staticmethod
    def compare_consecutive(first, second):
        cleaned_list = [MathORM.clean_latex(latex) for latex in [first, second]]
        parsed_exprs = [MathORM.parse_expression(latex) for latex in cleaned_list]
        if hasattr(parsed_exprs[0], 'equals') and hasattr(parsed_exprs[1], 'equals'):
            value = parsed_exprs[0].equals(parsed_exprs[1])
        else:
            value = parsed_exprs[0] == parsed_exprs[1]
        if value is None:
            value = False
        return value

    def __call__(self, infer_requests: List[Union['InferRequest', Dict]], ground_truths: List[str],
                 **kwargs) -> List[float]:
        rewards = []
        predictions = [request.messages[-1]['content'] for request in infer_requests]
        for prediction, ground_truth in zip(predictions, ground_truths):
            if '# Answer' in prediction:
                prediction = prediction.split('# Answer')[1]
            if '# Answer' in ground_truth:
                ground_truth = ground_truth.split('# Answer')[1]
            prediction = prediction.strip()
            ground_truth = ground_truth.strip()
            prediction = MathORM.extract_boxed_result(prediction)
            ground_truth = MathORM.extract_boxed_result(ground_truth)
            if self.use_opencompass:
                reward = self.evaluator.is_equiv(prediction, ground_truth)
            else:
                reward = MathORM.compare_consecutive(prediction, ground_truth)
            rewards.append(float(reward))
        return rewards


class MathAccuracy(ORM):

    def __init__(self):
        import importlib.util
        assert importlib.util.find_spec('math_verify') is not None, (
            'The math_verify package is required but not installed. '
            "Please install it using 'pip install math_verify==0.5.2'.")

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        from latex2sympy2_extended import NormalizationConfig
        from math_verify import LatexExtractionConfig, parse, verify
        rewards = []
        for content, sol in zip(completions, solution):
            gold_parsed = parse(sol, extraction_mode='first_match')
            if len(gold_parsed) != 0:
                # We require the answer to be provided in correct latex (no malformed operators)
                answer_parsed = parse(
                    content,
                    extraction_config=[
                        LatexExtractionConfig(
                            normalization_config=NormalizationConfig(
                                nits=False,
                                malformed_operators=False,
                                basic_latex=True,
                                equations=True,
                                boxed=True,
                                units=True,
                            ),
                            # Ensures that boxed is tried first
                            boxed_match_priority=0,
                            try_extract_without_anchor=False,
                        )
                    ],
                    extraction_mode='first_match',
                )
                # edge case
                try:
                    reward = float(verify(gold_parsed, answer_parsed))
                except Exception:
                    reward = 0.0
            else:
                # If the gold solution is not parseable, we reward 0 to skip this example
                reward = 0.0
            rewards.append(reward)
        return rewards


class Format(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*<answer>.*?</answer>(?![\s\S])'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class ReActFormat(ORM):

    def __call__(self, completions, **kwargs) -> List[float]:
        """Reward function that checks if the completion has a specific format."""
        pattern = r'^<think>.*?</think>\s*Action:.*?Action Input:.*?$'
        matches = [re.match(pattern, content, re.DOTALL | re.MULTILINE) for content in completions]
        return [1.0 if match else 0.0 for match in matches]


class CosineReward(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self,
                 tokenizer=None,
                 cosine_min_len_value_wrong: float = -0.5,
                 cosine_max_len_value_wrong: float = 0.0,
                 cosine_min_len_value_correct: float = 1.0,
                 cosine_max_len_value_correct: float = 0.5,
                 cosine_max_len: int = 1000,
                 accuracy_orm=None):
        self.tokenizer = tokenizer
        self.min_len_value_wrong = cosine_min_len_value_wrong
        self.max_len_value_wrong = cosine_max_len_value_wrong
        self.min_len_value_correct = cosine_min_len_value_correct
        self.max_len_value_correct = cosine_max_len_value_correct
        self.max_len = cosine_max_len
        self.accuracy_orm = accuracy_orm or MathAccuracy()

    @staticmethod
    def cosfn(t, T, min_value, max_value):
        import math
        return max_value - (max_value - min_value) * (1 - math.cos(t * math.pi / T)) / 2

    def __call__(self, completions, solution, **kwargs) -> List[float]:
        acc_rewards = self.accuracy_orm(completions, solution, **kwargs)
        rewards = []
        for content, acc_reward in zip(completions, acc_rewards):
            is_correct = acc_reward >= 1.
            if is_correct:
                # Swap min/max for correct answers
                min_value = self.max_len_value_correct
                max_value = self.min_len_value_correct
            else:
                min_value = self.max_len_value_wrong
                max_value = self.min_len_value_wrong
            gen_len = len(self.tokenizer.encode(content))
            reward = self.cosfn(gen_len, self.max_len, min_value, max_value)
            rewards.append(reward)
        return rewards


class RepetitionPenalty(ORM):
    # https://arxiv.org/abs/2502.03373
    def __init__(self, repetition_n_grams: int = 3, repetition_max_penalty: float = -1.0):
        self.ngram_size = repetition_n_grams
        self.max_penalty = repetition_max_penalty

    @staticmethod
    def zipngram(text: str, ngram_size: int):
        words = text.lower().split()
        return zip(*[words[i:] for i in range(ngram_size)])

    def __call__(self, completions, **kwargs) -> List[float]:
        """
        reward function the penalizes repetitions

        Args:
            completions: List of model completions
        """
        rewards = []
        for completion in completions:
            if completion == '':
                rewards.append(0.0)
                continue
            if len(completion.split()) < self.ngram_size:
                rewards.append(0.0)
                continue

            ngrams = set()
            total = 0
            for ng in self.zipngram(completion, self.ngram_size):
                ngrams.add(ng)
                total += 1

            scaling = 1 - len(ngrams) / total
            reward = scaling * self.max_penalty
            rewards.append(reward)
        return rewards


class SoftOverlong(ORM):

    def __init__(self, tokenizer, soft_max_length, soft_cache_length):
        self.tokenizer = tokenizer
        assert soft_cache_length < soft_max_length
        self.soft_max_length = soft_max_length
        self.soft_cache_length = soft_cache_length

    def __call__(self, completions, **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            expected_len = self.soft_max_length - self.soft_cache_length
            exceed_len = completion_length - expected_len
            rewards.append(min(-exceed_len / self.soft_cache_length, 0))
        return rewards


class CompletionLength(ORM):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, completions: List[Any], **kwargs) -> List[float]:
        rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            if completion_length >= 128 and completion_length <= 768:
                rewards.append(1.0)
            else:
                rewards.append(-1.0)
        return rewards


from collections import Counter
from typing import List, Any

class VoteAnswer(ORM):
    """
    以出现次数最多的 completion 作为 gold，
    对每个 completion 逐一比较：
        相同 -> 奖励  +1
        不同 -> 奖励  -1
    """
    def __call__(self, completions: List[Any], **kwargs) -> List[float]:
        # 1. 边界情况：空输入
        if not completions:
            return []

        # 2. 选出出现次数最多的作为 gold（出现次数并列时取第一次出现的）
        counts = Counter(completions)
        most_common_val, most_common_cnt = counts.most_common(1)[0]

        # 如果希望出现次数并列时也严格“投票”，可以启用下面注释的代码：
        # top_count = most_common_cnt
        # candidates = [c for c, cnt in counts.items() if cnt == top_count]
        # most_common_val = candidates[0]  # 取最先出现的

        # 3. 计算奖励
        rewards = [1.0 if c == most_common_val else 0.0 for c in completions]
        return rewards


from collections import Counter
from typing import List, Any

class CombinedReward(ORM):
    """
    综合奖励函数，结合长度奖励和投票奖励
    - 长度奖励：completion 长度在 128-768 token 范围内给予奖励
    - 投票奖励：以出现次数最多的 completion 作为参考给予奖励
    """
    def __init__(self, tokenizer, length_weight=0.3, vote_weight=0.7):
        self.tokenizer = tokenizer
        self.length_weight = length_weight
        self.vote_weight = vote_weight

    def __call__(self, completions: List[Any], **kwargs) -> List[float]:
        # 边界情况：空输入
        if not completions:
            return []

        # 计算长度奖励
        length_rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            if completion_length >= 128 and completion_length <= 768:
                length_rewards.append(1.0)
            else:
                length_rewards.append(-1.0)

        # 计算投票奖励
        counts = Counter(completions)
        most_common_val, most_common_cnt = counts.most_common(1)[0]
        vote_rewards = [1.0 if c == most_common_val else 0.0 for c in completions]

        # 合并两个奖励
        combined_rewards = [
            self.length_weight * length_reward + self.vote_weight * vote_reward
            for length_reward, vote_reward in zip(length_rewards, vote_rewards)
        ]

        return combined_rewards

import numpy as np
from collections import Counter
from typing import List, Any

class FinalCombinedReward(ORM):
    """
    最终修正版的综合奖励函数（适配 GRPO）
    - 长度奖励：使用平滑的高斯函数，奖励在目标范围内平滑过渡。
    - 软投票奖励：奖励与得票数成正比，而不是赢者通吃。
    - **移除了内部的奖励标准化步骤**，将此任务交由 GRPO 算法本身处理。
    """
    def __init__(self, tokenizer, length_weight=0.8, vote_weight=0.2):
        self.tokenizer = tokenizer
        self.length_weight = length_weight
        self.vote_weight = vote_weight

        # 定义长度奖励的目标参数
        self.target_length = (128 + 768) / 2  # 理想中心长度: 448
        self.tolerance = (768 - 128) / 2    # 长度容忍范围: 320

    def _calculate_smooth_length_rewards(self, completions: List[str]) -> List[float]:
        """计算平滑的长度奖励，使用高斯函数"""
        rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            exponent = -((completion_length - self.target_length) ** 2) / (2 * self.tolerance ** 2)
            reward = np.exp(exponent)
            rewards.append(reward)
        return rewards

    def _calculate_soft_vote_rewards(self, completions: List[str]) -> List[float]:
        """计算软投票奖励，奖励与票数成正比"""
        if not completions:
            return []
        
        counts = Counter(completions)
        if not counts:
            return [0.0] * len(completions)

        most_common_cnt = counts.most_common(1)[0][1]
        if most_common_cnt == 0:
             return [0.0] * len(completions)

        rewards = [counts[c] / most_common_cnt for c in completions]
        return rewards

    def __call__(self, completions: List[Any], **kwargs) -> List[float]:
        if not completions:
            return []

        # 1. 计算各部分奖励
        length_rewards = self._calculate_smooth_length_rewards(completions)
        vote_rewards = self._calculate_soft_vote_rewards(completions)

        # 2. 加权合并奖励
        # 这个 combined_rewards 是一个绝对分数，其值域大约在 [0, 1] 之间
        # GRPO 将会接收这个列表，并在其内部进行标准化来计算优势
        combined_rewards = [
            self.length_weight * lr + self.vote_weight * vr
            for lr, vr in zip(length_rewards, vote_rewards)
        ]

        # 直接返回合并后的原始奖励分数，不再进行任何标准化处理
        return combined_rewards

import numpy as np
from typing import List, Any
from rouge_score import rouge_scorer

class RougeReward(ORM):
    """
    使用 ROUGE-L 分数作为奖励，替代投票法，以鼓励语义一致性和表达多样性。
    - 长度奖励：依旧使用平滑的高斯函数。
    - ROUGE 奖励：计算每个 completion 与批次内其他所有 completions 的平均 ROUGE-L F1-score。
    """
    def __init__(self, tokenizer, length_weight=0.5, rouge_weight=0.5):
        self.tokenizer = tokenizer
        self.length_weight = length_weight
        self.rouge_weight = rouge_weight
        
        # 初始化 ROUGE scorer
        self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)

        # 长度奖励参数
        self.target_length = (128 + 768) / 2
        self.tolerance = (768 - 128) / 2

    def _calculate_smooth_length_rewards(self, completions: List[str]) -> List[float]:
        rewards = []
        for completion in completions:
            completion_length = len(self.tokenizer.encode(completion))
            exponent = -((completion_length - self.target_length) ** 2) / (2 * self.tolerance ** 2)
            rewards.append(np.exp(exponent))
        return rewards

    def _calculate_rouge_rewards(self, completions: List[str]) -> List[float]:
        if len(completions) <= 1:
            return [1.0] * len(completions)

        rouge_rewards = []
        for i, comp_i in enumerate(completions):
            scores = []
            for j, comp_j in enumerate(completions):
                if i == j:
                    continue
                # 计算 comp_i 和 comp_j 之间的 ROUGE-L fmeasure
                score = self.scorer.score(comp_i, comp_j)['rougeL'].fmeasure
                scores.append(score)
            
            # 使用平均分作为这个 completion 的奖励
            # 如果 scores 为空（只有一个 completion），则奖励为1.0
            avg_score = np.mean(scores) if scores else 1.0
            rouge_rewards.append(avg_score)
            
        return rouge_rewards

    def __call__(self, completions: List[Any], **kwargs) -> List[float]:
        if not completions:
            return []

        length_rewards = self._calculate_smooth_length_rewards(completions)
        rouge_rewards = self._calculate_rouge_rewards(completions)

        combined_rewards = [
            self.length_weight * lr + self.rouge_weight * rr
            for lr, rr in zip(length_rewards, rouge_rewards)
        ]
        
        return combined_rewards


orms = {
    'toolbench': ReactORM,
    'math': MathORM,
    'accuracy': MathAccuracy,
    'format': Format,
    'react_format': ReActFormat,
    'cosine': CosineReward,
    'repetition': RepetitionPenalty,
    'soft_overlong': SoftOverlong,
    'vote_answer': VoteAnswer,
    'completionlength': CompletionLength,
    'CombinedReward': CombinedReward,
    'FinalCombinedReward': FinalCombinedReward,
    "RougeReward": RougeReward
}
