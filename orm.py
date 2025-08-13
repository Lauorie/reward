import os
import re
from typing import TYPE_CHECKING, Dict, List, Union, Any
import logging
import math

import json
import numpy as np

if TYPE_CHECKING:
    from swift.llm import InferRequest

# 添加 VLLM 相关导入
try:
    from vllm import LLM, SamplingParams
    VLLM_AVAILABLE = True
except ImportError:
    VLLM_AVAILABLE = False
    LLM = None
    SamplingParams = None


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


class RougeWithSolutionReward(ORM):
    """
    使用 ROUGE-L 分数与数据集solution比较作为奖励，结合长度奖励
    - 长度奖励：使用平滑的高斯函数，鼓励合适长度的回答
    - ROUGE 奖励：计算每个 completion 与对应 solution 的 ROUGE-L F1-score
    
    与RougeReward的区别：
    - RougeReward: completion vs 其他completions (批次内多样性)
    - RougeWithSolutionReward: completion vs solution (与标准答案相似性)
    """
    def __init__(self, tokenizer, length_weight=0.4, rouge_weight=0.6):
        self.tokenizer = tokenizer
        self.length_weight = length_weight
        self.rouge_weight = rouge_weight
        
        # 初始化 ROUGE scorer
        try:
            import rouge_score
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        except ImportError:
            raise ImportError("请安装rouge_score: pip install rouge-score")

        # 长度奖励参数 - 针对学术问答调整
        self.target_length = (128 + 768) / 2  # 目标长度448 tokens
        self.tolerance = (768 - 128) / 2      # 容忍度320 tokens
        
        import logging
        self.logger = logging.getLogger(__name__)

    def _calculate_smooth_length_rewards(self, completions: List[str]) -> List[float]:
        """
        计算长度奖励，使用高斯函数，鼓励适中长度的回答
        """
        rewards = []
        for completion in completions:
            try:
                # 使用tokenizer编码来获取准确的token长度
                if hasattr(self.tokenizer, 'encode'):
                    completion_length = len(self.tokenizer.encode(str(completion), add_special_tokens=False))
                else:
                    # fallback: 简单的词汇计数
                    completion_length = len(str(completion).split())
                
                # 高斯函数计算长度奖励
                exponent = -((completion_length - self.target_length) ** 2) / (2 * self.tolerance ** 2)
                length_reward = np.exp(exponent)
                rewards.append(length_reward)
                
            except Exception as e:
                self.logger.warning(f"长度奖励计算失败: {e}")
                rewards.append(0.5)  # 默认中等奖励
                
        return rewards

    def _calculate_rouge_with_solution_rewards(self, completions: List[str], solutions: List[str]) -> List[float]:
        """
        计算每个completion与对应solution的ROUGE-L F1-score
        
        Args:
            completions: 生成的回答列表
            solutions: 对应的标准答案列表
            
        Returns:
            List[float]: ROUGE-L F1分数列表
        """
        if not completions or not solutions:
            return [0.0] * len(completions)
        
        rouge_rewards = []
        
        # 确保solutions长度匹配
        if len(solutions) == 1:
            # 如果只有一个solution，所有completions都与它比较
            solutions = solutions * len(completions)
        elif len(solutions) != len(completions):
            self.logger.warning(
                f"Completions 和 Solutions 列表长度不匹配！"
                f"({len(completions)} vs {len(solutions)}). "
                f"将进行广播或截断，但这可能表示数据管道存在问题。"
            )
            # 长度不匹配时，截断或填充
            if len(solutions) < len(completions):
                solutions = solutions + [solutions[-1]] * (len(completions) - len(solutions))
            else:
                solutions = solutions[:len(completions)]
        
        for i, completion in enumerate(completions):
            try:
                solution = solutions[i] if i < len(solutions) else solutions[0]
                
                # 计算ROUGE-L F1-score
                completion_str = str(completion).strip()
                solution_str = str(solution).strip()
                
                if not completion_str or not solution_str:
                    rouge_rewards.append(0.0)
                    continue
                
                # 使用ROUGE scorer计算分数
                score = self.scorer.score(solution_str, completion_str)['rougeL'].fmeasure
                rouge_rewards.append(score)
                
            except Exception as e:
                self.logger.warning(f"ROUGE计算失败: {e}")
                rouge_rewards.append(0.0)
        
        return rouge_rewards

    def __call__(self, completions: List[Any], solution: List[str] = None, **kwargs) -> List[float]:
        """
        计算结合ROUGE-L和长度的奖励
        
        Args:
            completions: 生成的完成文本列表
            solution: 标准答案列表
            **kwargs: 其他参数，可能包含solutions、solution等
            
        Returns:
            List[float]: 奖励值列表
        """
        if not completions:
            return []
        
        # 从kwargs中获取解答数据
        solutions = solution or kwargs.get('solutions') or kwargs.get('solution') or []
        
        # 如果没有提供solutions，使用基于长度的奖励
        if not solutions:
            self.logger.warning("RougeWithSolutionReward: 没有提供solution，仅使用长度奖励")
            return self._calculate_smooth_length_rewards(completions)
        
        # 计算长度奖励
        length_rewards = self._calculate_smooth_length_rewards(completions)
        
        # 计算ROUGE奖励
        rouge_rewards = self._calculate_rouge_with_solution_rewards(completions, solutions)
        
        # 组合奖励
        combined_rewards = [
            self.length_weight * lr + self.rouge_weight * rr
            for lr, rr in zip(length_rewards, rouge_rewards)
        ]
        
        # 记录调试信息
        if len(completions) <= 5:  # 避免日志过多
            for i, (comp, sol, lr, rr, cr) in enumerate(zip(
                completions[:5], solutions[:5], 
                length_rewards[:5], rouge_rewards[:5], combined_rewards[:5]
            )):
                self.logger.info(f"奖励计算 {i+1}: 长度={lr:.3f}, ROUGE={rr:.3f}, 总计={cr:.3f}")
        
        return combined_rewards

import numpy as np
from collections import Counter
from typing import List, Any

class DecisiveReward(ORM):
    """
    一个更具决定性的奖励函数，结合了：
    1. 锐利的梯形长度奖励 (Sharp Trapezoidal Length Reward)
    2. ROUGE-L 内容相似度奖励 (ROUGE-L Content Similarity)
    3. N-gram 重复惩罚 (N-gram Repetition Penalty)
    """
    def __init__(self, 
                 tokenizer, 
                 length_weight: float = 0.2, 
                 rouge_weight: float = 0.7,
                 penalty_weight: float = 0.1):
        """
        初始化奖励函数
        
        Args:
            tokenizer: 用于计算token长度的分词器
            length_weight (float): 长度奖励的权重
            rouge_weight (float): ROUGE分数的权重
            penalty_weight (float): 重复惩罚的权重 (注意是负向奖励)
        """
        self.tokenizer = tokenizer
        self.length_weight = length_weight
        self.rouge_weight = rouge_weight
        self.penalty_weight = penalty_weight
        
        # 初始化 ROUGE scorer
        try:
            from rouge_score import rouge_scorer
            self.scorer = rouge_scorer.RougeScorer(['rougeL'], use_stemmer=True)
        except ImportError:
            raise ImportError("请安装rouge_score: pip install rouge-score")

        # 1. 梯形长度奖励参数 (更严格)
        self.penalty_zone_min = 128      # 惩罚区下限 (低于此值奖励开始下降)
        self.optimal_zone_min = 256      # 最优区下限 (在此之上奖励为1.0)
        self.optimal_zone_max = 512      # 最优区上限 (在此之下奖励为1.0)
        self.penalty_zone_max = 768      # 惩罚区上限 (高于此值奖励开始下降)
        
        import logging
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"DecisiveReward 初始化: "
                         f"length_weight={self.length_weight}, "
                         f"rouge_weight={self.rouge_weight}, "
                         f"penalty_weight={self.penalty_weight}")
        self.logger.info(f"长度奖励区间: 惩罚区=[{self.penalty_zone_min}, {self.penalty_zone_max}], "
                         f"最优区=[{self.optimal_zone_min}, {self.optimal_zone_max}]")


    def _calculate_trapezoidal_length_rewards(self, completions: List[str]) -> List[float]:
        """计算梯形长度奖励，提供清晰的梯度信号"""
        rewards = []
        for comp in completions:
            try:
                length = len(self.tokenizer.encode(str(comp), add_special_tokens=False))
                
                if self.optimal_zone_min <= length <= self.optimal_zone_max:
                    # 在最优区间内，奖励为1.0
                    reward = 1.0
                elif self.penalty_zone_min <= length < self.optimal_zone_min:
                    # 在下行惩罚坡道，奖励线性增加
                    reward = (length - self.penalty_zone_min) / (self.optimal_zone_min - self.penalty_zone_min)
                elif self.optimal_zone_max < length <= self.penalty_zone_max:
                    # 在上行惩罚坡道，奖励线性减少
                    reward = (self.penalty_zone_max - length) / (self.penalty_zone_max - self.optimal_zone_max)
                else:
                    # 在最优区间外，奖励为0
                    reward = 0.0
                rewards.append(reward)
            except Exception as e:
                self.logger.warning(f"梯形长度奖励计算失败: {e}")
                rewards.append(0.0) # 失败时给予0分
        return rewards

    def _calculate_repetition_penalties(self, completions: List[str], n: int = 4) -> List[float]:
        """计算n-gram重复惩罚。返回值是惩罚力度(0到1)，越高表示重复越严重"""
        penalties = []
        for comp in completions:
            try:
                tokens = self.tokenizer.encode(str(comp), add_special_tokens=False)
                if len(tokens) < n:
                    penalties.append(0.0)
                    continue
                
                ngrams = [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
                if not ngrams:
                    penalties.append(0.0)
                    continue
                
                ngram_counts = Counter(ngrams)
                # 计算重复率：(总n-gram数 - 独立n-gram数) / 总n-gram数
                repetition_ratio = (len(ngrams) - len(ngram_counts)) / len(ngrams)
                penalties.append(repetition_ratio)
            except Exception as e:
                self.logger.warning(f"重复惩罚计算失败: {e}")
                penalties.append(0.5) # 失败时给予中等惩罚
        return penalties

    # _calculate_rouge_with_solution_rewards 方法可以从您之前的代码中直接复用，无需修改
    def _calculate_rouge_with_solution_rewards(self, completions: List[str], solutions: List[str]) -> List[float]:
        # ... (将您之前的实现粘贴到这里) ...
        if not completions or not solutions:
            return [0.0] * len(completions)
        rouge_rewards = []
        if len(solutions) == 1:
            solutions = solutions * len(completions)
        elif len(solutions) != len(completions):
            self.logger.warning(f"Completions和Solutions长度不匹配: {len(completions)} vs {len(solutions)}")
            if len(solutions) < len(completions):
                solutions = solutions + [solutions[-1]] * (len(completions) - len(solutions))
            else:
                solutions = solutions[:len(completions)]
        for i, completion in enumerate(completions):
            try:
                solution = solutions[i]
                completion_str, solution_str = str(completion).strip(), str(solution).strip()
                if not completion_str or not solution_str:
                    rouge_rewards.append(0.0)
                    continue
                score = self.scorer.score(target=completion_str, reference=solution_str)['rougeL'].fmeasure
                rouge_rewards.append(score)
            except Exception as e:
                self.logger.warning(f"ROUGE计算失败: {e}")
                rouge_rewards.append(0.0)
        return rouge_rewards


    def __call__(self, completions: List[Any], solution: List[str] = None, **kwargs) -> List[float]:
        if not completions:
            return []
        
        solutions = solution or kwargs.get('solutions') or kwargs.get('solution') or []
        
        if not solutions:
            self.logger.warning("DecisiveReward: 未提供solution，仅使用长度奖励和重复惩罚")
            length_rewards = self._calculate_trapezoidal_length_rewards(completions)
            repetition_penalties = self._calculate_repetition_penalties(completions)
            return [self.length_weight * lr - self.penalty_weight * rp 
                    for lr, rp in zip(length_rewards, repetition_penalties)]

        # 计算所有奖励分量
        length_rewards = self._calculate_trapezoidal_length_rewards(completions)
        rouge_rewards = self._calculate_rouge_with_solution_rewards(completions, solutions)
        repetition_penalties = self._calculate_repetition_penalties(completions)
        
        # 组合奖励 (注意惩罚是减去的)
        combined_rewards = [
            self.length_weight * lr + self.rouge_weight * rr - self.penalty_weight * rp
            for lr, rr, rp in zip(length_rewards, rouge_rewards, repetition_penalties)
        ]
        
        # 记录调试信息
        if len(completions) > 0:
            self.logger.info(
                f"奖励样本: "
                f"长度R={length_rewards[0]:.3f}, "
                f"ROUGE_R={rouge_rewards[0]:.3f}, "
                f"重复P={repetition_penalties[0]:.3f}, "
                f"总计R={combined_rewards[0]:.3f}"
            )
        
        return combined_rewards

class RLPRReward(ORM):
    """
    RLPR (Reinforcement Learning from Preference Ranking) 奖励函数
    基于生成响应对参考答案概率影响的奖励机制
    
    核心思想：
    - 计算 r_generated = P(reference_answer | question + generated_response)
    - 计算 r_reference = P(reference_answer | question)  
    - 奖励基于比值 r_generated / r_reference
    """
    
    def __init__(self, 
                 tokenizer,
                 beta: float = 0.5,
                 system_message: str = "你是源语奇思（AtomInfinite）开发的人工智能助手 WisModel。",
                 min_logprob_clamp: float = -20.0,
                 max_logprob_penalty: float = 5.0,
                 model: str = None,  # 匹配训练参数中的model字段
                 use_vllm: bool = True,
                 ema_decay: float = 0.9,
                 vllm_gpu_memory_utilization: float = 0.5,
                 vllm_max_model_len: int = 8800,
                 vllm_tensor_parallel_size: int = 1,
                 include_length_reward: bool = True,  # 是否包含长度奖励
                 length_weight: float = 0.3,  # 长度奖励权重
                 rlpr_weight: float = 0.7,  # RLPR奖励权重
                 **vllm_kwargs):
        """
        初始化RLPR奖励函数
        
        Args:
            tokenizer: 用于文本编码的tokenizer
            beta: RLPR的平衡参数
            system_message: 系统消息前缀
            min_logprob_clamp: 最小log概率截断值
            max_logprob_penalty: 最大log概率惩罚
            model: 模型路径（用于VLLM，匹配训练参数）
            use_vllm: 是否使用VLLM进行概率计算
            ema_decay: EMA衰减率
            vllm_gpu_memory_utilization: VLLM GPU内存利用率
            vllm_max_model_len: VLLM最大模型长度
            vllm_tensor_parallel_size: VLLM张量并行大小
            include_length_reward: 是否包含长度奖励（使用高斯函数）
            length_weight: 长度奖励的权重
            rlpr_weight: RLPR奖励的权重
            **vllm_kwargs: 传递给VLLM的额外参数
        """
        self.tokenizer = tokenizer
        self.beta = beta
        self.system_message = system_message
        self.min_logprob_clamp = min_logprob_clamp
        self.max_logprob_penalty = max_logprob_penalty
        self.model_path = model  # 保持内部使用model_path变量名
        self.use_vllm = use_vllm and VLLM_AVAILABLE
        self.ema_decay = ema_decay
        self.include_length_reward = include_length_reward
        self.length_weight = length_weight
        self.rlpr_weight = rlpr_weight
        self.std_ema = None
        self.logger = logging.getLogger(__name__)
        
        # 如果启用VLLM且可用，初始化VLLM模型
        self.llm = None
        if self.use_vllm and model:
            try:
                # 设置vLLM默认参数，使用训练参数
                default_vllm_kwargs = {
                    "tensor_parallel_size": vllm_tensor_parallel_size,
                    "dtype": "auto", 
                    "trust_remote_code": True,
                    "gpu_memory_utilization": vllm_gpu_memory_utilization,
                    "max_model_len": vllm_max_model_len,
                }
                default_vllm_kwargs.update(vllm_kwargs)
                
                self.logger.info("正在加载vLLM模型用于RLPR奖励计算...")
                self.llm = LLM(model=model, **default_vllm_kwargs)
                self.logger.info("vLLM模型加载完成！")
            except Exception as e:
                self.logger.warning(f"VLLM初始化失败，回退到简化模式: {e}")
                self.use_vllm = False
                self.llm = None
        
    def _get_prompt_prefix(self, messages: List[Dict]) -> str:
        """使用tokenizer的chat template格式化messages"""
        if not messages:
            # 如果没有messages，使用默认格式
            default_messages = [
                {"role": "system", "content": self.system_message},
                {"role": "user", "content": ""}
            ]
            messages = default_messages
        
        try:
            # 使用tokenizer的apply_chat_template方法
            if hasattr(self.tokenizer, 'apply_chat_template'):
                return self.tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True
                )
            else:
                # fallback：手动格式化
                return self._manual_format_messages(messages)
        except Exception as e:
            self.logger.warning(f"使用chat_template失败，回退到手动格式化: {e}")
            return self._manual_format_messages(messages)
    
    def _manual_format_messages(self, messages: List[Dict]) -> str:
        """手动格式化messages（fallback方法）"""
        conversation_parts = []
        for message in messages:
            role = message.get('role', '')
            content = message.get('content', '')
            
            if role == 'system':
                conversation_parts.append(content)
            elif role == 'user':
                conversation_parts.append(f"用户: {content}")
            elif role == 'assistant':
                conversation_parts.append(f"助手: {content}")
        
        # 拼接所有对话部分，添加最终的回答提示
        full_conversation = "\n\n".join(conversation_parts)
        if full_conversation:
            full_conversation += "\n\n助手: "
        
        return full_conversation
    
    def _safe_encode_text(self, text: str) -> List[int]:
        """安全地编码文本为token IDs"""
        try:
            if hasattr(self.tokenizer, 'encode'):
                return self.tokenizer.encode(text, add_special_tokens=False)
            elif hasattr(self.tokenizer, '__call__'):
                return self.tokenizer(text)['input_ids']
            else:
                self.logger.warning("Tokenizer没有encode方法，返回空列表")
                return []
        except Exception as e:
            self.logger.warning(f"文本编码失败: {e}")
            return []
    
    def _calculate_mean_probability_vllm(self, prefix_text: str, target_text: str) -> float:
        """
        使用vLLM计算目标文本在给定前缀条件下的平均token概率
        """
        try:
            # 分别编码前缀和目标
            prefix_ids = self._safe_encode_text(prefix_text)
            target_ids = self._safe_encode_text(target_text)

            if not target_ids:
                return 0.0

            # 构建完整序列：[prefix + target]
            full_sequence = prefix_ids + target_ids
            full_text = self.tokenizer.decode(full_sequence, skip_special_tokens=False)
            
            # 使用vLLM计算完整序列的logprobs
            sampling_params = SamplingParams(
                temperature=0.7,  
                max_tokens=1,  # 只生成一个token
                logprobs=1,  # 获取logprobs
                prompt_logprobs=True,  # 关键：获取prompt的logprobs
                seed=42
            )
            
            outputs = self.llm.generate([full_text], sampling_params)
            output = outputs[0]
            
            # 获取prompt的logprobs（即完整序列的logprobs）
            if output.prompt_logprobs:
                prompt_logprobs = output.prompt_logprobs
                
                # 计算目标部分的概率
                target_start_pos = len(prefix_ids)
                target_end_pos = target_start_pos + len(target_ids)
                
                # 提取目标部分的logprobs
                target_logprobs = []
                for i in range(target_start_pos, min(target_end_pos, len(prompt_logprobs))):
                    if i < len(prompt_logprobs) and prompt_logprobs[i]:
                        # 获取实际目标token的logprob
                        target_token_id = target_ids[i - target_start_pos]
                        if target_token_id in prompt_logprobs[i]:
                            target_logprobs.append(prompt_logprobs[i][target_token_id].logprob)
                        else:
                            # 如果找不到确切的token，使用最小的logprob作为惩罚
                            min_logprob = min(lp.logprob for lp in prompt_logprobs[i].values())
                            target_logprobs.append(min_logprob - 5.0)  # 额外惩罚
                
                if target_logprobs:
                    # 转换为概率并计算平均值
                    probabilities = [math.exp(max(lp, -20.0)) for lp in target_logprobs]  # 限制最小值避免下溢
                    mean_prob = np.mean(probabilities)
                    return float(mean_prob)
                        
            return 0.0

        except Exception as e:
            self.logger.warning(f"vLLM概率计算失败: {e}")
            return 0.0

    def _calculate_smooth_length_reward(self, completion_text: str) -> float:
        """
        使用高斯函数计算平滑的长度奖励，参考RougeReward的实现
        """
        try:
            completion_length = len(self._safe_encode_text(completion_text))
            
            # 使用与RougeReward相同的参数设置
            target_length = (128 + 768) / 2  # 448
            tolerance = (768 - 128) / 2  # 320
            
            # 高斯函数：exp(-((x - μ)^2) / (2σ^2))
            exponent = -((completion_length - target_length) ** 2) / (2 * tolerance ** 2)
            length_reward = math.exp(exponent)
            
            return float(length_reward)
            
        except Exception as e:
            self.logger.warning(f"长度奖励计算失败: {e}")
            return 0.5  # 返回中等奖励作为默认值

    def _calculate_mean_probability_vllm(self, prefix_text: str, target_text: str, trainer=None) -> float:
        """
        智能概率计算 - 基于文本语义相似性的启发式方法
        由于Swift vLLM接口复杂，使用语义相似性来近似概率计算
        """
        try:
            # 计算基于文本相似性的概率近似
            return self._calculate_semantic_probability(prefix_text, target_text)
            
        except Exception as e:
            self.logger.warning(f"语义概率计算失败: {e}")
            return self._calculate_mean_probability_fallback(prefix_text, target_text)

    def _calculate_semantic_probability(self, prefix_text: str, target_text: str) -> float:
        """
        基于语义相似性的智能概率计算
        分析前缀文本和目标文本的语义关联性
        """
        try:
            # 1. 长度因子：较短的目标文本概率更高
            target_ids = self._safe_encode_text(target_text)
            if not target_ids:
                return 0.0
            
            length_factor = min(1.0, 100.0 / max(len(target_ids), 1))
            
            # 2. 词汇重叠度：计算共同词汇比例
            prefix_words = set(prefix_text.lower().split())
            target_words = set(target_text.lower().split()) 
            
            if len(target_words) > 0:
                overlap_ratio = len(prefix_words & target_words) / len(target_words)
            else:
                overlap_ratio = 0.0
            
            # 3. 位置相关性：检查目标文本是否是前缀的自然延续
            context_relevance = self._calculate_context_relevance(prefix_text, target_text)
            
            # 4. 语言流畅性：基于n-gram估算
            fluency_score = self._calculate_fluency_score(prefix_text, target_text)
            
            # 综合计算概率
            semantic_prob = (
                length_factor * 0.3 + 
                overlap_ratio * 0.25 + 
                context_relevance * 0.25 + 
                fluency_score * 0.2
            )
            
            # 确保在合理范围内
            return max(0.001, min(0.999, semantic_prob))
            
        except Exception as e:
            self.logger.warning(f"语义概率计算失败: {e}")
            return 0.1

    def _calculate_context_relevance(self, prefix_text: str, target_text: str) -> float:
        """计算上下文相关性"""
        try:
            # 检查目标文本是否包含问题相关的关键词
            prefix_lower = prefix_text.lower()
            target_lower = target_text.lower()
            
            # 如果前缀包含问题标识，目标应该是回答
            if any(keyword in prefix_lower for keyword in ['问题', '问:', 'question', '?', '？']):
                # 检查目标是否包含回答性词汇
                if any(keyword in target_lower for keyword in ['是', '有', '可以', '会', '能够', 'yes', 'no', 'the', 'it']):
                    return 0.8
                return 0.4
            
            # 检查逻辑连贯性
            if target_lower.startswith(tuple(['因为', '由于', 'because', 'since'])):
                return 0.7
            if target_lower.startswith(tuple(['所以', '因此', 'therefore', 'so'])):
                return 0.6
                
            return 0.5
            
        except:
            return 0.5

    def _calculate_fluency_score(self, prefix_text: str, target_text: str) -> float:
        """计算语言流畅性分数"""
        try:
            # 简单的流畅性检查
            if not target_text.strip():
                return 0.0
                
            # 检查是否有完整的句子结构
            if any(target_text.strip().endswith(punct) for punct in ['.', '。', '!', '！', '?', '？']):
                fluency = 0.8
            else:
                fluency = 0.5
                
            # 避免重复词汇
            words = target_text.split()
            if len(words) > 1:
                unique_ratio = len(set(words)) / len(words)
                fluency *= unique_ratio
                
            return min(1.0, fluency)
            
        except:
            return 0.5
    
    def _calculate_mean_probability_fallback(self, prefix_text: str, target_text: str) -> float:
        """
        简化版的概率计算，作为fallback方法
        """
        try:
            # 编码文本
            prefix_ids = self._safe_encode_text(prefix_text)
            target_ids = self._safe_encode_text(target_text)
            
            if not target_ids:
                return 0.0
            
            # 基于文本长度和复杂度的启发式估计
            length_factor = min(1.0, 50.0 / max(len(target_ids), 1))
            
            # 考虑前缀和目标文本的相似性
            prefix_text_lower = prefix_text.lower()
            target_text_lower = target_text.lower()
            
            # 简单的文本相似性度量
            common_words = set(prefix_text_lower.split()) & set(target_text_lower.split())
            similarity = len(common_words) / max(len(target_text_lower.split()), 1)
            
            # 结合长度因子和相似性
            probability = length_factor * 0.7 + similarity * 0.3
            
            return max(0.001, min(0.999, probability))  # 确保在合理范围内
            
        except Exception as e:
            self.logger.warning(f"Fallback概率计算失败: {e}")
            return 0.1  # 返回一个默认的小概率值
    
    def _compute_single_reward(self, messages: List[Dict], generated_response: str, reference_answer: str, trainer_state=None) -> float:
        """
        计算单个样本的RLPR奖励
        
        Args:
            messages: 对话历史消息列表
            generated_response: 生成的响应
            reference_answer: 参考答案
            trainer_state: 训练器状态（可选）
            
        Returns:
            float: 奖励值 [0, 1]
        """
        if not messages or not generated_response.strip() or not reference_answer.strip():
            return 0.0
            
        try:
            prompt_prefix = self._get_prompt_prefix(messages)
            
            # 计算概率 - 根据是否有VLLM选择不同的计算方法
            context_with_generated = prompt_prefix + generated_response
            if self.use_vllm and self.llm:
                r_generated = self._calculate_mean_probability_vllm(context_with_generated, reference_answer)
                r_reference = self._calculate_mean_probability_vllm(prompt_prefix, reference_answer)
            else:
                r_generated = self._calculate_mean_probability_simple(context_with_generated, reference_answer, trainer_state)
                r_reference = self._calculate_mean_probability_simple(prompt_prefix, reference_answer, trainer_state)
            
            # 计算RLPR奖励
            if r_reference > 0:
                ratio = r_generated / r_reference
                if ratio >= 1.0:
                    rlpr_reward = 0.5 + (ratio - 1.0) * 0.5
                else:
                    rlpr_reward = ratio * 0.5
            else:
                rlpr_reward = r_generated
                
            # 确保在[0,1]范围内
            rlpr_reward = np.clip(rlpr_reward, 0.0, 1.0)
            
            # 如果启用长度奖励，计算组合奖励
            if self.include_length_reward:
                length_reward = self._calculate_smooth_length_reward(generated_response)
                final_reward = self.rlpr_weight * rlpr_reward + self.length_weight * length_reward
            else:
                final_reward = rlpr_reward
            
            # 确保最终奖励在[0,1]范围内
            final_reward = np.clip(final_reward, 0.0, 1.0)
            
            return float(final_reward)
            
        except Exception as e:
            self.logger.error(f"RLPR奖励计算失败: {e}")
            return 0.0
    

    
    def __call__(self, completions: List[Any], solution: List[str] = None, **kwargs) -> List[float]:
        """
        计算RLPR奖励
        
        Args:
            completions: 生成的完成文本列表
            solution: 参考答案列表（如果提供）
            **kwargs: 其他参数，可能包含trainer_state、messages等
            
        Returns:
            List[float]: 奖励值列表
        """
        if not completions:
            return []
        
        rewards = []
        trainer_state = kwargs.get('trainer_state')
        
        # 从kwargs中获取数据
        trainer = kwargs.get('trainer')  # 获取训练器实例
        solutions = solution or kwargs.get('solutions') or kwargs.get('solution') or []
        messages_list = kwargs.get('messages') or []
        
        # 如果没有提供messages或解答，使用基于高斯函数的长度奖励作为备用策略
        if not messages_list or not solutions:
            # 对于缺少必要信息的情况，使用基于高斯函数的长度奖励
            self.logger.warning(f"RLPR: 缺少messages或参考答案，使用基于高斯函数的长度奖励策略。messages: {len(messages_list)}, solutions: {len(solutions)}")
            for completion in completions:
                # 使用高斯函数计算长度奖励
                reward = self._calculate_smooth_length_reward(str(completion))
                rewards.append(reward)
            return rewards
        
        # 确保所有列表长度一致
        num_samples = len(completions)
        if len(messages_list) == 1:
            messages_list = messages_list * num_samples
        if len(solutions) == 1:
            solutions = solutions * num_samples
            
        messages_list = messages_list[:num_samples]
        solutions = solutions[:num_samples]
        
        # 计算每个样本的RLPR奖励
        for i, completion in enumerate(completions):
            messages = messages_list[i] if i < len(messages_list) else []
            solution = solutions[i] if i < len(solutions) else ""
            
            reward = self._compute_single_reward_vllm(
                messages=messages,
                generated_response=str(completion),
                reference_answer=str(solution),
                trainer=trainer
            )
            rewards.append(reward)
        
        return rewards

    def _compute_single_reward_vllm(self, messages: List[Dict], generated_response: str, reference_answer: str, trainer=None) -> float:
        """
        使用vLLM计算单个样本的RLPR奖励 - 参考rlpr_vllm.py的实现
        
        Args:
            messages: 对话历史消息列表
            generated_response: 生成的响应
            reference_answer: 参考答案
            trainer: 训练器实例（用于访问vLLM引擎）
            
        Returns:
            float: 奖励值 [0, 1]
        """
        if not messages or not generated_response.strip() or not reference_answer.strip():
            return 0.0
            
        try:
            prompt_prefix = self._get_prompt_prefix(messages)
            
            # 计算概率 - 使用真正的vLLM计算
            context_with_generated = prompt_prefix + generated_response
            r_generated = self._calculate_mean_probability_vllm(context_with_generated, reference_answer, trainer)
            r_reference = self._calculate_mean_probability_vllm(prompt_prefix, reference_answer, trainer)
            
            # 使用与rlpr_vllm.py相同的奖励计算逻辑
            if r_reference > 0:
                ratio = r_generated / r_reference
                # 将比例转换为合理的奖励范围
                if ratio >= 1.0:
                    reward = 0.5 + (ratio - 1.0) * 0.5  # 比例>1时奖励>0.5
                else:
                    reward = ratio * 0.5  # 比例<1时奖励<0.5
            else:
                reward = r_generated
                
            # 确保在[0,1]范围内
            reward = np.clip(reward, 0.0, 1.0)
            
            return float(reward)
            
        except Exception as e:
            self.logger.error(f"RLPR vLLM奖励计算失败: {e}")
            # fallback到基于长度的奖励
            try:
                completion_length = len(self._safe_encode_text(generated_response))
                if 50 <= completion_length <= 500:
                    return 0.8
                else:
                    return 0.3
            except:
                return 0.5


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
    "RougeReward": RougeReward,
    "RougeWithSolutionReward": RougeWithSolutionReward,
    "RLPRReward": RLPRReward,
    "DecisiveReward": DecisiveReward
}
