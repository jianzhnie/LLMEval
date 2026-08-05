from __future__ import annotations

CompassVerifier_PROMPT = """Please as a grading expert, judge whether the final answers given by the candidates
below are consistent with the standard answers, that is, whether the candidates answered correctly.

Here are some evaluation criteria:
1. Please refer to the given standard answer. You don't need to re-generate the answer to the question
   because the standard answer has been given. You only need to judge whether the candidate's answer is
   consistent with the standard answer according to the form of the question. THE STANDARD ANSWER IS
   ALWAYS CORRECT AND THE QUESTION IS PERFECTLY VALID. NEVER QUESTION THEM.

2. ONLY compare the FINAL ANSWER - COMPLETELY IGNORE any potential errors in the REASONING PROCESSES.

3. Some answers may be expressed in different ways, such as some answers may be a mathematical expression,
   some answers may be a textual description, as long as the meaning expressed is the same. Before making
   a judgment, please understand the question and the standard answer first, and then judge whether the
   candidate's answer is correct.

4. Some answers may consist of multiple items, such as multiple-choice questions, multiple-select questions,
   fill-in-the-blank questions, etc. Regardless of the question type, the final answer will be considered
   correct as long as it matches the standard answer, regardless of whether the reasoning process is correct.
   For multiple-select questions and multi-blank fill-in-the-blank questions, all corresponding options or
   blanks must be answered correctly and match the standard answer exactly to be deemed correct.

5. If the prediction is given with \\boxed{}, please ignore the \\boxed{} and only judge whether the
   candidate's answer is consistent with the standard answer.

6. If the candidate's answer is invalid (e.g., incomplete (cut off mid-response), lots of abnormal repetitive
   content, or irrelevant to the question, saying it can't answer the question because some irresistible
   factors, like ethical issues, no enough information, etc.), select option C (INVALID).

Please judge whether the following answers are consistent with the standard answer based on the above
criteria. Grade the predicted answer of this new question as one of:
A: CORRECT
B: INCORRECT
C: INVALID

Just return the letters "A", "B", or "C", with no text around it.

<Original Question Begin>:
{question}
<Original Question End>

<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>

<Candidate's Answer Begin>:
{llm_response}
<Candidate's Answer End>

Judging the correctness of the candidate's answer:
"""

CompassVerifier_PROMPT_ZH = """请作为评分专家，判断下面候选人的最终答案是否与标准答案一致，即候选人是否回答正确。

以下是一些评估标准：
1. 请参考给定的标准答案。您不需要重新生成问题的答案，因为标准答案已经给出。您只需要根据问题的形式判断候选人的答案是否与标准答案一致。标准答案总是正确的，问题是完全有效的。永远不要质疑它们。

2. 只比较最终答案 - 完全忽略推理过程中的任何潜在错误。

3. 一些答案可能以不同的方式表达，比如一些答案可能是数学表达式，一些答案可能是文本描述，只要表达的含义相同即可。在做出判断之前，请先理解问题和标准答案，然后判断候选人的答案是否正确。

4. 一些答案可能包含多个项目，如多选题、多选问题、填空题等。无论问题类型如何，只要最终答案与标准答案匹配，就认为答案正确，无论推理过程是否正确。对于多选题和多空填空题，所有对应的选项或空白都必须正确回答并与标准答案完全匹配才能被视为正确。

5. 如果预测结果用\\boxed{}给出，请忽略\\boxed{}，只判断候选人的答案是否与标准答案一致。

6. 如果候选人的答案无效（例如，不完整（在回答中途被截断）、大量不正常的重复内容，或与问题无关，说由于一些不可抗拒的因素无法回答问题，如伦理问题、信息不足等），选择选项C（无效）。

请根据上述标准判断以下答案是否与标准答案一致。将此新问题的预测答案评为以下之一：
A: 正确
B: 不正确
C: 无效

只返回字母"A"、"B"或"C"，周围不要有任何文字。

<原始问题开始>：
{question}
<原始问题结束>

<标准答案开始>：
{gold_answer}
<标准答案结束>

<候选人答案开始>：
{llm_response}
<候选人答案结束>

判断候选人答案的正确性：
"""

CompassVerifier_COT_PROMPT = """As a grading expert, your task is to determine whether the candidate's final
answer matches the provided standard answer. Follow these evaluation guidelines precisely:

Evaluation Protocol:
1. Reference Standard:
   - The standard answer is definitive and always correct
   - The question is perfectly valid - never question them
   - Do not regenerate answers; only compare with the given standard

2. Comparison Method:
   - Carefully analyze the question's requirements and the standard answer's structure
     * Determine whether the question expects exact matching of the entire standard answer or allows
       partial matching of its components.
     * This determination must be made based on the question's phrasing and the nature of the
       standard answer.
   - Compare ONLY the candidate's final answer (ignore all reasoning/explanation errors)
   - Disregard any differences in formatting or presentation style
   - For mathematical expressions: calculate step by step whether the two formulas are equivalent
   - For multiple-choice questions: compare only the final choice and corresponding option content

3. Multi-part Answers:
   - For questions requiring multiple responses (e.g., multi-select):
   - All parts must match the standard answer exactly.
   - Compare each sub-answer step by step. Partial matches are considered incorrect.

4. Validity Check:
   - Reject answers that are:
     * Incomplete (cut off mid-sentence in the final sentence, lacking a complete response) →
       Label as INCOMPLETE
     * Repetitive (repetition of words or phrases in a loop) → Label as REPETITIVE
     * Explicit refusals (e.g., directly return "I cannot answer/provide/access ...") →
       Label as REFUSAL
   - For invalid answers, specify the type in the judgment (e.g., \\boxed{C} - INCOMPLETE).

Grading Scale:
\\boxed{A} - CORRECT:
   - Answer matches standard exactly (including equivalent expressions)
   - For numerical answers: consider as equivalent if values match when rounded appropriately
   - Semantically equivalent responses

\\boxed{B} - INCORRECT:
   - Any deviation from standard answer
   - Partial matches for multi-part questions

\\boxed{C} - INCOMPLETE/REPETITIVE/REFUSAL:
   - Fails validity criteria above (must specify: INCOMPLETE/REPETITIVE/REFUSAL)

Execution Steps and Output Formats:

Analysis step by step: [
Thoroughly evaluate the candidate's answer including:
(1) First check if the answer is INCOMPLETE (cut off mid-sentence), REPETITIVE (looping repetition),
    or a REFUSAL (explicit denial) - if so, immediately classify as \\boxed{C} with the
    corresponding type.
(2) Analyze the question's core requirements and the standard answer's structure, for example:
- Strict requirements: Identify mandatory constraints (e.g., simplification, answer order,
  multi-part completeness)
- Tolerant allowances: Ignore non-critical deviations (e.g., missing option labels in MCQs,
  equivalent but unformatted expressions)
- Required answer type, precision level, etc.
(3) Perform a detailed comparison between the candidate's final answer and the standard answer,
    for example:
- Content equivalence
- Permitted variations in numerical precision
- Allowed expression formats]
Final Judgment: \\boxed{A/B/C} - <CORRECT/INCORRECT/INCOMPLETE/REPETITIVE/REFUSAL>

Here is your task.
<Original Question Begin>
{question}
<Original Question End>

<Standard Answer Begin>
{gold_answer}
<Standard Answer End>

<Candidate's Answer Begin>
{llm_response}
<Candidate's Answer End>

Analysis step by step and Final Judgment:
"""

CompassVerifier_COT_PROMPT_ZH = """作为评分专家，您的任务是确定候选人的最终答案是否与提供的标准答案匹配。请严格按照以下评估指南执行：

评估协议：
1. 参考标准：
   - 标准答案是确定的且总是正确的
   - 问题是完全有效的 - 永远不要质疑它们
   - 不要重新生成答案；只与给定的标准进行比较

2. 比较方法：
   - 仔细分析问题的要求和标准答案的结构
     * 确定问题是期望与整个标准答案完全匹配还是允许与其组成部分部分匹配。
     * 这一确定必须基于问题的措辞和标准答案的性质。
   - 只比较候选人的最终答案（忽略所有推理/解释错误）
   - 忽略格式或呈现风格的任何差异
   - 对于数学表达式：逐步计算两个公式是否等价
   - 对于选择题：只比较最终选择和相应的选项内容

3. 多部分答案：
   - 对于需要多个回答的问题（例如，多选）：
   - 所有部分必须与标准答案完全匹配。
   - 逐步比较每个子答案。部分匹配被视为不正确。

4. 有效性检查：
   - 拒绝以下答案：
     * 不完整（在最终句子中中途截断，缺乏完整回答）→ 标记为不完整
     * 重复（单词或短语循环重复）→ 标记为重复
     * 明确拒绝（例如，直接返回"我无法回答/提供/访问..."）→ 标记为拒绝
   - 对于无效答案，在判断中指定类型（例如，\\boxed{C} - 不完整）。

评分标准：
\\boxed{A} - 正确：
   - 答案与标准完全匹配（包括等价表达式）
   - 对于数值答案：如果值在适当舍入时匹配，则视为等价
   - 语义等价的回答

\\boxed{B} - 不正确：
   - 与标准答案的任何偏差
   - 多部分问题的部分匹配

\\boxed{C} - 不完整/重复/拒绝：
   - 不符合上述有效性标准（必须指定：不完整/重复/拒绝）

执行步骤和输出格式：

逐步分析：[
彻底评估候选人的答案，包括：
(1) 首先检查答案是否不完整（中途截断）、重复（循环重复）或拒绝（明确否认）- 如果是，立即分类为\\boxed{C}并指定相应类型。
(2) 分析问题的核心要求和标准答案的结构，例如：
- 严格要求：识别强制性约束（例如，简化、答案顺序、多部分完整性）
- 宽容允许：忽略非关键偏差（例如，选择题中缺少选项标签、等价但未格式化的表达式）
- 所需答案类型、精度级别等。
(3) 在候选人的最终答案和标准答案之间进行详细比较，例如：
- 内容等价性
- 数值精度的允许变化
- 允许的表达式格式]
最终判断：\\boxed{A/B/C} - <正确/不正确/不完整/重复/拒绝>

这是您的任务。
<原始问题开始>
{question}
<原始问题结束>

<标准答案开始>
{gold_answer}
<标准答案结束>

<候选人答案开始>
{llm_response}
<候选人答案结束>

逐步分析和最终判断：
"""


VERIFY_PROMPT_FACTORY: dict[str, str] = {
    "compassverify_prompt": CompassVerifier_PROMPT,
    "compassverify_prompt_zh": CompassVerifier_PROMPT_ZH,
    "compassverify_cot_prompt": CompassVerifier_COT_PROMPT,
    "compassverify_cot_prompt_zh": CompassVerifier_COT_PROMPT_ZH,
}
