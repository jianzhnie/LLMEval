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

FDD_PROMPT_CURSOR = """You are a mathematical problem evaluation expert. Given a math problem, standard answer, and candidate answer,
perform the following tasks with precision and consistency.

## CORE TASKS
1. **Problem Cleaning**: Remove unnecessary text, formatting artifacts, or embedded answers while preserving
   the mathematical essence and solvability of the problem.

2. **Problem Validation**: Assess problem quality for:
   - Completeness (sufficient information provided)
   - Logical consistency (no contradictions)
   - Solvability (mathematically tractable)
   - Single focus (one clear question, not multiple subproblems)

3. **Answer Comparison**: Evaluate candidate answer against standard answer using flexible mathematical
   equivalence criteria.

## EVALUATION CRITERIA

### Problem Quality Assessment:
- **D (UNREASONABLE)**: Missing critical information, logical contradictions, or unsolvable problems
- **C (MULTIPLE_SUBPROBLEMS)**: Contains multiple distinct questions or subproblems requiring separate solutions

### Answer Comparison Logic:
- **A (CORRECT)**: Mathematically equivalent to standard answer, considering:
  * Different but equivalent expressions (e.g., 1/2 vs 0.5)
  * Missing units when standard includes them
  * Option content matching option labels
  * Minor decimal precision differences (last 1-2 digits)
  * Different but equivalent forms (factored vs expanded)

- **B (INCORRECT)**: Substantially different from standard answer

### Answer Extraction:
- Focus exclusively on content within \\boxed{} tags
- Ignore formatting, explanations, or work shown outside the box

## OUTPUT FORMAT
1. Clean the problem and place result between:
   <Cleaned Problem Begin>
   [cleaned problem text]
   <Cleaned Problem End>

2. Provide evaluation result in \\boxed{[A|B|C|D]} with no additional explanation

## IMPORTANT CONSTRAINTS
- Do NOT attempt to solve the mathematical problem
- Do NOT provide explanations for your evaluation
- Do NOT apologize for or correct previous responses
- Maintain consistency in evaluation standards

<Original Question Begin>:
{question}
<Original Question End>

<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>

<Candidate's Answer Begin>:
{llm_response}
<Candidate's Answer End>

Execute the evaluation following the specified format and constraints."""

FDD_PROMPT = """As a question-setting expert, given a math problem, a standard answer, and a candidate answer,
you need to accomplish the following tasks:

Task Requirements:
1. Clean the problem description: Remove unnecessary sentences or language fragments from the problem
   description without affecting the understanding and solving of the problem. If the problem description
   itself contains the answer, remove the answer.

2. Do not try to solve the problem, do not try to solve the problem, do not try to solve the problem.
   Important things are said three times.

3. Judge whether the math problem is reasonable, primarily to rule out issues such as missing information,
   logical confusion, contradiction or incompleteness. If the problem is not reasonable, return D directly.

4. If the problem itself contains multiple subproblems or multiple questions, return C directly.

5. For the candidate answer, only consider the content inside \\boxed{}, ignoring any other potential errors.

6. Flexibly compare the standard answer and the candidate answer.

Here are some specific comparison cases:
a. If the standard answer is an option, but the candidate answer provides the content of the option,
   and they match, return A directly.

b. If the standard answer includes units (dimensions), but the candidate answer does not, they can be
   considered matching, return A directly.

c. If the mathematical expressions of the standard answer and the candidate answer are different,
   consider whether they are mathematically equivalent. If they are equivalent, return A directly.

d. For answers involving decimals, if most digits are the same and only the last one or two digits differ,
   you may considerably return A.

e. For other cases where the standard answer and the candidate answer do not match, return B directly.

Grading Scale:
A: CORRECT - Answer matches standard exactly or is mathematically equivalent
B: INCORRECT - Answer does not match the standard answer
C: MULTIPLE_SUBPROBLEMS - Problem contains multiple subproblems or questions
D: UNREASONABLE - Problem has missing information, logical confusion, or contradictions

Here is your task. Simply clean the problem and return one of A, B, C, or D as required.
Don't apologize or correct yourself if there was a mistake; we are just trying to evaluate the problem.

<Original Question Begin>:
{question}
<Original Question End>

<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>

<Candidate's Answer Begin>:
{llm_response}
<Candidate's Answer End>

Please clean the problem description and place the cleaned description within
<Cleaned Problem Begin>...<Cleaned Problem End>, do not try to modify the question.
Then, simply return one of A, B, C or D as required and put it in the \\boxed{} without any
explanation or description. Remember do not try to solve the question. Just clean the problem
and return one specific option within the \\boxed{}.
"""

FDD_PROMPT_ZH = """作为问题设置专家，给定一个数学问题、标准答案和候选人答案，您需要完成以下任务：

任务要求：
1. 清理问题描述：从问题描述中删除不必要的句子或语言片段，而不影响问题的理解和解决；如果问题描述本身包含答案，请删除答案。

2. 不要尝试解决问题，不要尝试解决问题，不要尝试解决问题。重要的事情说三遍。

3. 判断数学问题是否合理，主要是排除缺失信息、逻辑混乱、矛盾或不完整等问题。如果问题不合理，直接返回D。

4. 如果问题本身包含多个子问题或多个问题，直接返回C。

5. 对于候选人答案，只考虑\\boxed{}内的内容，忽略任何其他潜在错误。

6. 灵活比较标准答案和候选人答案。

以下是一些具体的比较情况：
a. 如果标准答案是选项，但候选人答案提供了选项的内容，并且它们匹配，直接返回A。

b. 如果标准答案包含单位（维度），但候选人答案没有，它们可以被视为匹配，直接返回A。

c. 如果标准答案和候选人答案的数学表达式不同，考虑它们在数学上是否等价。如果等价，直接返回A。

d. 对于涉及小数的答案，如果大多数数字相同，只有最后一位或两位数字不同，您可以考虑返回A。

e. 对于标准答案和候选人答案不匹配的其他情况，直接返回B。

评分标准：
A: 正确 - 答案与标准完全匹配或在数学上等价
B: 不正确 - 答案与标准答案不匹配
C: 多个子问题 - 问题包含多个子问题或问题
D: 不合理 - 问题有缺失信息、逻辑混乱或矛盾

这是您的任务。只需清理问题并根据需要返回A、B、C或D之一。如果出现错误，不要道歉或纠正自己；我们只是在尝试评估问题。

<原始问题开始>：
{question}
<原始问题结束>

<标准答案开始>：
{gold_answer}
<标准答案结束>

<候选人答案开始>：
{llm_response}
<候选人答案结束>

请清理问题描述并将清理后的描述放在<清理后问题开始>...<清理后问题结束>内，不要尝试修改问题。
然后，根据需要简单地返回A、B、C或D之一，并将其放在\\boxed{}中，不要任何解释或描述。
记住不要尝试解决问题。只需清理问题并在\\boxed{}内返回一个特定选项。
"""

FDD_Verify_PROMPT = """As a math scoring expert, given a standard answer and a candidate answer, you need to
compare whether the standard answer and the candidate answer are consistent. If they are consistent,
return 1; if not, return 0. Remember the returned value should always be put in the \\boxed{}.

Here are the evaluation criteria:
1. For the candidate answer, only consider the content inside \\boxed{}, ignoring any other text or
   error. If no \\boxed{} found, return 0 directly.

2. If the standard answer and the candidate answer are different but mathematically equivalent,
   return 1.

3. For answers involving decimals, if most digits are the same and only the last one or two digits
   differ, you may considerably return 1.

4. For all other cases where the standard answer and the candidate answer do not match, return 0.

Grading Scale:
1: CORRECT - Answer matches standard exactly or is mathematically equivalent
0: INCORRECT - Answer does not match the standard answer

Here is your task. Simply compare the answers and return 0 or 1 as required.
Don't apologize or correct yourself if there was a mistake; we are just trying to evaluate the answer.

<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>

<Candidate Answer Begin>:
{llm_response}
<Candidate Answer End>

Please put your return value (0 or 1) as required above in the \\boxed{} without any explanation
or description.
"""

FDD_Verify_PROMPT_ZH = """作为数学评分专家，给定标准答案和候选人答案，您需要比较标准答案和候选人答案是否一致。
如果一致，返回1；如果不一致，返回0。记住返回值应始终放在\\boxed{}中。

以下是评估标准：
1. 对于候选人答案，只考虑\\boxed{}内的内容，忽略任何其他文本或错误。如果没有找到\\boxed{}，直接返回0。

2. 如果标准答案和候选人答案不同但在数学上等价，返回1。

3. 对于涉及小数的答案，如果大多数数字相同，只有最后一位或两位数字不同，您可以考虑返回1。

4. 对于标准答案和候选人答案不匹配的所有其他情况，返回0。

评分标准：
1: 正确 - 答案与标准完全匹配或在数学上等价
0: 不正确 - 答案与标准答案不匹配

这是您的任务。只需比较答案并根据需要返回0或1。如果出现错误，不要道歉或纠正自己；我们只是在尝试评估答案。

<标准答案开始>：
{gold_answer}
<标准答案结束>

<候选人答案开始>：
{llm_response}
<候选人答案结束>

请将您的返回值（0或1）按要求放在\\boxed{}中，不要任何解释或描述。
"""

VERIFY_PROMPT_FACTORY = {
    'compassverify_prompt': CompassVerifier_PROMPT,
    'compassverify_prompt_zh': CompassVerifier_PROMPT_ZH,
    'compassverify_cot_prompt': CompassVerifier_COT_PROMPT,
    'compassverify_cot_prompt_zh': CompassVerifier_COT_PROMPT_ZH,
    'fdd_prompt_cursor': FDD_PROMPT_CURSOR,
    'fdd_prompt': FDD_PROMPT,
    'fdd_verify_prompt': FDD_Verify_PROMPT,
    'fdd_verify_prompt_zh': FDD_Verify_PROMPT_ZH
}
