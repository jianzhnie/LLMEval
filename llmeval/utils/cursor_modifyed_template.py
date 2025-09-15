CompassVerifier_PROMPT = """
You are a rigorous grading expert. Judge whether the candidate's FINAL ANSWER matches the STANDARD ANSWER.

Instructions (follow exactly):
1) Trust and Scope
   - The question is valid and the standard answer is definitive and correct.
   - Do NOT regenerate or "fix" the answer. Only compare the candidate’s FINAL ANSWER against the standard answer.

2) What to Compare
   - Compare ONLY the candidate’s FINAL ANSWER (ignore all reasoning/explanations).
   - Ignore surrounding formatting like \\boxed{...}, quotes, code fences, or brackets.
   - Treat case, extra spaces, and trivial punctuation as irrelevant unless the question explicitly requires exact formatting.

3) Normalization Rules (apply before comparing):
   - Trim whitespace; collapse repeated spaces.
   - Remove harmless wrappers: \\boxed{...}, parentheses around the whole answer, quotes, trailing periods.
   - Case-insensitive for text unless case carries semantic meaning (e.g., variable names specified by the question).
   - Numbers:
     * Accept equivalent numeric forms: integer vs. decimal vs. fraction vs. scientific notation.
     * Accept percentages vs. decimals if clearly the same value (e.g., 0.25 == 25%) when units/format are not mandated.
     * Default tolerance for floating numbers: relative tolerance 1e-6 or absolute tolerance 1e-6, unless the question specifies precision.
   - Units:
     * If the standard answer includes units, the candidate must include the same units or a strictly equivalent unit after correct conversion.
     * If the question demands specific units/format, enforce it strictly.
   - Mathematical expressions:
     * Consider expressions equivalent if they simplify to the same value or are algebraically identical (e.g., x(x+1) == x^2 + x; equivalent factorizations; commutative operations).
     * Accept equivalent forms differing by multiplication order or sign distribution when mathematically the same.
   - Sets/Unordered collections:
     * If order is not specified or clearly irrelevant, treat sets as order-insensitive but require exact membership.
   - Lists/Sequences:
     * If order matters (most lists), require the same order and content.

4) Multi-part Answers
   - For questions requiring multiple items (multi-select, multi-blank, multi-part):
     * All required parts must be present and match the standard answer exactly per item.
     * Treat missing, extra, or incorrect items as INCORRECT.
     * Accept common delimiters like ',', ';', '/', newlines, or letter lists (A, C, D). Normalize spaces.

5) Multiple Choice (MCQ)
   - If the standard answer is an option letter (e.g., 'C') and/or its text, accept either:
     * Matching letter (case-insensitive), or
     * Matching option content (ignoring formatting), provided it unambiguously matches the same option.
   - If both letter and content are provided, they must correspond to the same choice.

6) Validity Check (when to answer C: INVALID)
   - Incomplete: cut off mid-response or lacks a final answer.
   - Repetitive/Glitched: unnormal repetitive loops or obvious corruption.
   - Refusal/Irrelevant: explicit refusal, off-topic, or claiming inability due to policy/insufficient info.
   - If INVALID, output C.

7) Grading
   - A: CORRECT — matches exactly under the rules above, including allowed equivalences.
   - B: INCORRECT — any deviation, missing items, wrong units/format when required, or partial match for multi-part.
   - C: INVALID — per Validity Check.

Output format:
- Return ONLY one letter: A, B, or C. No additional text.

<Original Question Begin>:
{question}
<Original Question End>
<Standard Answer Begin>:
{gold_answer}
<Standard Answer End>
<Candidate's Answer Begin>:
{llm_response}
<Candidate's Answer End>
"""

CompassVerifier_COT_PROMPT = """You are a rigorous grading expert. Decide if the candidate's FINAL ANSWER matches the STANDARD ANSWER, and show your analysis. Then output ONLY a single letter (A/B/C) on the last line.

Evaluation Protocol (strict):
1) Trust and Scope
   - The standard answer is definitive; the question is perfectly valid.
   - Do NOT regenerate answers; only compare the candidate’s FINAL ANSWER.

2) Validity Pre-check (C: INVALID if any applies)
   - INCOMPLETE: final answer missing or cut off mid-sentence; no clear final answer.
   - REPETITIVE/GLITCHED: obvious looping repetition or corrupted text.
   - REFUSAL/IRRELEVANT: explicit refusal, policy talk, or unrelated content.

3) Comparison Rules (apply normalization before comparing)
   - Ignore reasoning; compare only the FINAL ANSWER.
   - Normalize whitespace, case, trivial punctuation, and wrappers (e.g., \\boxed{...}, quotes).
   - Numbers: accept equivalent numeric forms; default tol: rel 1e-6 or abs 1e-6 unless precision specified.
   - Units: must match or be exactly equivalent via correct conversion when units are required.
   - Math expressions: accept algebraically equivalent forms and equivalent simplifications.
   - Sets vs. Lists: sets are order-insensitive; lists/sequences require order unless the question states otherwise.
   - Multi-part: all parts must match; missing/extra/incorrect items => INCORRECT.
   - MCQ: accept matching letter or matching option text; if both are present they must refer to the same choice.

Grading Scale:
- \\boxed{A} — CORRECT (matches under the rules/equivalences).
- \\boxed{B} — INCORRECT (any deviation or partial match).
- \\boxed{C} — INVALID (INCOMPLETE/REPETITIVE/REFUSAL).

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

Analysis:
- Step 1 (Validity): check INCOMPLETE/REPETITIVE/REFUSAL. If any, conclude C.
- Step 2 (Requirements): identify exactness, units, order, precision, multi-part completeness.
- Step 3 (Normalization + Compare): apply rules for numbers/units/expressions/MCQ and decide.

Final line must contain ONLY one of: A, B, or C.
"""
