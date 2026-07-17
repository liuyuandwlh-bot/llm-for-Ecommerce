"""
Answer Engine for Financial RAG

Generates answers with citations and calculator support.
Based on recommended plan:
- Cite page numbers and evidence spans
- Calculator tool for numerical reasoning
- Confidence and limitations reporting
"""

import json
import re
from dataclasses import dataclass, field
from typing import Any


@dataclass
class Claim:
    """A factual claim in the answer."""

    text: str
    doc_id: str
    page: int
    evidence: str
    verified: bool = True


@dataclass
class Calculation:
    """A calculation performed."""

    formula: str
    operands: list[Any]
    result: Any
    unit: str | None = None


@dataclass
class Answer:
    """A complete answer with citations."""

    answer: str
    claims: list[Claim]
    calculations: list[Calculation] = field(default_factory=list)
    confidence: str = "high"  # high, medium, low
    limitations: list[str] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class Calculator:
    """Financial calculator for numerical reasoning."""

    def extract_numbers(self, text: str) -> list[float]:
        """Extract numbers from text."""
        # Match numbers with optional units
        pattern = r"([\d,]+\.?\d*)\s*(亿|万|千|%)?"
        matches = re.findall(pattern, text)

        numbers = []
        for num_str, unit in matches:
            try:
                num = float(num_str.replace(",", ""))

                # Apply unit multipliers
                if unit == "亿":
                    num *= 1e8
                elif unit == "万":
                    num *= 1e4
                elif unit == "千":
                    num *= 1e3
                elif unit == "%":
                    num /= 100

                numbers.append(num)
            except ValueError:
                continue

        return numbers

    def parse_formula(self, formula: str) -> str | None:
        """Parse a formula like 'revenue * margin'."""
        # Simple formula parsing
        formula = formula.lower().strip()

        # Define common financial terms

        return formula

    def calculate(
        self,
        formula: str,
        context: dict[str, Any],
        tolerance: float = 0.01,
    ) -> Calculation | None:
        """Perform a calculation."""
        # Simple calculation support
        # In practice, would use a proper parser or calculator tool

        operands = []
        result = None

        # Try to extract operands from context
        for key, value in context.items():
            if isinstance(value, (int, float)):
                operands.append((key, value))

        return Calculation(
            formula=formula,
            operands=[v for _, v in operands],
            result=result,
        )


class AnswerEngine:
    """
    Answer generation engine for financial RAG.

    Features:
    - Grounded generation with retrieved context
    - Citation extraction and verification
    - Numerical calculation
    - Confidence assessment
    - Limitation reporting
    """

    def __init__(
        self,
        model_path: str = "Qwen/Qwen3-4B-Instruct-2507",
        device: str = "auto",
    ):
        self.model_path = model_path
        self.device = device
        self.calculator = Calculator()

    def generate(
        self,
        query: str,
        retrieved_chunks: list[dict],
        system_prompt: str | None = None,
    ) -> Answer:
        """
        Generate an answer with citations.

        Args:
            query: User question
            retrieved_chunks: Retrieved context chunks
            system_prompt: Optional system prompt

        Returns:
            Answer with claims, calculations, and metadata
        """
        # Build context from chunks
        context = self._build_context(retrieved_chunks)

        # Build prompt
        prompt = self._build_prompt(query, context, system_prompt)

        # Generate answer (simplified - would use actual model)
        answer_text = self._generate_with_model(prompt)

        # Extract claims and citations
        claims = self._extract_claims(query, retrieved_chunks, answer_text)

        # Check for calculations needed
        calculations = self._check_calculations(query, retrieved_chunks)

        # Assess confidence
        confidence = self._assess_confidence(query, retrieved_chunks, claims)

        # Identify limitations
        limitations = self._identify_limitations(query, retrieved_chunks, claims)

        return Answer(
            answer=answer_text,
            claims=claims,
            calculations=calculations,
            confidence=confidence,
            limitations=limitations,
            metadata={
                "num_chunks": len(retrieved_chunks),
                "query": query,
            },
        )

    def _build_context(self, chunks: list[dict]) -> str:
        """Build context string from chunks."""
        context_parts = []

        for i, chunk in enumerate(chunks, 1):
            page_info = f"(页{chunk.get('page_start', '?')}-{chunk.get('page_end', '?')})"
            context_parts.append(
                f"[文档{i} {chunk.get('doc_id', '')} {page_info}]\n{chunk.get('text', '')}"
            )

        return "\n\n".join(context_parts)

    def _build_prompt(
        self,
        query: str,
        context: str,
        system_prompt: str | None = None,
    ) -> str:
        """Build generation prompt."""
        if system_prompt is None:
            system_prompt = """你是一个专业的金融分析助手，负责根据提供的信息回答用户问题。

重要原则：
1. 只基于提供的文档内容回答，不要编造信息
2. 所有事实性陈述必须标注来源，包括文档ID和页码
3. 如果信息不足以回答问题，明确说明"信息不足"
4. 计算题请列出计算过程
5. 区分确定信息和推测信息
6. 如涉及投资建议，添加免责声明"""

        prompt = f"""{system_prompt}

[用户问题]
{query}

[参考文档]
{context}

[回答要求]
1. 直接回答问题，不要重复问题
2. 每个事实陈述后用【来源：文档ID, 页码】标注
3. 如需计算，展示计算过程
4. 如信息不足，明确说明
5. 添加必要的免责声明（如涉及投资建议）

[回答]
"""
        return prompt

    def _generate_with_model(self, prompt: str) -> str:
        """Generate answer using model (placeholder)."""
        # In practice, would call actual model
        return "[此处为模型生成的答案]"

    def _extract_claims(
        self,
        query: str,
        chunks: list[dict],
        answer: str,
    ) -> list[Claim]:
        """Extract claims and verify against sources."""
        claims = []

        # Simple claim extraction (in practice, would use NER/extraction model)
        # Look for numerical claims
        number_pattern = r"([\d,]+\.?\d*)\s*(亿|万|%|元)?"
        matches = re.findall(number_pattern, answer)

        for chunk in chunks:
            for num_str, unit in matches:
                if num_str in chunk.get("text", ""):
                    claims.append(
                        Claim(
                            text=f"数值：{num_str}{unit}",
                            doc_id=chunk.get("doc_id", ""),
                            page=chunk.get("page_start", 0),
                            evidence=num_str,
                            verified=True,
                        )
                    )

        return claims

    def _check_calculations(
        self,
        query: str,
        chunks: list[dict],
    ) -> list[Calculation]:
        """Check if query requires calculations."""
        calc_keywords = ["增长", "下降", "占比", "同比", "环比", "平均", "总计", "增加", "减少"]

        if not any(kw in query for kw in calc_keywords):
            return []

        # Extract relevant data from chunks
        context = {}
        for chunk in chunks:
            numbers = self.calculator.extract_numbers(chunk.get("text", ""))
            if numbers:
                context[f"doc_{chunk.get('doc_id', '')}"] = numbers[0]

        if context:
            return [self.calculator.calculate("extracted", context)]

        return []

    def _assess_confidence(
        self,
        query: str,
        chunks: list[dict],
        claims: list[Claim],
    ) -> str:
        """Assess answer confidence."""
        if not chunks:
            return "low"

        if len(chunks) >= 3 and all(c.verified for c in claims):
            return "high"

        if len(chunks) >= 1:
            return "medium"

        return "low"

    def _identify_limitations(
        self,
        query: str,
        chunks: list[dict],
        claims: list[Claim],
    ) -> list[str]:
        """Identify answer limitations."""
        limitations = []

        if not chunks:
            limitations.append("没有找到相关文档")
            return limitations

        # Check for chart-only information
        if any("图" in chunk.get("text", "") or "表" in chunk.get("text", "") for chunk in chunks):
            limitations.append("部分信息可能仅存在于图表中，需要视觉理解能力")

        # Check document recency
        # (would compare with query timestamp)

        # Check for incomplete evidence
        unverified = [c for c in claims if not c.verified]
        if unverified:
            limitations.append(f"{len(unverified)} 个声明未经完全验证")

        return limitations

    def to_json(self, answer: Answer) -> dict:
        """Convert answer to JSON-serializable dict."""
        return {
            "answer": answer.answer,
            "claims": [
                {
                    "text": c.text,
                    "doc_id": c.doc_id,
                    "page": c.page,
                    "evidence": c.evidence,
                    "verified": c.verified,
                }
                for c in answer.claims
            ],
            "calculations": [
                {
                    "formula": calc.formula,
                    "operands": calc.operands,
                    "result": calc.result,
                    "unit": calc.unit,
                }
                for calc in answer.calculations
            ],
            "confidence": answer.confidence,
            "limitations": answer.limitations,
            "metadata": answer.metadata,
        }


def demo_answer_engine():
    """Demo the answer engine."""
    engine = AnswerEngine()

    query = "某公司2025年毛利率是多少？"
    chunks = [
        {
            "chunk_id": "chunk1",
            "doc_id": "annual_2025",
            "page_start": 45,
            "page_end": 46,
            "text": "2025年度，公司实现营业收入1,000亿元，毛利率为35%。",
            "score": 0.9,
        }
    ]

    answer = engine.generate(query, chunks)
    print(json.dumps(engine.to_json(answer), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    demo_answer_engine()
