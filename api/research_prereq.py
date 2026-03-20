#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import textwrap
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Type, TypeVar
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import time

import instructor
import requests
from bs4 import BeautifulSoup
from openai import OpenAI
from pydantic import BaseModel, Field

from .llm_logging import append_log


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gemini-2.5-flash")
DEFAULT_BASE_URL = os.getenv(
    "OPENAI_BASE_URL", "https://generativelanguage.googleapis.com/v1beta/openai/"
)
DEFAULT_OUTPUT = "reports.md"
USER_AGENT = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36"
)


# ============================================================================
# Pydantic Models for Instructor
# ============================================================================


class PlannerResponse(BaseModel):
    problem_statement: str = Field(description="清晰的问题陈述，包含研究背景和动机")
    research_gap: str = Field(description="当前研究空白或不足")
    innovation_points: List[str] = Field(description="本研究的创新点")
    clarifying_questions: List[str] = Field(description="需要明确的关键问题")
    initial_research_ideas: List[str] = Field(description="初步研究思路描述")
    candidate_directions: List[str] = Field(description="候选方向：技术路线+预期效果")
    theoretical_foundation: List[str] = Field(description="相关理论基础")
    search_queries: List[str] = Field(description="用于查阅网页文献的检索关键词")
    assumptions: List[str] = Field(description="需验证的前提条件与假设")
    feasibility_concerns: List[str] = Field(description="可行性顾虑")


class DataRequirement(BaseModel):
    data_name: str
    data_type: str
    scale_requirement: str
    why_needed: str
    key_fields: List[str]
    quality_requirements: str
    possible_sources: List[str]
    preprocessing_steps: List[str]
    acquisition_difficulty: str
    estimated_time: str


class ExperimentDesign(BaseModel):
    baseline_methods: List[str]
    evaluation_metrics: List[str]
    control_variables: List[str]
    expected_challenges: List[str]


class LiteratureSupport(BaseModel):
    topic: str
    key_findings: str
    relevance: str


class ResourceRequirements(BaseModel):
    computational: str
    human: str
    financial: str
    time: str


class SynthesizerResponse(BaseModel):
    round_summary: str = Field(description="第二轮分析总结")
    refined_ideas: List[str] = Field(description="细化后的研究思路")
    refined_directions: List[str] = Field(description="验证后的方向")
    data_requirements: List[DataRequirement] = Field(description="数据需求详情")
    experiment_design: ExperimentDesign = Field(description="初步实验设计")
    literature_support: List[LiteratureSupport] = Field(description="文献证据支持")
    risks_or_gaps: List[str] = Field(description="风险/空白应对策略")
    resource_requirements: ResourceRequirements = Field(description="资源需求估算")


class FinalResearchIdea(BaseModel):
    idea: str
    rationale: str
    validation_method: str
    expected_contribution: str


class FinalResearchDirection(BaseModel):
    direction: str
    priority: str
    feasibility_score: str
    innovation_score: str
    technical_approach: str
    success_criteria: str
    fallback_plan: str


class FinalDataNeeded(BaseModel):
    data_name: str
    data_type: str
    scale_requirement: str
    purpose: str
    key_fields: List[str]
    quality_requirements: str
    possible_sources: List[str]
    preprocessing_steps: List[str]
    acquisition_difficulty: str
    estimated_cost: str
    estimated_time: str
    alternative_sources: List[str]
    notes: str


class Phase(BaseModel):
    phase_name: str
    objectives: List[str]
    tasks: List[str]
    deliverables: List[str]
    duration: str


class FinalExperimentPlan(BaseModel):
    phases: List[Phase]
    baseline_methods: List[str]
    evaluation_metrics: List[str]
    validation_strategy: str


class RiskMitigation(BaseModel):
    risk: str
    impact: str
    probability: str
    mitigation_strategy: str
    contingency_plan: str


class Milestone(BaseModel):
    milestone: str
    target_date: str
    success_criteria: List[str]
    dependencies: List[str]


class ReadinessChecklist(BaseModel):
    item: str
    status: str
    notes: str


class AcademicReport(BaseModel):
    title: str
    abstract: str
    keywords: List[str]
    introduction: str
    related_work: str
    methodology: str
    experiment_design: str
    expected_outcomes: str
    discussion: str
    conclusion: str
    references: List[str]


class CriticResponse(BaseModel):
    final_research_ideas: List[FinalResearchIdea]
    final_research_directions: List[FinalResearchDirection]
    final_data_needed: List[FinalDataNeeded]
    experiment_plan: FinalExperimentPlan
    risk_mitigation: List[RiskMitigation]
    milestones: List[Milestone]
    discussion_highlights: List[str]
    final_assumptions: List[str]
    source_usage_notes: List[str]
    readiness_checklist: List[ReadinessChecklist]
    academic_report: AcademicReport


# ============================================================================


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str
    query: str


T = TypeVar("T", bound=BaseModel)


class ResearchPrerequisiteAgent:
    def __init__(
        self,
        api_key: str,
        model: str = DEFAULT_MODEL,
        base_url: str = DEFAULT_BASE_URL,
        timeout: int = 120,
        log_path: Optional[str] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> None:
        self.api_key = api_key
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.log_path = log_path
        self.progress_callback = progress_callback

        self.client = instructor.from_openai(
            OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout),
            mode=instructor.Mode.JSON,
        )

        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": USER_AGENT,
            }
        )

    def _emit_progress(self, status: str, message: str) -> None:
        append_log(self.log_path, "agent_step", {"status": status, "message": message})
        if self.progress_callback:
            self.progress_callback({"status": status, "message": message})

    T = TypeVar("T", bound=BaseModel)

    def chat_json(
        self,
        messages: List[Dict[str, str]],
        response_model: Type[T],
        temperature: float = 0.2,
        max_retries: int = 5,
    ) -> T:
        self._emit_progress(
            "llm_request", f"向大模型发起请求，期望结构: {response_model.__name__}"
        )
        request_url = f"{self.base_url.rstrip('/')}/chat/completions"
        append_log(
            self.log_path,
            "chat_json_request",
            {
                "method": "POST",
                "url": request_url,
                "model": self.model,
                "messages": messages,
                "response_model": response_model.__name__,
            },
        )
        last_exc: Exception = RuntimeError("未知错误")
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    response_model=response_model,
                    messages=messages,  # type: ignore
                    temperature=temperature,
                )
                append_log(self.log_path, "chat_json_response", resp.model_dump())
                self._emit_progress(
                    "llm_response", f"成功收到 {response_model.__name__} 响应"
                )
                return resp
            except Exception as e:
                last_exc = e
                err_str = str(e)
                is_rate_limit = (
                    "429" in err_str
                    or "rate limit" in err_str.lower()
                    or "请求数限制" in err_str
                )
                is_timeout = (
                    "timed out" in err_str.lower() or "timeout" in err_str.lower()
                )
                if (is_rate_limit or is_timeout) and attempt < max_retries:
                    wait = 30 * attempt  # 30s, 60s, 90s, 120s
                    append_log(
                        self.log_path,
                        "chat_json_retry",
                        {"attempt": attempt, "wait_seconds": wait, "error": err_str},
                    )
                    self._emit_progress(
                        "llm_retry",
                        f"{'限流' if is_rate_limit else '超时'}，{wait}秒后重试 (第{attempt}次/{max_retries}次)",
                    )
                    time.sleep(wait)
                else:
                    append_log(self.log_path, "chat_json_error", {"error": err_str})
                    self._emit_progress("error", f"LLM 接口调用失败: {err_str}")
                    raise RuntimeError(f"LLM 接口调用失败: {err_str}") from e
        append_log(self.log_path, "chat_json_error", {"error": str(last_exc)})
        self._emit_progress("error", f"LLM 接口调用失败: {str(last_exc)}")
        raise RuntimeError(f"LLM 接口调用失败: {str(last_exc)}") from last_exc

    def search_bing(self, query: str, max_results: int = 5) -> List[SearchResult]:
        url = f"https://www.bing.com/search?q={quote_plus(query)}&count={max_results}"
        headers = {
            "User-Agent": USER_AGENT,
            "Accept-Language": "zh-CN,zh;q=0.9,en;q=0.8",
        }
        response = self.session.get(url, headers=headers, timeout=self.timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        results: List[SearchResult] = []
        for li in soup.select("li.b_algo"):
            h2 = li.select_one("h2")
            a = h2.select_one("a") if h2 else None
            if not a:
                continue
            title = normalize_text(a.get_text(" ", strip=True))
            href = a.get("href", "")
            if isinstance(href, list):
                href = str(href[0])
            else:
                href = str(href)
            href = href.strip()
            if not href.startswith("http"):
                continue
            snippet = ""
            snippet_el = li.select_one(".b_caption p")
            if snippet_el:
                snippet = normalize_text(snippet_el.get_text(" ", strip=True))
            results.append(
                SearchResult(title=title, url=href, snippet=snippet, query=query)
            )
            if len(results) >= max_results:
                break
        return results

    def search_web(self, query: str, max_results: int = 5) -> List[SearchResult]:
        url = f"https://html.duckduckgo.com/html/?q={quote_plus(query)}"
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")

        results: List[SearchResult] = []
        for anchor in soup.select("a.result__a"):
            title = normalize_text(anchor.get_text(" ", strip=True))
            href_attr = anchor.get("href")
            href = ""
            if isinstance(href_attr, list):
                href = href_attr[0]
            elif isinstance(href_attr, str):
                href = href_attr
            href = href.strip()
            if not title or not href:
                continue
            clean_url = self._clean_duckduckgo_url(href)
            if not clean_url.startswith("http"):
                continue

            snippet_node = anchor.find_parent(class_="result")
            snippet = ""
            if snippet_node and hasattr(snippet_node, "select_one"):
                snippet_el = getattr(snippet_node, "select_one")(".result__snippet")
                if snippet_el:
                    snippet = normalize_text(snippet_el.get_text(" ", strip=True))

            results.append(
                SearchResult(
                    title=title,
                    url=clean_url,
                    snippet=snippet,
                    query=query,
                )
            )
            if len(results) >= max_results:
                break
        return results

    def fetch_page_text(self, url: str, max_chars: int = 6000) -> str:
        response = self.session.get(url, timeout=self.timeout)
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if (
            "text/html" not in content_type
            and "application/xhtml+xml" not in content_type
        ):
            text = response.text[:max_chars]
            return normalize_text(text)

        soup = BeautifulSoup(response.text, "html.parser")
        for tag in soup(["script", "style", "noscript", "svg", "img"]):
            tag.decompose()
        text = normalize_text(soup.get_text(" ", strip=True))
        return text[:max_chars]

    def collect_web_evidence(
        self,
        queries: Iterable[str],
        max_queries: int = 3,
        max_results_per_query: int = 3,
        max_pages: int = 5,
    ) -> List[Dict[str, str]]:
        evidence: List[Dict[str, str]] = []
        visited: set[str] = set()

        for query in list(queries)[:max_queries]:
            self._emit_progress(
                "search_query", f"正在搜索关键词（DuckDuckGo + Bing）：{query}"
            )
            append_log(self.log_path, "web_search_query", {"query": query})

            try:
                ddg_results = self.search_web(query, max_results=max_results_per_query)
            except Exception as exc:
                self._emit_progress("search_error", f"DuckDuckGo 搜索失败：{exc}")
                append_log(
                    self.log_path,
                    "web_search_error",
                    {"source": "duckduckgo", "query": query, "error": str(exc)},
                )
                ddg_results = []

            try:
                bing_results = self.search_bing(
                    query, max_results=max_results_per_query
                )
            except Exception as exc:
                self._emit_progress("search_error", f"Bing 搜索失败：{exc}")
                append_log(
                    self.log_path,
                    "web_search_error",
                    {"source": "bing", "query": query, "error": str(exc)},
                )
                bing_results = []

            seen_urls: set[str] = set()
            results: List[SearchResult] = []
            for r in ddg_results + bing_results:
                if r.url not in seen_urls:
                    seen_urls.add(r.url)
                    results.append(r)

            self._emit_progress(
                "search_results",
                f"关键词“{query}”获得 {len(results)} 条结果（DDG+Bing）",
            )
            append_log(
                self.log_path,
                "web_search_results",
                {
                    "query": query,
                    "results": [
                        {"title": r.title, "url": r.url, "snippet": r.snippet}
                        for r in results
                    ],
                },
            )

            for result in results:
                if result.url in visited:
                    continue
                visited.add(result.url)
                try:
                    self._emit_progress("fetch_page", f"正在抓取页面：{result.title}")
                    content_text = self.fetch_page_text(result.url)
                    self._emit_progress("page_fetched", f"已抓取页面：{result.title}")
                    append_log(
                        self.log_path,
                        "web_page_fetched",
                        {
                            "query": result.query,
                            "title": result.title,
                            "url": result.url,
                            "snippet": result.snippet,
                            "content": content_text,
                        },
                    )
                except Exception as exc:
                    content_text = f"Fetch failed: {exc}"
                    self._emit_progress("fetch_error", f"页面抓取失败：{result.title}")
                    append_log(
                        self.log_path,
                        "web_page_fetch_error",
                        {"url": result.url, "error": str(exc)},
                    )

                evidence.append(
                    {
                        "query": result.query,
                        "title": result.title,
                        "url": result.url,
                        "snippet": result.snippet,
                        "content": content_text,
                    }
                )
                if len(evidence) >= max_pages:
                    return evidence
        return evidence

    def generate_report(
        self, topic: str, format_style: str = "", report_language: str = "zh-CN"
    ) -> Iterable[Dict[str, Any]]:
        style_instruction = ""
        if format_style:
            style_instruction = f"\n用户要求输出的报告需遵循标准的学术格式参考（例如：{format_style}格式的论文大纲）。请在输出的 JSON 中的 'academic_report' 字段，按照该格式的结构（Title, Abstract, Introduction, Methodology 等）生成对应的研究大纲。\n"

        # Language enforcement instruction injected into every round
        if report_language == "en":
            lang_instruction = "\n[IMPORTANT] You MUST write ALL content in the final report (all JSON field values, all markdown output) in English only. Do not use any other language.\n"
        else:
            lang_instruction = "\n[重要] 你必须用简体中文输出所有报告内容（所有 JSON 字段值及 Markdown 内容），不得使用其他语言。\n"

        self._emit_progress("planning", "启动研究前置分析第一轮...")
        yield {"status": "planning", "message": "启动研究前置分析第一轮..."}

        planner_obj = self.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": textwrap.dedent(
                        f"""
                        你是论文写作前置研究助手。目标是在写论文前，帮助用户把模糊想法收敛为严谨、可执行的研究计划。
                        
                        你需要从以下维度进行第一轮分析：
                        1. **研究思路**：明确研究问题、创新点、理论基础
                        2. **研究方向**：具体的技术路线、方法论选择
                        3. **数据需求**：数据类型、规模、获取方式、预处理要求
                        4. **可行性评估**：技术难点、资源需求、时间估算
                        5. **文献基础**：相关领域的关键工作、研究空白
                        {style_instruction}
                        {lang_instruction}
                        """
                    ).strip(),
                },
                {
                    "role": "user",
                    "content": f"用户输入的一句话研究主题：{topic}",
                },
            ],
            response_model=PlannerResponse,
        )
        planner = planner_obj.model_dump()

        self._emit_progress("synthesizing", "进行第二轮深度收敛与验证...")
        yield {"status": "synthesizing", "message": "进行第二轮深度收敛与验证..."}

        synthesizer_obj = self.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": textwrap.dedent(
                        f"""
                        你是论文前置研究讨论中的第二位研究员。你要基于主题、第一轮分析、网页证据，进行第二轮深度收敛。
                        
                        你需要重点关注：
                        1. **研究思路细化**：将初步想法转化为可操作的研究步骤
                        2. **研究方向验证**：基于文献证据，评估各方向的可行性和创新性
                        3. **数据需求具体化**：明确数据的详细规格、获取难度、处理流程
                        4. **实验设计**：初步的实验方案、对照组设置、评估指标
                        5. **风险识别**：技术风险、数据风险、时间风险
                        {lang_instruction}
                        """
                    ).strip(),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "topic": topic,
                            "planner": planner,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            response_model=SynthesizerResponse,
        )
        synthesizer = synthesizer_obj.model_dump()

        self._emit_progress("criticizing", "启动第三轮批判性审查与整合...")
        yield {"status": "criticizing", "message": "启动第三轮批判性审查与整合..."}

        critic_obj = self.chat_json(
            messages=[
                {
                    "role": "system",
                    "content": textwrap.dedent(
                        f"""
                        你是第三位研究员，负责批判性审查并给出最终建议。你必须综合前两轮输出，确保研究计划的严谨性和可执行性。
                        
                        你的职责：
                        1. **研究思路最终确认**：确保逻辑严密、创新点突出、可验证
                        2. **研究方向优先级排序**：基于可行性、创新性、影响力排序
                        3. **数据需求完整性检查**：确保数据获取路径清晰、处理方案可行
                        4. **实验方案可行性评估**：验证实验设计的科学性和完整性
                        5. **风险缓解策略**：为每个识别的风险提供应对方案
                        6. **里程碑规划**：制定阶段性目标和验收标准
                        
                        避免空泛结论，优先给出可执行、可收集数据、可落地验证的建议。
                        
                        请注意，academic_report 字段应当是一篇完整的遵循 SciSpace 标准格式（如 IEEE/APA 等标准科研文章风格）的输出文档大纲及主体内容。
                        {lang_instruction}
                        """
                    ).strip(),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "topic": topic,
                            "planner": planner,
                            "synthesizer": synthesizer,
                        },
                        ensure_ascii=False,
                        indent=2,
                    ),
                },
            ],
            response_model=CriticResponse,
        )
        critic = critic_obj.model_dump()

        self._emit_progress("formatting", "格式化最终报告...")
        yield {"status": "formatting", "message": "格式化最终报告..."}

        self._emit_progress("searching", "正在补充收集网页相关文献与证据...")
        yield {"status": "searching", "message": "正在补充收集网页相关文献与证据..."}

        evidence = self.collect_web_evidence(planner.get("search_queries", []))

        markdown = render_markdown_report(
            topic,
            planner,
            synthesizer,
            critic,
            evidence,
            report_language=report_language,
        )
        yield {
            "status": "done",
            "message": "生成完毕",
            "result": {
                "planner": planner,
                "synthesizer": synthesizer,
                "critic": critic,
                "evidence": evidence,
                "markdown": markdown,
            },
        }

    @staticmethod
    def _clean_duckduckgo_url(url: str) -> str:
        if url.startswith("//"):
            return f"https:{url}"
        if "duckduckgo.com/l/?" not in url:
            return url
        parsed = urlparse(url)
        query = parse_qs(parsed.query)
        uddg = query.get("uddg", [""])[0]
        return unquote(uddg) if uddg else url


def normalize_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def render_markdown_report(
    topic: str,
    planner: Dict[str, Any],
    synthesizer: Dict[str, Any],
    critic: Dict[str, Any],
    evidence: List[Dict[str, str]],
    report_language: str = "zh-CN",
) -> str:
    en = report_language == "en"
    ideas = (
        critic.get("final_research_ideas")
        or synthesizer.get("refined_ideas")
        or planner.get("initial_research_ideas")
        or []
    )
    directions = (
        critic.get("final_research_directions")
        or synthesizer.get("refined_directions")
        or planner.get("candidate_directions")
        or []
    )
    data_needed = (
        critic.get("final_data_needed") or synthesizer.get("data_requirements") or []
    )
    discussion = critic.get("discussion_highlights") or []
    assumptions = critic.get("final_assumptions") or planner.get("assumptions") or []

    lines: List[str] = []
    lines.append("## Input Topic" if en else "## 输入主题")
    lines.append("")
    lines.append(topic)
    lines.append("")

    lines.append("## Research Ideas" if en else "## 研究思路")
    lines.append("")

    ideas_list = critic.get("final_research_ideas", [])
    if isinstance(ideas_list, list) and ideas_list and isinstance(ideas_list[0], dict):
        for idx, item in enumerate(ideas_list, 1):
            lines.append(
                f"### {'Idea' if en else '思路'} {idx}：{item.get('idea', '')}"
            )
            lines.append(
                f"- **{'Rationale' if en else '选择理由'}**：{item.get('rationale', '')}"
            )
            lines.append(
                f"- **{'Validation Method' if en else '验证方法'}**：{item.get('validation_method', '')}"
            )
            lines.append(
                f"- **{'Expected Contribution' if en else '预期贡献'}**：{item.get('expected_contribution', '')}"
            )
            lines.append("")
    elif ideas:
        for item in ideas:
            if isinstance(item, str):
                lines.append(f"- {item}")
            else:
                lines.append(f"- {json.dumps(item, ensure_ascii=False)}")
    else:
        lines.append("- No results" if en else "- 暂无结果")
    lines.append("")

    lines.append("## Research Directions" if en else "## 研究方向")
    lines.append("")
    directions_list = critic.get("final_research_directions", [])
    if (
        isinstance(directions_list, list)
        and directions_list
        and isinstance(directions_list[0], dict)
    ):
        for idx, item in enumerate(directions_list, 1):
            lines.append(
                f"### {'Direction' if en else '方向'} {idx}：{item.get('direction', '')}"
            )
            lines.append(
                f"- **{'Priority' if en else '优先级'}**：{item.get('priority', '')}"
            )
            lines.append(
                f"- **{'Feasibility Score' if en else '可行性得分'}**：{item.get('feasibility_score', '')}"
            )
            lines.append(
                f"- **{'Innovation Score' if en else '创新性得分'}**：{item.get('innovation_score', '')}"
            )
            lines.append(
                f"- **{'Technical Approach' if en else '具体技术方案'}**：{item.get('technical_approach', '')}"
            )
            lines.append(
                f"- **{'Success Criteria' if en else '成功标准'}**：{item.get('success_criteria', '')}"
            )
            lines.append(
                f"- **{'Fallback Plan' if en else '备选方案'}**：{item.get('fallback_plan', '')}"
            )
            lines.append("")
    elif directions:
        for idx, item in enumerate(directions, start=1):
            if isinstance(item, str):
                lines.append(f"{idx}. {item}")
            else:
                lines.append(f"{idx}. {json.dumps(item, ensure_ascii=False)}")
    else:
        lines.append("1. No results" if en else "1. 暂无结果")
    lines.append("")

    lines.append("## Data Requirements" if en else "## 需要哪些数据")
    lines.append("")
    if data_needed:
        for item in data_needed:
            if isinstance(item, str):
                lines.append(f"- {item}")
                continue

            data_name = item.get("data_name", "Unnamed" if en else "未命名数据")
            purpose = item.get("purpose") or item.get("why_needed") or ""
            key_fields = item.get("key_fields", [])
            sources = item.get("possible_sources", [])
            notes = item.get("notes", "")

            lines.append(f"### {data_name}")
            if purpose:
                lines.append(f"- **{'Purpose' if en else '用途'}**：{purpose}")

            data_type = item.get("data_type", "")
            if data_type:
                lines.append(f"- **{'Data Type' if en else '数据类型'}**：{data_type}")

            scale = item.get("scale_requirement", "")
            if scale:
                lines.append(
                    f"- **{'Scale Requirement' if en else '规模要求'}**：{scale}"
                )

            if key_fields:
                lines.append(
                    f"- **{'Key Fields' if en else '关键字段'}**：{', '.join(key_fields)}"
                )
            if sources:
                lines.append(f"- **{'Potential Sources' if en else '潜在来源'}**：")
                for src in sources:
                    lines.append(f"  - {src}")

            difficulty = item.get("acquisition_difficulty", "")
            if difficulty:
                lines.append(
                    f"- **{'Acquisition Difficulty' if en else '获取难度'}**：{difficulty}"
                )

            cost = item.get("estimated_cost", "")
            if cost:
                lines.append(f"- **{'Estimated Cost' if en else '成本估算'}**：{cost}")

            time = item.get("estimated_time", "")
            if time:
                lines.append(f"- **{'Estimated Time' if en else '时间估算'}**：{time}")

            if notes:
                lines.append(f"- **{'Notes' if en else '备注'}**：{notes}")
            lines.append("")
    else:
        lines.append("- No results" if en else "- 暂无结果")
        lines.append("")

    exp_plan = critic.get("experiment_plan")
    if exp_plan:
        lines.append("## Experiment Plan" if en else "## 实验方案规划")
        lines.append("")
        phases = exp_plan.get("phases", [])
        if phases:
            lines.append("### Implementation Phases" if en else "### 实施阶段")
            for phase in phases:
                lines.append(
                    f"#### {phase.get('phase_name', 'Phase' if en else '阶段')} ({phase.get('duration', 'TBD' if en else '时长未定')})"
                )
                lines.append(
                    f"- **{'Objectives' if en else '目标'}**: {', '.join(phase.get('objectives', []))}"
                )
                lines.append(
                    f"- **{'Tasks' if en else '任务'}**: {', '.join(phase.get('tasks', []))}"
                )
                lines.append(
                    f"- **{'Deliverables' if en else '交付物'}**: {', '.join(phase.get('deliverables', []))}"
                )
                lines.append("")

        baselines = exp_plan.get("baseline_methods", [])
        if baselines:
            lines.append(
                f"- **{'Baseline Methods' if en else '基线方法'}**: {', '.join(baselines)}"
            )

        metrics = exp_plan.get("evaluation_metrics", [])
        if metrics:
            lines.append(
                f"- **{'Evaluation Metrics' if en else '评估指标'}**: {', '.join(metrics)}"
            )

        val_strategy = exp_plan.get("validation_strategy", "")
        if val_strategy:
            lines.append(
                f"- **{'Validation Strategy' if en else '验证策略'}**: {val_strategy}"
            )
        lines.append("")

    risks = critic.get("risk_mitigation")
    if risks:
        lines.append("## Risk Mitigation" if en else "## 风险缓解策略")
        lines.append("")
        for risk in risks:
            lines.append(
                f"### {risk.get('risk', 'Unnamed Risk' if en else '未命名风险')}"
            )
            lines.append(
                f"- **{'Impact' if en else '影响程度'}**: {risk.get('impact', '')} | **{'Probability' if en else '发生概率'}**: {risk.get('probability', '')}"
            )
            lines.append(
                f"- **{'Mitigation Strategy' if en else '缓解策略'}**: {risk.get('mitigation_strategy', '')}"
            )
            lines.append(
                f"- **{'Contingency Plan' if en else '应急预案'}**: {risk.get('contingency_plan', '')}"
            )
            lines.append("")

    if assumptions:
        lines.append("## Key Assumptions" if en else "## 关键假设")
        lines.append("")
        for item in assumptions:
            lines.append(f"- {item}")
        lines.append("")

    if discussion:
        lines.append("## Discussion Highlights" if en else "## 多轮讨论摘要")
        lines.append("")
        for item in discussion:
            lines.append(f"- {item}")
        lines.append("")

    if evidence:
        lines.append("## References" if en else "## 参考网页")
        lines.append("")
        for item in evidence:
            title = item.get("title", "Untitled")
            url = item.get("url", "")
            query = item.get("query", "")
            if url:
                lines.append(f"- [{title}]({url})  ")
                lines.append(f"  query: `{query}`")
            else:
                lines.append(f"- {title} (`{query}`)")
        lines.append("")

    academic_report = critic.get("academic_report")
    if academic_report:
        lines.append("---")
        lines.append("")
        lines.append(
            "# Draft in Standard Format"
            if en
            else "# 标准格式草案 (Draft in Standard Format)"
        )
        lines.append("")
        lines.append(f"**Title**: {academic_report.get('title', 'Untitled')}")
        lines.append("")
        lines.append("## Abstract")
        lines.append(academic_report.get("abstract", ""))
        lines.append("")
        lines.append("## Introduction (Research Ideas)")
        lines.append(academic_report.get("introduction", ""))
        lines.append("")
        lines.append("## Methodology (Directions & Data Required)")
        lines.append(academic_report.get("methodology", ""))
        lines.append("")
        lines.append("## Expected Outcomes")
        lines.append(academic_report.get("expected_outcomes", ""))
        lines.append("")
        lines.append("## References")
        for ref in academic_report.get("references", []):
            lines.append(f"- {ref}")
        lines.append("")

    return "\n".join(lines).strip() + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="输入一句话研究主题，自动进行多轮 LLM 讨论并生成 reports.md"
    )
    parser.add_argument(
        "topic",
        nargs="?",
        help="一句话研究主题，例如：用多模态数据预测抑郁症风险",
    )
    parser.add_argument(
        "-o",
        "--output",
        default=DEFAULT_OUTPUT,
        help=f"输出 Markdown 文件路径，默认 {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--format",
        default="IEEE/APA标准期刊",
        help="指定的标准学术输出格式，默认 IEEE/APA标准期刊",
    )
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help=f"LLM 模型名，默认 {DEFAULT_MODEL}",
    )
    parser.add_argument(
        "--base-url",
        default=DEFAULT_BASE_URL,
        help=f"OpenAI 兼容接口地址，默认 {DEFAULT_BASE_URL}",
    )
    return parser.parse_args()


def ensure_topic(topic: str | None) -> str:
    if topic:
        return topic.strip()
    value = input("请输入一句话研究主题：").strip()
    if not value:
        raise ValueError("研究主题不能为空")
    return value


def main() -> int:
    args = parse_args()
    topic = ensure_topic(args.topic)
    api_key = os.getenv("OPENAI_API_KEY", "").strip()
    if not api_key:
        print(
            "缺少 OPENAI_API_KEY。请先通过 env 传入 Gemini OpenAI-Compatible API Key 后再运行。",
            file=sys.stderr,
        )
        return 1

    agent = ResearchPrerequisiteAgent(
        api_key=api_key,
        model=args.model,
        base_url=args.base_url,
    )

    try:
        generator = agent.generate_report(topic, format_style=args.format)
        result = None
        for step in generator:
            print(step.get("message", ""))
            if step.get("status") == "done":
                result = step["result"]
                break
        if not result:
            print("生成失败: 未收到完成状态", file=sys.stderr)
            return 1
    except Exception as exc:
        print(f"生成失败: {exc}", file=sys.stderr)
        return 1

    if not isinstance(result, dict) or "markdown" not in result:
        print("生成失败: 结果格式错误", file=sys.stderr)
        return 1

    with open(args.output, "w", encoding="utf-8") as fp:
        fp.write(result["markdown"])

    print(f"已生成 {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
