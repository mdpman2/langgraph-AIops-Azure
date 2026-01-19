# Copyright (c) Microsoft. All rights reserved.
"""
LangGraph-style Workflow Implementation using Azure OpenAI

Planning → Execution → Reflection → Decision 사이클을 구현한 워크플로우
Azure OpenAI SDK를 직접 사용
"""

import asyncio
import json
import structlog
from datetime import datetime, timezone
from functools import lru_cache
from typing import Any, Dict, List, Optional, TypeVar
from uuid import uuid4

from openai import AsyncAzureOpenAI
from azure.identity.aio import DefaultAzureCredential, get_bearer_token_provider

from .models import (
    AgentState,
    Plan,
    PlanStep,
    ExecutionResult,
    ReflectionResult,
    Decision,
    DecisionType,
    WorkflowStage,
)
from .config import AgentConfig, load_config

logger = structlog.get_logger(__name__)

# Type variable for generic typing
T = TypeVar('T', bound=Dict[str, Any])


def _extract_json_from_text(text: str, default: Optional[T] = None) -> T:
    """텍스트에서 JSON 추출 - 최적화된 파싱"""
    if not text:
        return default or {}

    try:
        # JSON 블록 추출 (한 번만 검색)
        json_start = text.find("{")
        if json_start == -1:
            return default or {}

        json_end = text.rfind("}") + 1
        if json_end <= json_start:
            return default or {}

        return json.loads(text[json_start:json_end])
    except json.JSONDecodeError:
        return default or {}


@lru_cache(maxsize=10)
def _get_system_prompt(prompt_type: str) -> str:
    """시스템 프롬프트 캐싱"""
    prompts = {
        "planning": """당신은 전문 계획 수립 에이전트입니다.
사용자의 요청을 분석하고 단계별 실행 계획을 수립합니다.

출력 형식 (JSON):
{
    "goal": "최종 목표",
    "steps": [
        {
            "step_number": 1,
            "description": "단계 설명",
            "action": "수행할 액션",
            "expected_output": "예상 출력"
        }
    ]
}

규칙:
1. 각 단계는 구체적이고 실행 가능해야 합니다
2. 단계 간 의존성을 고려하세요
3. 최대 5단계로 제한하세요
4. 재계획 시 이전 피드백을 반영하세요""",

        "execution": """당신은 실행 에이전트입니다.
주어진 작업을 수행하고 결과를 반환합니다.
명확하고 구체적인 결과를 제공하세요.""",

        "reflection": """당신은 비판적 평가 에이전트입니다.
실행 결과를 분석하고 품질을 평가합니다.

출력 형식 (JSON):
{
    "quality_score": 0.0-1.0,
    "goal_alignment": 0.0-1.0,
    "completeness": 0.0-1.0,
    "strengths": ["잘된 점들"],
    "weaknesses": ["개선 필요 점들"],
    "suggestions": ["개선 제안들"],
    "needs_retry": true/false,
    "needs_replan": true/false,
    "reasoning": "판단 근거"
}

평가 기준:
1. 목표와의 일치도
2. 결과의 완성도
3. 품질 수준
4. 실행 효율성"""
    }
    return prompts.get(prompt_type, "")


class AzureOpenAIClient:
    """Azure OpenAI 클라이언트 래퍼"""

    def __init__(self, config: AgentConfig):
        self.config = config
        self._client: Optional[AsyncAzureOpenAI] = None
        self._credential: Optional[DefaultAzureCredential] = None

    async def get_client(self) -> AsyncAzureOpenAI:
        """Azure OpenAI 클라이언트 가져오기 (싱글톤)"""
        if self._client is None:
            if self.config.azure_openai_api_key:
                # API 키 사용
                self._client = AsyncAzureOpenAI(
                    api_key=self.config.azure_openai_api_key,
                    azure_endpoint=self.config.azure_openai_endpoint,
                    api_version="2024-10-01-preview",
                )
            else:
                # DefaultAzureCredential 사용
                self._credential = DefaultAzureCredential()
                token_provider = get_bearer_token_provider(
                    self._credential,
                    "https://cognitiveservices.azure.com/.default"
                )
                self._client = AsyncAzureOpenAI(
                    azure_ad_token_provider=token_provider,
                    azure_endpoint=self.config.azure_openai_endpoint or self.config.azure_foundry_project_endpoint,
                    api_version="2024-10-01-preview",
                )
        return self._client

    async def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2000,
    ) -> str:
        """채팅 완성 요청"""
        client = await self.get_client()
        deployment = self.config.azure_openai_deployment_name or self.config.azure_foundry_model_deployment

        response = await client.chat.completions.create(
            model=deployment,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )

        return response.choices[0].message.content or ""

    async def close(self):
        """리소스 정리"""
        if self._client:
            await self._client.close()
        if self._credential:
            await self._credential.close()


# ============================================
# Planning Node - 계획 수립
# ============================================

async def planning_node(
    client: AzureOpenAIClient,
    state: AgentState,
    feedback: Optional[str] = None
) -> AgentState:
    """계획 수립 노드"""
    logger.info("planning_started", session_id=state.session_id, feedback=feedback)

    system_prompt = _get_system_prompt("planning")
    user_prompt = f"사용자 요청: {state.user_request}"
    if feedback:
        user_prompt = f"{user_prompt}\n\n이전 시도의 피드백: {feedback}"

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response_text = await client.chat(messages)

    # JSON 파싱
    default_plan = {"goal": state.user_request, "steps": []}
    plan_data = _extract_json_from_text(response_text, default_plan)

    # Plan 객체 생성
    steps = []
    for step_data in plan_data.get("steps", []):
        steps.append(PlanStep(
            step_number=step_data.get("step_number", len(steps) + 1),
            description=step_data.get("description", ""),
            action=step_data.get("action", ""),
            expected_output=step_data.get("expected_output", ""),
        ))

    # 상태 업데이트
    version = (state.current_plan.version + 1) if state.current_plan else 1
    state.current_plan = Plan(
        goal=plan_data.get("goal", state.user_request),
        steps=steps,
        version=version,
    )
    state.current_stage = WorkflowStage.EXECUTION
    state.add_message("assistant", f"계획 수립 완료: {len(steps)}개 단계")

    logger.info("planning_completed",
               session_id=state.session_id,
               plan_version=version,
               steps_count=len(steps))

    return state


# ============================================
# Execution Node - 실행
# ============================================

async def execution_node(client: AzureOpenAIClient, state: AgentState) -> AgentState:
    """실행 노드"""
    logger.info("execution_started", session_id=state.session_id)

    if not state.current_plan:
        state.error_message = "실행할 계획이 없습니다"
        state.current_stage = WorkflowStage.FAILED
        return state

    # 다음 실행할 단계 가져오기
    next_step = state.current_plan.get_next_step()
    if not next_step:
        state.current_stage = WorkflowStage.REFLECTION
        return state

    next_step.status = "in_progress"
    start_time = datetime.now(timezone.utc)

    system_prompt = _get_system_prompt("execution")
    user_prompt = f"""
작업: {next_step.description}
액션: {next_step.action}
예상 출력: {next_step.expected_output}

전체 목표: {state.current_plan.goal}
사용자 요청: {state.user_request}

이 작업을 수행하고 결과를 반환하세요.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    try:
        response_text = await client.chat(messages)
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        execution_result = ExecutionResult(
            step_id=next_step.step_id,
            success=True,
            output=response_text,
            duration_seconds=duration,
        )
        state.execution_history.append(execution_result)
        next_step.status = "completed"
        state.add_message("assistant", f"단계 {next_step.step_number} 실행 완료")

        logger.info("execution_step_completed",
                   session_id=state.session_id,
                   step_id=next_step.step_id,
                   duration=duration)

    except Exception as e:
        end_time = datetime.now(timezone.utc)
        duration = (end_time - start_time).total_seconds()

        execution_result = ExecutionResult(
            step_id=next_step.step_id,
            success=False,
            output=None,
            error=str(e),
            duration_seconds=duration,
        )
        state.execution_history.append(execution_result)
        next_step.status = "failed"

        logger.error("execution_step_failed",
                    session_id=state.session_id,
                    step_id=next_step.step_id,
                    error=str(e))

    # 다음 단계 확인
    if state.current_plan.is_complete():
        state.current_stage = WorkflowStage.REFLECTION
    else:
        state.current_stage = WorkflowStage.EXECUTION

    state.iteration_count += 1
    return state


# ============================================
# Reflection Node - 성찰
# ============================================

async def reflection_node(client: AzureOpenAIClient, state: AgentState) -> AgentState:
    """성찰 노드"""
    logger.info("reflection_started", session_id=state.session_id)

    recent_executions = state.execution_history[-5:] if state.execution_history else []
    system_prompt = _get_system_prompt("reflection")

    execution_summary = "\n".join([
        f"- 단계 {e.step_id}: {'성공' if e.success else '실패'}, 출력: {str(e.output)[:200]}"
        for e in recent_executions
    ])

    user_prompt = f"""
원래 목표: {state.current_plan.goal if state.current_plan else state.user_request}
사용자 요청: {state.user_request}

실행 결과:
{execution_summary}

이 결과들을 평가하고 분석해주세요.
"""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt}
    ]

    response_text = await client.chat(messages)

    default_reflection = {
        "quality_score": 0.5,
        "goal_alignment": 0.5,
        "completeness": 0.5,
        "reasoning": "파싱 실패로 기본값 사용"
    }
    reflection_data = _extract_json_from_text(response_text, default_reflection)

    latest_execution = state.get_latest_execution()
    reflection_result = ReflectionResult(
        execution_id=latest_execution.execution_id if latest_execution else "",
        quality_score=reflection_data.get("quality_score", 0.5),
        goal_alignment=reflection_data.get("goal_alignment", 0.5),
        completeness=reflection_data.get("completeness", 0.5),
        strengths=reflection_data.get("strengths", []),
        weaknesses=reflection_data.get("weaknesses", []),
        suggestions=reflection_data.get("suggestions", []),
        needs_retry=reflection_data.get("needs_retry", False),
        needs_replan=reflection_data.get("needs_replan", False),
        reasoning=reflection_data.get("reasoning", ""),
    )

    state.reflection_history.append(reflection_result)
    state.reflection_count += 1
    state.current_stage = WorkflowStage.DECISION
    state.add_message("assistant", f"성찰 완료: 품질 점수 {reflection_result.quality_score:.2f}")

    logger.info("reflection_completed",
               session_id=state.session_id,
               quality_score=reflection_result.quality_score,
               needs_retry=reflection_result.needs_retry,
               needs_replan=reflection_result.needs_replan)

    return state


# ============================================
# Decision Node - 결정
# ============================================

def decision_node(config: AgentConfig, state: AgentState) -> tuple[AgentState, Decision]:
    """결정 노드"""
    logger.info("decision_started", session_id=state.session_id)

    latest_reflection = state.get_latest_reflection()

    if not latest_reflection:
        decision = Decision(
            decision_type=DecisionType.FAIL,
            reasoning="성찰 결과가 없음",
            next_action="fail",
            confidence=1.0,
        )
        state.decision_history.append(decision)
        state.current_stage = WorkflowStage.FAILED
        state.error_message = "성찰 결과가 없어 진행 불가"
        return state, decision

    # 결정 로직
    if latest_reflection.quality_score >= 0.8 and latest_reflection.completeness >= 0.8:
        decision_type = DecisionType.COMPLETE
        reasoning = f"품질 점수 {latest_reflection.quality_score:.2f}, 완성도 {latest_reflection.completeness:.2f}로 목표 달성"
        next_action = "complete"
        confidence = 0.9

    elif latest_reflection.needs_replan and state.reflection_count < config.max_reflection_iterations:
        decision_type = DecisionType.REPLAN
        reasoning = f"성찰 결과 재계획 필요: {latest_reflection.reasoning}"
        next_action = "replan"
        confidence = 0.7

    elif latest_reflection.needs_retry and state.reflection_count < config.max_reflection_iterations:
        decision_type = DecisionType.RETRY
        reasoning = f"성찰 결과 재시도 필요: {latest_reflection.reasoning}"
        next_action = "retry"
        confidence = 0.7

    elif state.reflection_count >= config.max_reflection_iterations:
        if latest_reflection.quality_score >= 0.5:
            decision_type = DecisionType.COMPLETE
            reasoning = f"최대 반복 횟수 도달, 현재 품질 {latest_reflection.quality_score:.2f}로 완료 처리"
            next_action = "complete"
            confidence = 0.6
        else:
            decision_type = DecisionType.FAIL
            reasoning = "최대 반복 횟수 도달, 품질 미달로 실패 처리"
            next_action = "fail"
            confidence = 0.8
    else:
        decision_type = DecisionType.CONTINUE
        reasoning = "추가 작업이 필요함"
        next_action = "continue"
        confidence = 0.7

    decision = Decision(
        decision_type=decision_type,
        reasoning=reasoning,
        next_action=next_action,
        confidence=confidence,
        feedback_for_planning="\n".join(latest_reflection.suggestions) if decision_type == DecisionType.REPLAN else None,
        feedback_for_execution="\n".join(latest_reflection.weaknesses) if decision_type == DecisionType.RETRY else None,
    )

    state.decision_history.append(decision)
    state.add_message("assistant", f"결정: {decision_type.value} (신뢰도: {confidence:.2f})")

    logger.info("decision_completed",
               session_id=state.session_id,
               decision_type=decision_type.value,
               confidence=confidence)

    return state, decision


def _handle_decision_result(
    state: AgentState,
    decision: Decision,
) -> tuple[AgentState, Optional[str]]:
    """결정 결과에 따른 상태 업데이트 - 공통 로직 추출"""
    feedback = None

    if decision.decision_type == DecisionType.COMPLETE:
        state.current_stage = WorkflowStage.COMPLETED
        state.final_output = _generate_final_output(state)
    elif decision.decision_type == DecisionType.FAIL:
        state.current_stage = WorkflowStage.FAILED
    elif decision.decision_type == DecisionType.REPLAN:
        state.current_stage = WorkflowStage.PLANNING
        feedback = decision.feedback_for_planning
    elif decision.decision_type == DecisionType.RETRY:
        if state.current_plan:
            for step in reversed(state.current_plan.steps):
                if step.status == "failed":
                    step.status = "pending"
                    break
        state.current_stage = WorkflowStage.EXECUTION
    else:  # CONTINUE
        state.current_stage = WorkflowStage.EXECUTION

    return state, feedback


def _generate_final_output(state: AgentState) -> str:
    """최종 출력 생성"""
    outputs = []
    outputs.append(f"## 작업 완료\n")
    outputs.append(f"**원래 요청:** {state.user_request}\n")

    if state.current_plan:
        outputs.append(f"\n### 수행된 계획")
        outputs.append(f"**목표:** {state.current_plan.goal}\n")
        for step in state.current_plan.steps:
            status_icon = "✅" if step.status == "completed" else "❌"
            outputs.append(f"{status_icon} 단계 {step.step_number}: {step.description}")

    if state.execution_history:
        outputs.append(f"\n### 실행 결과")
        for i, execution in enumerate(state.execution_history[-3:], 1):
            outputs.append(f"\n**실행 {i}:**")
            outputs.append(str(execution.output)[:500])

    if state.reflection_history:
        latest = state.reflection_history[-1]
        outputs.append(f"\n### 최종 평가")
        outputs.append(f"- 품질 점수: {latest.quality_score:.2f}")
        outputs.append(f"- 목표 정렬도: {latest.goal_alignment:.2f}")
        outputs.append(f"- 완성도: {latest.completeness:.2f}")

    return "\n".join(outputs)


# ============================================
# Agent Workflow - 전체 워크플로우 조합
# ============================================

class AgentWorkflow:
    """
    LangGraph 스타일 AI 에이전트 워크플로우

    Planning → Execution → Reflection → Decision 사이클 구현
    Azure OpenAI SDK 직접 사용
    """

    def __init__(self, config: Optional[AgentConfig] = None):
        self.config = config or load_config()
        self._client: Optional[AzureOpenAIClient] = None

    async def _get_client(self) -> AzureOpenAIClient:
        """클라이언트 가져오기"""
        if self._client is None:
            self._client = AzureOpenAIClient(self.config)
        return self._client

    async def run_stream(self, user_request: str):
        """스트리밍 워크플로우 실행 - 각 단계별로 진행 상황을 yield"""
        client = await self._get_client()

        # 초기 상태 생성
        state = AgentState(user_request=user_request)

        logger.info("workflow_started",
                   session_id=state.session_id,
                   user_request=user_request[:100])

        yield {"type": "start", "message": "워크플로우 시작", "session_id": state.session_id}

        feedback: Optional[str] = None

        while state.should_continue():
            current_stage = state.current_stage

            if current_stage == WorkflowStage.PLANNING:
                yield {"type": "stage", "stage": "planning", "message": "📋 계획 수립 중..."}
                state = await planning_node(client, state, feedback)
                feedback = None

                if state.current_plan:
                    plan_info = {
                        "type": "plan",
                        "goal": state.current_plan.goal,
                        "steps": [
                            {"step": s.step_number, "description": s.description}
                            for s in state.current_plan.steps
                        ]
                    }
                    yield plan_info

            elif current_stage == WorkflowStage.EXECUTION:
                yield {"type": "stage", "stage": "execution", "message": "⚡ 실행 중..."}

                # 모든 단계 실행
                while state.current_stage == WorkflowStage.EXECUTION:
                    if state.current_plan:
                        next_step = state.current_plan.get_next_step()
                        if next_step:
                            yield {
                                "type": "step",
                                "step_number": next_step.step_number,
                                "description": next_step.description,
                                "status": "executing"
                            }

                    state = await execution_node(client, state)

                    # 실행 결과 전송
                    if state.execution_history:
                        latest = state.execution_history[-1]
                        yield {
                            "type": "execution_result",
                            "step_id": latest.step_id,
                            "success": latest.success,
                            "output": str(latest.output)[:500] if latest.output else None
                        }

            elif current_stage == WorkflowStage.REFLECTION:
                yield {"type": "stage", "stage": "reflection", "message": "🔍 결과 분석 중..."}
                state = await reflection_node(client, state)

                if state.reflection_history:
                    latest = state.reflection_history[-1]
                    yield {
                        "type": "reflection",
                        "quality_score": latest.quality_score,
                        "goal_alignment": latest.goal_alignment,
                        "completeness": latest.completeness
                    }

            elif current_stage == WorkflowStage.DECISION:
                yield {"type": "stage", "stage": "decision", "message": "🎯 결정 중..."}
                state, decision = decision_node(self.config, state)

                yield {
                    "type": "decision",
                    "decision_type": decision.decision_type.value,
                    "reasoning": decision.reasoning
                }

                state, feedback = _handle_decision_result(state, decision)

                if decision.decision_type in (DecisionType.COMPLETE, DecisionType.FAIL):
                    break

        # 최종 결과
        if state.final_output:
            yield {"type": "complete", "result": state.final_output}
        elif state.error_message:
            yield {"type": "error", "message": state.error_message}
        else:
            yield {"type": "complete", "result": "워크플로우가 완료되었으나 출력이 없습니다."}

        logger.info("workflow_completed",
                   session_id=state.session_id,
                   final_stage=state.current_stage.value)

    async def run(self, user_request: str) -> str:
        """워크플로우 실행"""
        client = await self._get_client()

        # 초기 상태 생성
        state = AgentState(user_request=user_request)

        logger.info("workflow_started",
                   session_id=state.session_id,
                   user_request=user_request[:100])

        feedback: Optional[str] = None

        while state.should_continue():
            current_stage = state.current_stage

            if current_stage == WorkflowStage.PLANNING:
                state = await planning_node(client, state, feedback)
                feedback = None

            elif current_stage == WorkflowStage.EXECUTION:
                # 모든 단계 실행
                while state.current_stage == WorkflowStage.EXECUTION:
                    state = await execution_node(client, state)

            elif current_stage == WorkflowStage.REFLECTION:
                state = await reflection_node(client, state)

            elif current_stage == WorkflowStage.DECISION:
                state, decision = decision_node(self.config, state)
                state, feedback = _handle_decision_result(state, decision)

                if decision.decision_type in (DecisionType.COMPLETE, DecisionType.FAIL):
                    break

        logger.info("workflow_completed",
                   session_id=state.session_id,
                   final_stage=state.current_stage.value)

        if state.final_output:
            return state.final_output
        elif state.error_message:
            return f"실패: {state.error_message}"
        else:
            return "워크플로우가 완료되었으나 출력이 없습니다."

    async def close(self):
        """리소스 정리"""
        if self._client:
            await self._client.close()
