# Copyright (c) Microsoft. All rights reserved.
"""
Data models for the LangGraph-style Agent Framework

Planning → Execution → Reflection → Decision 워크플로우의 상태 및 데이터 모델

최적화:
- Pydantic v2 성능 최적화 설정
- 캐싱 및 지연 로딩
- 타입 힌트 강화
"""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from functools import cached_property
from typing import Any, Dict, List, Optional, TYPE_CHECKING
from pydantic import BaseModel, Field, field_validator, model_validator
from uuid import uuid4

if TYPE_CHECKING:
    from typing import Self


class WorkflowStage(str, Enum):
    """현재 워크플로우 단계"""
    __slots__ = ()  # 메모리 최적화

    PLANNING = "planning"
    EXECUTION = "execution"
    REFLECTION = "reflection"
    DECISION = "decision"
    COMPLETED = "completed"
    FAILED = "failed"


class DecisionType(str, Enum):
    """Decision 노드의 결정 유형"""
    __slots__ = ()  # 메모리 최적화

    CONTINUE = "continue"       # 다음 단계로 진행
    RETRY = "retry"             # 현재 단계 재시도
    REPLAN = "replan"           # 계획 재수립
    COMPLETE = "complete"       # 작업 완료
    FAIL = "fail"               # 실패 처리


class PlanStep(BaseModel):
    """계획의 개별 단계"""

    model_config = {
        "json_schema_extra": {
            "example": {
                "step_id": "step_001",
                "step_number": 1,
                "description": "사용자 요청 분석",
                "action": "analyze_request",
                "expected_output": "분석된 요구사항 목록",
                "dependencies": [],
                "status": "pending"
            }
        }
    }

    step_id: str = Field(default_factory=lambda: str(uuid4())[:8])
    step_number: int = Field(description="단계 번호")
    description: str = Field(description="단계 설명")
    action: str = Field(description="수행할 액션")
    expected_output: str = Field(description="예상 출력")
    dependencies: List[str] = Field(default_factory=list, description="의존하는 이전 단계 ID들")
    status: str = Field(default="pending", description="단계 상태: pending, in_progress, completed, failed")


class Plan(BaseModel):
    """전체 실행 계획"""

    model_config = {"frozen": False, "extra": "ignore"}

    plan_id: str = Field(default_factory=lambda: str(uuid4()))
    goal: str = Field(description="최종 목표")
    steps: List[PlanStep] = Field(default_factory=list, description="실행 단계들")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = Field(default=1, description="계획 버전 (재계획 시 증가)")

    @cached_property
    def total_steps(self) -> int:
        """총 단계 수 (캐시됨)"""
        return len(self.steps)

    def get_next_step(self) -> Optional[PlanStep]:
        """다음 실행할 단계 반환 - 선형 검색 최적화"""
        return next((step for step in self.steps if step.status == "pending"), None)

    def mark_step_completed(self, step_id: str) -> None:
        """단계를 완료로 표시"""
        for step in self.steps:
            if step.step_id == step_id:
                step.status = "completed"
                break

    def is_complete(self) -> bool:
        """모든 단계가 완료되었는지 확인"""
        return all(step.status == "completed" for step in self.steps)


class ExecutionResult(BaseModel):
    """실행 결과"""

    model_config = {"frozen": True}  # 불변 객체로 설정

    execution_id: str = Field(default_factory=lambda: str(uuid4()))
    step_id: str = Field(description="실행된 계획 단계 ID")
    success: bool = Field(description="실행 성공 여부")
    output: Any = Field(default=None, description="실행 출력")
    error: Optional[str] = Field(default=None, description="에러 메시지")
    duration_seconds: float = Field(ge=0, description="실행 시간(초)")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="추가 메타데이터")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    @field_validator('duration_seconds')
    @classmethod
    def validate_duration(cls, v: float) -> float:
        """duration 음수 방지"""
        return max(0.0, v)


class ReflectionResult(BaseModel):
    """성찰 결과"""
    reflection_id: str = Field(default_factory=lambda: str(uuid4()))
    execution_id: str = Field(description="성찰 대상 실행 ID")

    # 성찰 분석
    quality_score: float = Field(ge=0.0, le=1.0, description="품질 점수 (0-1)")
    goal_alignment: float = Field(ge=0.0, le=1.0, description="목표 정렬도 (0-1)")
    completeness: float = Field(ge=0.0, le=1.0, description="완성도 (0-1)")

    # 분석 내용
    strengths: List[str] = Field(default_factory=list, description="잘된 점")
    weaknesses: List[str] = Field(default_factory=list, description="개선 필요 점")
    suggestions: List[str] = Field(default_factory=list, description="개선 제안")

    # 판단
    needs_retry: bool = Field(default=False, description="재시도 필요 여부")
    needs_replan: bool = Field(default=False, description="재계획 필요 여부")
    reasoning: str = Field(description="판단 근거")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Decision(BaseModel):
    """Decision 노드의 결정"""
    decision_id: str = Field(default_factory=lambda: str(uuid4()))
    decision_type: DecisionType = Field(description="결정 유형")
    reasoning: str = Field(description="결정 근거")
    next_action: str = Field(description="다음 수행할 액션")
    confidence: float = Field(ge=0.0, le=1.0, description="결정 신뢰도")

    # 선택적 피드백
    feedback_for_planning: Optional[str] = Field(default=None, description="Planning에 전달할 피드백")
    feedback_for_execution: Optional[str] = Field(default=None, description="Execution에 전달할 피드백")

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class ConversationMessage(BaseModel):
    """대화 메시지"""
    role: str = Field(description="메시지 역할: user, assistant, system")
    content: str = Field(description="메시지 내용")
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = Field(default_factory=dict)


class AgentState(BaseModel):
    """에이전트 전체 상태 - LangGraph 스타일 상태 관리"""

    # 기본 정보
    session_id: str = Field(default_factory=lambda: str(uuid4()))
    user_request: str = Field(description="사용자의 원래 요청")

    # 워크플로우 상태
    current_stage: WorkflowStage = Field(default=WorkflowStage.PLANNING)
    iteration_count: int = Field(default=0, description="전체 반복 횟수")
    reflection_count: int = Field(default=0, description="성찰 반복 횟수")

    # 계획 및 실행 상태
    current_plan: Optional[Plan] = Field(default=None, description="현재 계획")
    execution_history: List[ExecutionResult] = Field(default_factory=list, description="실행 기록")
    reflection_history: List[ReflectionResult] = Field(default_factory=list, description="성찰 기록")
    decision_history: List[Decision] = Field(default_factory=list, description="결정 기록")

    # 대화 기록
    conversation_history: List[ConversationMessage] = Field(default_factory=list, description="대화 기록")

    # 컨텍스트 및 메모리
    context: Dict[str, Any] = Field(default_factory=dict, description="추가 컨텍스트")
    memory: Dict[str, Any] = Field(default_factory=dict, description="장기 메모리")

    # 최종 결과
    final_output: Optional[str] = Field(default=None, description="최종 출력")
    error_message: Optional[str] = Field(default=None, description="에러 메시지")

    # 타임스탬프
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # 완료 상태 집합 (성능 최적화)
    _TERMINAL_STAGES: frozenset = frozenset({WorkflowStage.COMPLETED, WorkflowStage.FAILED})

    def add_message(self, role: str, content: str, metadata: Optional[Dict] = None) -> None:
        """대화 기록에 메시지 추가 - 최적화된 버전"""
        self.conversation_history.append(
            ConversationMessage(
                role=role,
                content=content,
                metadata=metadata or {}
            )
        )
        self.updated_at = datetime.now(timezone.utc)

    def get_latest_execution(self) -> Optional[ExecutionResult]:
        """최근 실행 결과 반환 - O(1)"""
        return self.execution_history[-1] if self.execution_history else None

    def get_latest_reflection(self) -> Optional[ReflectionResult]:
        """최근 성찰 결과 반환 - O(1)"""
        return self.reflection_history[-1] if self.reflection_history else None

    def get_latest_decision(self) -> Optional[Decision]:
        """최근 결정 반환 - O(1)"""
        return self.decision_history[-1] if self.decision_history else None

    def should_continue(self, max_iterations: int = 10) -> bool:
        """워크플로우를 계속 진행해야 하는지 확인 - 최적화된 조건 검사"""
        return (
            self.current_stage not in self._TERMINAL_STAGES
            and self.iteration_count < max_iterations
        )

    @cached_property
    def has_plan(self) -> bool:
        """계획 존재 여부 (캐시됨)"""
        return self.current_plan is not None

    def to_context_string(self) -> str:
        """LLM 컨텍스트를 위한 문자열 생성 - 리스트 조인 최적화"""
        parts = [
            f"세션 ID: {self.session_id}",
            f"사용자 요청: {self.user_request}",
            f"현재 단계: {self.current_stage.value}",
            f"반복 횟수: {self.iteration_count}",
        ]

        if self.current_plan:
            parts.extend([
                f"현재 계획 목표: {self.current_plan.goal}",
                f"계획 단계 수: {len(self.current_plan.steps)}"
            ])

        if self.execution_history:
            parts.append(f"최근 실행 성공 여부: {self.execution_history[-1].success}")

        return "\n".join(parts)


class WorkflowConfig(BaseModel):
    """워크플로우 설정"""

    model_config = {"frozen": True}  # 불변 설정 객체

    max_iterations: int = Field(default=10, ge=1, le=100, description="최대 반복 횟수")
    max_reflection_iterations: int = Field(default=3, ge=1, le=10, description="최대 성찰 반복 횟수")
    quality_threshold: float = Field(default=0.7, ge=0.0, le=1.0, description="품질 임계값")
    enable_logging: bool = Field(default=True, description="로깅 활성화")
    enable_tracing: bool = Field(default=True, description="트레이싱 활성화")
