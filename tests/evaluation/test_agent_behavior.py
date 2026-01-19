# Copyright (c) Microsoft. All rights reserved.
"""
Agent Behavior Tests - CI/CD 파이프라인에서 실행되는 행동 기반 테스트
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import sys
sys.path.insert(0, str(__file__).replace("\\tests\\evaluation\\test_agent_behavior.py", "\\src"))

from langgraph_agent.models import (
    AgentState,
    Plan,
    PlanStep,
    ExecutionResult,
    ReflectionResult,
    Decision,
    DecisionType,
    WorkflowStage,
)
from langgraph_agent.config import AgentConfig


class TestAgentStateManagement:
    """에이전트 상태 관리 테스트"""

    def test_initial_state_creation(self):
        """초기 상태 생성 테스트"""
        state = AgentState(user_request="테스트 요청")

        assert state.user_request == "테스트 요청"
        assert state.current_stage == WorkflowStage.PLANNING
        assert state.iteration_count == 0
        assert state.reflection_count == 0
        assert state.current_plan is None
        assert len(state.execution_history) == 0

    def test_state_message_addition(self):
        """메시지 추가 테스트"""
        state = AgentState(user_request="테스트")
        state.add_message("user", "안녕하세요")
        state.add_message("assistant", "안녕하세요!")

        assert len(state.conversation_history) == 2
        assert state.conversation_history[0].role == "user"
        assert state.conversation_history[1].role == "assistant"

    def test_state_should_continue_normal(self):
        """정상 진행 조건 테스트"""
        state = AgentState(user_request="테스트")
        state.current_stage = WorkflowStage.EXECUTION
        state.iteration_count = 3

        assert state.should_continue(max_iterations=10) is True

    def test_state_should_not_continue_completed(self):
        """완료 시 중단 조건 테스트"""
        state = AgentState(user_request="테스트")
        state.current_stage = WorkflowStage.COMPLETED

        assert state.should_continue() is False

    def test_state_should_not_continue_max_iterations(self):
        """최대 반복 도달 시 중단 조건 테스트"""
        state = AgentState(user_request="테스트")
        state.iteration_count = 10

        assert state.should_continue(max_iterations=10) is False


class TestPlanManagement:
    """계획 관리 테스트"""

    def test_plan_creation(self):
        """계획 생성 테스트"""
        plan = Plan(
            goal="테스트 목표",
            steps=[
                PlanStep(step_number=1, description="단계 1", action="action1", expected_output="output1"),
                PlanStep(step_number=2, description="단계 2", action="action2", expected_output="output2"),
            ]
        )

        assert plan.goal == "테스트 목표"
        assert len(plan.steps) == 2
        assert plan.version == 1

    def test_plan_get_next_step(self):
        """다음 단계 가져오기 테스트"""
        plan = Plan(
            goal="테스트",
            steps=[
                PlanStep(step_number=1, description="단계 1", action="a1", expected_output="o1", status="completed"),
                PlanStep(step_number=2, description="단계 2", action="a2", expected_output="o2", status="pending"),
            ]
        )

        next_step = plan.get_next_step()
        assert next_step is not None
        assert next_step.step_number == 2

    def test_plan_get_next_step_all_completed(self):
        """모든 단계 완료 시 None 반환 테스트"""
        plan = Plan(
            goal="테스트",
            steps=[
                PlanStep(step_number=1, description="단계 1", action="a1", expected_output="o1", status="completed"),
            ]
        )

        assert plan.get_next_step() is None

    def test_plan_is_complete(self):
        """계획 완료 확인 테스트"""
        plan = Plan(
            goal="테스트",
            steps=[
                PlanStep(step_number=1, description="단계 1", action="a1", expected_output="o1", status="completed"),
                PlanStep(step_number=2, description="단계 2", action="a2", expected_output="o2", status="completed"),
            ]
        )

        assert plan.is_complete() is True


class TestReflectionLogic:
    """성찰 로직 테스트"""

    def test_reflection_result_creation(self):
        """성찰 결과 생성 테스트"""
        reflection = ReflectionResult(
            execution_id="exec_123",
            quality_score=0.85,
            goal_alignment=0.9,
            completeness=0.8,
            strengths=["잘된 점 1"],
            weaknesses=["개선점 1"],
            suggestions=["제안 1"],
            needs_retry=False,
            needs_replan=False,
            reasoning="좋은 결과입니다",
        )

        assert reflection.quality_score == 0.85
        assert not reflection.needs_retry
        assert not reflection.needs_replan

    def test_reflection_low_quality_triggers_retry(self):
        """낮은 품질 점수로 재시도 플래그 테스트"""
        reflection = ReflectionResult(
            execution_id="exec_123",
            quality_score=0.3,
            goal_alignment=0.4,
            completeness=0.3,
            needs_retry=True,
            needs_replan=False,
            reasoning="품질이 낮습니다",
        )

        assert reflection.needs_retry is True


class TestDecisionLogic:
    """결정 로직 테스트"""

    def test_decision_complete(self):
        """완료 결정 테스트"""
        decision = Decision(
            decision_type=DecisionType.COMPLETE,
            reasoning="목표 달성",
            next_action="complete",
            confidence=0.95,
        )

        assert decision.decision_type == DecisionType.COMPLETE
        assert decision.confidence >= 0.9

    def test_decision_replan(self):
        """재계획 결정 테스트"""
        decision = Decision(
            decision_type=DecisionType.REPLAN,
            reasoning="계획 수정 필요",
            next_action="replan",
            confidence=0.7,
            feedback_for_planning="더 상세한 계획 필요",
        )

        assert decision.decision_type == DecisionType.REPLAN
        assert decision.feedback_for_planning is not None

    def test_decision_retry(self):
        """재시도 결정 테스트"""
        decision = Decision(
            decision_type=DecisionType.RETRY,
            reasoning="실행 재시도 필요",
            next_action="retry",
            confidence=0.6,
            feedback_for_execution="더 정확한 실행 필요",
        )

        assert decision.decision_type == DecisionType.RETRY
        assert decision.feedback_for_execution is not None


class TestWorkflowStageTransitions:
    """워크플로우 단계 전환 테스트"""

    def test_planning_to_execution(self):
        """Planning → Execution 전환 테스트"""
        state = AgentState(user_request="테스트")
        state.current_stage = WorkflowStage.PLANNING

        # 계획 생성 시뮬레이션
        state.current_plan = Plan(
            goal="테스트 목표",
            steps=[PlanStep(step_number=1, description="단계", action="act", expected_output="out")]
        )
        state.current_stage = WorkflowStage.EXECUTION

        assert state.current_stage == WorkflowStage.EXECUTION
        assert state.current_plan is not None

    def test_execution_to_reflection(self):
        """Execution → Reflection 전환 테스트"""
        state = AgentState(user_request="테스트")
        state.current_stage = WorkflowStage.EXECUTION

        # 실행 완료 시뮬레이션
        state.execution_history.append(
            ExecutionResult(
                step_id="step_1",
                success=True,
                output="실행 결과",
                duration_seconds=1.5,
            )
        )
        state.current_stage = WorkflowStage.REFLECTION

        assert state.current_stage == WorkflowStage.REFLECTION
        assert len(state.execution_history) == 1

    def test_reflection_to_decision(self):
        """Reflection → Decision 전환 테스트"""
        state = AgentState(user_request="테스트")
        state.current_stage = WorkflowStage.REFLECTION

        # 성찰 완료 시뮬레이션
        state.reflection_history.append(
            ReflectionResult(
                execution_id="exec_1",
                quality_score=0.85,
                goal_alignment=0.9,
                completeness=0.8,
                reasoning="좋은 결과",
            )
        )
        state.current_stage = WorkflowStage.DECISION

        assert state.current_stage == WorkflowStage.DECISION
        assert len(state.reflection_history) == 1

    def test_decision_to_complete(self):
        """Decision → Complete 전환 테스트"""
        state = AgentState(user_request="테스트")
        state.current_stage = WorkflowStage.DECISION

        # 완료 결정 시뮬레이션
        state.decision_history.append(
            Decision(
                decision_type=DecisionType.COMPLETE,
                reasoning="목표 달성",
                next_action="complete",
                confidence=0.95,
            )
        )
        state.current_stage = WorkflowStage.COMPLETED
        state.final_output = "최종 결과"

        assert state.current_stage == WorkflowStage.COMPLETED
        assert state.final_output is not None


class TestConfigValidation:
    """설정 유효성 테스트"""

    def test_config_defaults(self):
        """기본 설정값 테스트"""
        config = AgentConfig()

        assert config.max_reflection_iterations == 3
        assert config.max_planning_depth == 5
        assert config.execution_timeout_seconds == 300

    def test_config_foundry_check(self):
        """Foundry 설정 확인 테스트"""
        # frozen=True이므로 생성자에서 값 설정
        config = AgentConfig(azure_foundry_project_endpoint="https://test.ai.azure.com")

        assert config.is_foundry_configured() is True

    def test_config_openai_check(self):
        """OpenAI 설정 확인 테스트"""
        # frozen=True이므로 생성자에서 값 설정
        config = AgentConfig(azure_openai_endpoint="https://test.openai.azure.com")

        assert config.is_openai_configured() is True


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
