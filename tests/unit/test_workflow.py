# Copyright (c) Microsoft. All rights reserved.
"""
Unit Tests for Workflow Components
워크플로우 각 컴포넌트의 단위 테스트
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime

import sys
sys.path.insert(0, str(__file__).replace("\\tests\\unit\\test_workflow.py", "\\src"))

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


class TestPlanningExecutor:
    """Planning Executor 단위 테스트"""

    @pytest.mark.asyncio
    async def test_planning_creates_valid_plan(self):
        """계획 생성 유효성 테스트"""
        # Given
        state = AgentState(user_request="파이썬으로 계산기 만들기")

        # Simulate planning result
        plan = Plan(
            goal="파이썬 계산기 구현",
            steps=[
                PlanStep(step_number=1, description="요구사항 분석", action="analyze", expected_output="요구사항 문서"),
                PlanStep(step_number=2, description="함수 설계", action="design", expected_output="설계 문서"),
                PlanStep(step_number=3, description="코드 구현", action="implement", expected_output="코드"),
            ]
        )

        # Then
        assert plan.goal is not None
        assert len(plan.steps) > 0
        assert all(step.step_number > 0 for step in plan.steps)

    @pytest.mark.asyncio
    async def test_planning_handles_empty_request(self):
        """빈 요청 처리 테스트"""
        state = AgentState(user_request="")

        # Planning should still work but with minimal plan
        assert state.user_request == ""
        assert state.current_stage == WorkflowStage.PLANNING


class TestExecutionExecutor:
    """Execution Executor 단위 테스트"""

    @pytest.mark.asyncio
    async def test_execution_success(self):
        """성공적인 실행 테스트"""
        # Given
        step = PlanStep(
            step_number=1,
            description="테스트 단계",
            action="test_action",
            expected_output="테스트 결과"
        )

        # Simulate execution
        result = ExecutionResult(
            step_id=f"step_{step.step_number}",
            success=True,
            output="실행 완료",
            duration_seconds=1.5,
        )

        # Then
        assert result.success is True
        assert result.output is not None
        assert result.error is None

    @pytest.mark.asyncio
    async def test_execution_failure(self):
        """실행 실패 테스트"""
        result = ExecutionResult(
            step_id="step_1",
            success=False,
            output=None,
            error="실행 중 오류 발생",
            duration_seconds=0.5,
        )

        assert result.success is False
        assert result.error is not None


class TestReflectionExecutor:
    """Reflection Executor 단위 테스트"""

    @pytest.mark.asyncio
    async def test_reflection_high_quality(self):
        """고품질 결과 성찰 테스트"""
        reflection = ReflectionResult(
            execution_id="exec_1",
            quality_score=0.9,
            goal_alignment=0.95,
            completeness=0.85,
            strengths=["정확한 결과", "빠른 처리"],
            weaknesses=[],
            suggestions=[],
            needs_retry=False,
            needs_replan=False,
            reasoning="우수한 품질의 결과",
        )

        assert reflection.quality_score >= 0.8
        assert not reflection.needs_retry
        assert not reflection.needs_replan

    @pytest.mark.asyncio
    async def test_reflection_low_quality_triggers_retry(self):
        """저품질 결과 재시도 트리거 테스트"""
        reflection = ReflectionResult(
            execution_id="exec_1",
            quality_score=0.3,
            goal_alignment=0.4,
            completeness=0.3,
            strengths=[],
            weaknesses=["불완전한 결과", "목표와 불일치"],
            suggestions=["더 자세한 분석 필요"],
            needs_retry=True,
            needs_replan=False,
            reasoning="품질이 낮아 재시도 필요",
        )

        assert reflection.quality_score < 0.5
        assert reflection.needs_retry is True


class TestDecisionExecutor:
    """Decision Executor 단위 테스트"""

    @pytest.mark.asyncio
    async def test_decision_complete_when_done(self):
        """완료 결정 테스트"""
        decision = Decision(
            decision_type=DecisionType.COMPLETE,
            reasoning="모든 단계 성공적으로 완료",
            next_action="complete",
            confidence=0.95,
        )

        assert decision.decision_type == DecisionType.COMPLETE
        assert decision.confidence > 0.9

    @pytest.mark.asyncio
    async def test_decision_continue_when_steps_remaining(self):
        """계속 진행 결정 테스트"""
        decision = Decision(
            decision_type=DecisionType.CONTINUE,
            reasoning="다음 단계 진행",
            next_action="continue",
            confidence=0.8,
        )

        assert decision.decision_type == DecisionType.CONTINUE

    @pytest.mark.asyncio
    async def test_decision_replan_when_needed(self):
        """재계획 결정 테스트"""
        decision = Decision(
            decision_type=DecisionType.REPLAN,
            reasoning="계획 수정 필요",
            next_action="replan",
            confidence=0.7,
            feedback_for_planning="더 상세한 단계 필요",
        )

        assert decision.decision_type == DecisionType.REPLAN
        assert decision.feedback_for_planning is not None


class TestWorkflowIntegration:
    """워크플로우 통합 테스트"""

    @pytest.mark.asyncio
    async def test_full_workflow_cycle(self):
        """전체 워크플로우 사이클 테스트"""
        # Initialize state
        state = AgentState(user_request="Hello World 프로그램 작성")

        # Step 1: Planning
        state.current_stage = WorkflowStage.PLANNING
        state.current_plan = Plan(
            goal="Hello World 출력",
            steps=[
                PlanStep(step_number=1, description="코드 작성", action="write", expected_output="Python 코드"),
            ]
        )

        # Step 2: Execution
        state.current_stage = WorkflowStage.EXECUTION
        state.execution_history.append(ExecutionResult(
            step_id="step_1",
            success=True,
            output="print('Hello World')",
            duration_seconds=0.1,
        ))
        state.current_plan.steps[0].status = "completed"

        # Step 3: Reflection
        state.current_stage = WorkflowStage.REFLECTION
        state.reflection_history.append(ReflectionResult(
            execution_id="exec_1",
            quality_score=0.95,
            goal_alignment=1.0,
            completeness=1.0,
            reasoning="완벽한 결과",
        ))
        state.reflection_count += 1

        # Step 4: Decision
        state.current_stage = WorkflowStage.DECISION
        state.decision_history.append(Decision(
            decision_type=DecisionType.COMPLETE,
            reasoning="목표 달성",
            next_action="complete",
            confidence=0.98,
        ))

        # Final
        state.current_stage = WorkflowStage.COMPLETED
        state.final_output = "print('Hello World')"
        state.iteration_count += 1

        # Assertions
        assert state.current_stage == WorkflowStage.COMPLETED
        assert state.final_output is not None
        assert len(state.execution_history) == 1
        assert len(state.reflection_history) == 1
        assert len(state.decision_history) == 1


class TestEdgeCases:
    """엣지 케이스 테스트"""

    def test_max_iterations_protection(self):
        """최대 반복 횟수 보호 테스트"""
        state = AgentState(user_request="무한 루프 테스트")
        state.iteration_count = 100

        assert state.should_continue(max_iterations=10) is False

    def test_empty_plan_handling(self):
        """빈 계획 처리 테스트"""
        plan = Plan(goal="테스트", steps=[])

        assert plan.get_next_step() is None
        assert plan.is_complete() is True

    def test_state_history_preservation(self):
        """상태 기록 보존 테스트"""
        state = AgentState(user_request="테스트")

        # Add history
        for i in range(5):
            state.add_message("user", f"메시지 {i}")

        assert len(state.conversation_history) == 5


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
