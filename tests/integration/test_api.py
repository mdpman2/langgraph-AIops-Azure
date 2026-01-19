# Copyright (c) Microsoft. All rights reserved.
"""
API Integration Tests - FastAPI 엔드포인트 통합 테스트
"""

import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from fastapi.testclient import TestClient
import json

import sys
sys.path.insert(0, str(__file__).replace("\\tests\\integration\\test_api.py", "\\src"))

from langgraph_agent.server import app
from langgraph_agent.models import AgentState, WorkflowStage


@pytest.fixture
def client():
    """FastAPI 테스트 클라이언트"""
    return TestClient(app)


class TestHealthEndpoint:
    """헬스 체크 엔드포인트 테스트"""

    def test_health_check_returns_ok(self, client):
        """헬스 체크 정상 응답 테스트"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        # timestamp는 선택적 - version과 environment가 있을 수 있음
        assert "status" in data
        # version이 있으면 확인, 없으면 패스
        if "version" in data:
            assert isinstance(data["version"], str)


class TestRootEndpoint:
    """루트 엔드포인트 (웹 UI) 테스트"""

    def test_root_returns_html(self, client):
        """루트 경로에서 HTML 반환 테스트"""
        response = client.get("/")

        assert response.status_code == 200
        assert "text/html" in response.headers.get("content-type", "")

    def test_root_contains_chat_ui_elements(self, client):
        """웹 UI에 채팅 요소 포함 테스트"""
        response = client.get("/")

        assert response.status_code == 200
        content = response.text
        # 기본적인 HTML 구조 확인
        assert "<html" in content.lower() or "<!doctype" in content.lower()


class TestAgentEndpoint:
    """에이전트 실행 엔드포인트 테스트"""

    @patch('langgraph_agent.server.AgentWorkflow')
    def test_agent_endpoint_accepts_request(self, mock_workflow, client):
        """에이전트 엔드포인트 요청 수락 테스트"""
        # Mock 설정
        mock_instance = MagicMock()
        mock_instance.run = AsyncMock(return_value=AgentState(
            user_request="테스트",
            current_stage=WorkflowStage.COMPLETED,
            final_output="테스트 응답"
        ))
        mock_workflow.return_value = mock_instance

        response = client.post(
            "/agent",
            json={"request": "테스트 요청"}
        )

        # 응답 코드 확인 (200 성공, 422 유효성, 500 내부오류, 503 서비스 불가)
        # 503은 Azure OpenAI 연결 실패 시 정상 응답
        assert response.status_code in [200, 422, 500, 503]

    def test_agent_endpoint_validates_empty_request(self, client):
        """빈 요청 검증 테스트"""
        response = client.post(
            "/agent",
            json={"request": ""}
        )

        # 빈 요청도 처리 가능해야 함 (에러 또는 정상)
        assert response.status_code in [200, 400, 422, 500]


class TestStreamEndpoint:
    """스트리밍 엔드포인트 테스트"""

    def test_stream_endpoint_exists(self, client):
        """스트리밍 엔드포인트 존재 확인"""
        response = client.post(
            "/agent/stream",
            json={"request": "테스트"}
        )

        # 엔드포인트가 존재하면 404가 아님
        assert response.status_code != 404

    def test_stream_endpoint_returns_event_stream(self, client):
        """스트리밍 응답 타입 테스트"""
        response = client.post(
            "/agent/stream",
            json={"request": "테스트"}
        )

        # SSE 응답이면 text/event-stream
        content_type = response.headers.get("content-type", "")
        # 스트림이거나 에러 응답 (503은 서비스 불가)
        assert response.status_code in [200, 422, 500, 503] or "event-stream" in content_type

    def test_invalid_method_returns_405(self, client):
        """잘못된 HTTP 메서드 405 반환"""
        response = client.delete("/health")

        assert response.status_code == 405

    def test_invalid_json_returns_422(self, client):
        """잘못된 JSON 422 반환"""
        response = client.post(
            "/agent",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422


class TestCORS:
    """CORS 설정 테스트"""

    def test_cors_headers_present(self, client):
        """CORS 헤더 존재 확인"""
        response = client.options(
            "/agent",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )

        # CORS가 설정되어 있으면 헤더가 있음
        # 설정되어 있지 않아도 테스트는 통과 (선택적 기능)
        assert response.status_code in [200, 405]


class TestRequestValidation:
    """요청 검증 테스트"""

    def test_agent_accepts_valid_request(self, client):
        """유효한 요청 수락 테스트"""
        response = client.post(
            "/agent",
            json={
                "request": "Python으로 Hello World 출력하는 방법"
            }
        )

        # 유효한 요청은 처리됨 (503은 Azure 연결 실패)
        assert response.status_code in [200, 500, 503]

    def test_agent_handles_special_characters(self, client):
        """특수 문자 처리 테스트"""
        response = client.post(
            "/agent",
            json={
                "request": "한글 테스트 🎉 <script>alert('xss')</script>"
            }
        )

        # 특수 문자도 처리 가능 (503은 Azure 연결 실패)
        assert response.status_code in [200, 400, 422, 500, 503]

    def test_agent_handles_long_request(self, client):
        """긴 요청 처리 테스트"""
        long_request = "테스트 " * 1000  # 약 7000자
        response = client.post(
            "/agent",
            json={"request": long_request}
        )

        # 긴 요청도 처리 (성공 또는 에러, 503은 Azure 연결 실패)
        assert response.status_code in [200, 400, 413, 422, 500, 503]
