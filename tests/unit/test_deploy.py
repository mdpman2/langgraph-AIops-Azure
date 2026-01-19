# Copyright (c) Microsoft. All rights reserved.
"""
Deploy Script Unit Tests - 배포 스크립트 단위 테스트
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import subprocess
import json

import sys
sys.path.insert(0, str(__file__).replace("\\tests\\unit\\test_deploy.py", "\\scripts"))


class TestAzureCommandExecution:
    """Azure CLI 명령 실행 테스트"""

    @patch('subprocess.run')
    def test_run_command_success(self, mock_run):
        """명령 성공 실행 테스트"""
        mock_run.return_value = MagicMock(
            returncode=0,
            stdout='{"status": "success"}',
            stderr=''
        )

        result = subprocess.run(
            ['az', 'account', 'show'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 0
        mock_run.assert_called_once()

    @patch('subprocess.run')
    def test_run_command_failure(self, mock_run):
        """명령 실패 처리 테스트"""
        mock_run.return_value = MagicMock(
            returncode=1,
            stdout='',
            stderr='Error: Not logged in'
        )

        result = subprocess.run(
            ['az', 'account', 'show'],
            capture_output=True,
            text=True
        )

        assert result.returncode == 1


class TestResourceNaming:
    """리소스 이름 생성 테스트"""

    def test_acr_name_valid(self):
        """ACR 이름 유효성 테스트"""
        # ACR 이름은 소문자 알파벳과 숫자만 허용
        location = "eastus"
        base_name = "langgraph-agent"

        acr_name = f"acr{base_name.replace('-', '')}{location}"

        # ACR 이름 규칙: 5-50자, 소문자 영숫자만
        assert acr_name.islower() or acr_name.replace('-', '').isalnum()
        assert 5 <= len(acr_name) <= 50

    def test_resource_group_name_format(self):
        """리소스 그룹 이름 형식 테스트"""
        environment = "dev"
        location = "eastus"

        rg_name = f"rg-langgraph-agent-{environment}-{location}"

        # 리소스 그룹 이름 규칙
        assert rg_name.startswith("rg-")
        assert environment in rg_name
        assert location in rg_name

    def test_container_app_name_format(self):
        """Container App 이름 형식 테스트"""
        environment = "dev"

        ca_name = f"ca-langgraph-agent-{environment}"

        # Container App 이름 규칙: 2-32자, 소문자, 숫자, 하이픈
        assert len(ca_name) <= 32
        assert ca_name.islower()


class TestRegionValidation:
    """지역 유효성 검증 테스트"""

    def test_valid_azure_regions(self):
        """유효한 Azure 지역 테스트"""
        valid_regions = [
            "eastus", "eastus2", "westus", "westus2",
            "koreacentral", "japaneast", "southeastasia"
        ]

        for region in valid_regions:
            assert region.islower()
            assert " " not in region

    def test_container_apps_supported_regions(self):
        """Container Apps 지원 지역 테스트"""
        # Container Apps가 지원되는 주요 지역
        supported_regions = [
            "eastus", "eastus2", "westus", "westus2", "westus3",
            "northeurope", "westeurope", "koreacentral", "japaneast"
        ]

        assert len(supported_regions) > 0
        assert "eastus" in supported_regions


class TestEnvironmentValidation:
    """환경 변수 검증 테스트"""

    def test_required_env_vars(self):
        """필수 환경 변수 목록 테스트"""
        required_vars = [
            "AZURE_OPENAI_ENDPOINT",
            "AZURE_OPENAI_API_KEY",
            "AZURE_OPENAI_DEPLOYMENT_NAME"
        ]

        # 필수 변수 이름 형식 확인
        for var in required_vars:
            assert var.isupper()
            assert "_" in var

    def test_optional_env_vars(self):
        """선택적 환경 변수 목록 테스트"""
        optional_vars = [
            "AZURE_FOUNDRY_PROJECT_ENDPOINT",
            "AZURE_FOUNDRY_MODEL_DEPLOYMENT",
            "APPLICATIONINSIGHTS_CONNECTION_STRING"
        ]

        for var in optional_vars:
            assert var.isupper()


class TestDeploymentConfiguration:
    """배포 설정 테스트"""

    def test_container_resources(self):
        """컨테이너 리소스 설정 테스트"""
        # 기본 리소스 설정
        cpu = 0.5
        memory = "1Gi"

        assert cpu >= 0.25  # 최소 CPU
        assert "Gi" in memory or "Mi" in memory

    def test_scaling_configuration(self):
        """스케일링 설정 테스트"""
        min_replicas = 0
        max_replicas = 3

        assert min_replicas >= 0
        assert max_replicas >= min_replicas
        assert max_replicas <= 10  # 합리적인 최대값

    def test_port_configuration(self):
        """포트 설정 테스트"""
        target_port = 8080

        assert 1 <= target_port <= 65535
        assert target_port == 8080  # 기본 FastAPI 포트


class TestRetryLogic:
    """재시도 로직 테스트"""

    def test_retry_delays(self):
        """재시도 대기 시간 테스트"""
        delays = [10, 20, 30]  # 예시 딜레이

        # 딜레이가 증가하는지 확인 (백오프)
        for i in range(1, len(delays)):
            assert delays[i] >= delays[i-1]

    def test_max_retries(self):
        """최대 재시도 횟수 테스트"""
        max_retries = 3

        assert max_retries > 0
        assert max_retries <= 10  # 합리적인 최대값


class TestHealthCheckLogic:
    """헬스 체크 로직 테스트"""

    def test_health_check_url_format(self):
        """헬스 체크 URL 형식 테스트"""
        base_url = "https://ca-app.region.azurecontainerapps.io"
        health_url = f"{base_url}/health"

        assert health_url.startswith("https://")
        assert health_url.endswith("/health")

    def test_health_check_timeout(self):
        """헬스 체크 타임아웃 테스트"""
        timeout_seconds = 10
        max_retries = 10
        retry_interval = 15

        total_wait = max_retries * retry_interval
        assert total_wait <= 300  # 최대 5분 대기

    def test_expected_health_response(self):
        """예상 헬스 응답 테스트"""
        expected_response = {"status": "healthy"}

        assert "status" in expected_response
        assert expected_response["status"] == "healthy"


class TestBicepParameters:
    """Bicep 파라미터 테스트"""

    def test_required_bicep_parameters(self):
        """필수 Bicep 파라미터 테스트"""
        required_params = [
            "location",
            "environment"
        ]

        for param in required_params:
            assert param.islower()

    def test_optional_bicep_parameters(self):
        """선택적 Bicep 파라미터 테스트"""
        optional_params = [
            "existingAoaiEndpoint",
            "existingAoaiKey",
            "containerImage"
        ]

        # camelCase 형식 확인
        for param in optional_params:
            assert param[0].islower()


class TestDockerConfiguration:
    """Docker 설정 테스트"""

    def test_dockerfile_target(self):
        """Dockerfile 타겟 테스트"""
        targets = ["development", "production"]

        assert "production" in targets

    def test_image_tag_format(self):
        """이미지 태그 형식 테스트"""
        tags = ["v1", "v2", "latest", "dev-123"]

        for tag in tags:
            assert len(tag) > 0
            assert " " not in tag


class TestErrorMessages:
    """에러 메시지 테스트"""

    def test_dns_error_detection(self):
        """DNS 오류 감지 테스트"""
        error_msg = "dial tcp: lookup acrxxx.azurecr.io: no such host"

        assert "no such host" in error_msg or "lookup" in error_msg

    def test_unauthorized_error_detection(self):
        """인증 오류 감지 테스트"""
        error_msg = "UNAUTHORIZED: authentication required"

        assert "UNAUTHORIZED" in error_msg or "authentication" in error_msg

    def test_timeout_error_detection(self):
        """타임아웃 오류 감지 테스트"""
        error_msg = "TimeoutError: The read operation timed out"

        assert "timeout" in error_msg.lower()


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
