# Copyright (c) Microsoft. All rights reserved.
"""
Configuration settings for the LangGraph-style Agent Framework
100% Azure 기반 설정

최적화:
- 싱글톤 패턴으로 설정 캐싱
- 불변 설정 객체 (frozen)
- 지연 검증 (lazy validation)
- 환경 변수 캐싱
"""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Optional

from pydantic import Field, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict


class AgentConfig(BaseSettings):
    """Agent configuration using Pydantic Settings - Azure 전용 (최적화)"""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        frozen=True,  # 불변 객체로 설정하여 해시 가능하게 함
        validate_default=True,
    )

    # ============================================
    # Azure AI Foundry Configuration (Primary)
    # ============================================
    azure_foundry_project_endpoint: str = Field(
        default="",
        description="Azure AI Foundry project endpoint",
        alias="AZURE_FOUNDRY_PROJECT_ENDPOINT"
    )
    azure_foundry_model_deployment: str = Field(
        default="gpt-4.1",
        description="Model deployment name in Azure AI Foundry",
        alias="AZURE_FOUNDRY_MODEL_DEPLOYMENT"
    )

    # ============================================
    # Azure OpenAI Configuration (Alternative)
    # ============================================
    azure_openai_endpoint: str = Field(
        default="",
        description="Azure OpenAI endpoint",
        alias="AZURE_OPENAI_ENDPOINT"
    )
    azure_openai_api_key: Optional[str] = Field(
        default=None,
        description="Azure OpenAI API key",
        alias="AZURE_OPENAI_API_KEY",
        repr=False,  # 로그에서 키 숨김
    )
    azure_openai_deployment_name: str = Field(
        default="gpt-4.1",
        description="Azure OpenAI deployment name",
        alias="AZURE_OPENAI_DEPLOYMENT_NAME"
    )

    # ============================================
    # Agent Workflow Configuration
    # ============================================
    max_reflection_iterations: int = Field(
        default=3,
        description="Maximum number of reflection iterations",
        alias="MAX_REFLECTION_ITERATIONS"
    )
    max_planning_depth: int = Field(
        default=5,
        description="Maximum planning depth",
        alias="MAX_PLANNING_DEPTH"
    )
    execution_timeout_seconds: int = Field(
        default=300,
        description="Execution timeout in seconds",
        alias="EXECUTION_TIMEOUT_SECONDS"
    )

    # ============================================
    # Azure Blob Storage (State Persistence)
    # ============================================
    azure_storage_connection_string: Optional[str] = Field(
        default=None,
        description="Azure Storage connection string for state persistence",
        alias="AZURE_STORAGE_CONNECTION_STRING",
        repr=False,  # 로그에서 연결 문자열 숨김
    )
    azure_storage_container_name: str = Field(
        default="agent-state",
        description="Azure Storage container name for state",
        alias="AZURE_STORAGE_CONTAINER_NAME"
    )
    azure_eval_results_container: str = Field(
        default="evaluation-results",
        description="Azure Storage container for evaluation results",
        alias="AZURE_EVAL_RESULTS_CONTAINER"
    )

    # ============================================
    # Azure Cosmos DB (Conversation History)
    # ============================================
    azure_cosmos_endpoint: Optional[str] = Field(
        default=None,
        description="Azure Cosmos DB endpoint",
        alias="AZURE_COSMOS_ENDPOINT"
    )
    azure_cosmos_key: Optional[str] = Field(
        default=None,
        description="Azure Cosmos DB key",
        alias="AZURE_COSMOS_KEY",
        repr=False,  # 로그에서 키 숨김
    )
    azure_cosmos_database: str = Field(
        default="agent-db",
        description="Cosmos DB database name",
        alias="AZURE_COSMOS_DATABASE"
    )
    azure_cosmos_container: str = Field(
        default="conversations",
        description="Cosmos DB container name",
        alias="AZURE_COSMOS_CONTAINER"
    )

    # ============================================
    # Azure Key Vault (Secrets Management)
    # ============================================
    azure_keyvault_endpoint: Optional[str] = Field(
        default=None,
        description="Azure Key Vault endpoint",
        alias="AZURE_KEYVAULT_ENDPOINT"
    )

    # ============================================
    # Azure Service Bus (Async Messaging)
    # ============================================
    azure_servicebus_connection_string: Optional[str] = Field(
        default=None,
        description="Azure Service Bus connection string",
        alias="AZURE_SERVICEBUS_CONNECTION_STRING",
        repr=False,  # 로그에서 연결 문자열 숨김
    )
    azure_servicebus_namespace: Optional[str] = Field(
        default=None,
        description="Azure Service Bus namespace",
        alias="AZURE_SERVICEBUS_NAMESPACE"
    )
    azure_servicebus_queue_name: str = Field(
        default="agent-tasks",
        description="Azure Service Bus queue name",
        alias="AZURE_SERVICEBUS_QUEUE_NAME"
    )

    # ============================================
    # Azure AI Search (RAG)
    # ============================================
    azure_search_endpoint: Optional[str] = Field(
        default=None,
        description="Azure AI Search endpoint",
        alias="AZURE_SEARCH_ENDPOINT"
    )
    azure_search_index_name: str = Field(
        default="agent-knowledge",
        description="Azure AI Search index name",
        alias="AZURE_SEARCH_INDEX_NAME"
    )

    # ============================================
    # Azure Application Insights (Observability)
    # ============================================
    applicationinsights_connection_string: Optional[str] = Field(
        default=None,
        description="Application Insights connection string",
        alias="APPLICATIONINSIGHTS_CONNECTION_STRING",
        repr=False,  # 로그에서 연결 문자열 숨김
    )

    # ============================================
    # Azure Content Safety
    # ============================================
    azure_content_safety_endpoint: Optional[str] = Field(
        default=None,
        description="Azure Content Safety endpoint",
        alias="AZURE_CONTENT_SAFETY_ENDPOINT"
    )

    # 기존 Config 클래스는 model_config로 대체됨

    def is_foundry_configured(self) -> bool:
        """Check if Azure AI Foundry is configured"""
        return bool(self.azure_foundry_project_endpoint)

    def is_openai_configured(self) -> bool:
        """Check if Azure OpenAI is configured"""
        return bool(self.azure_openai_endpoint)

    def is_cosmos_configured(self) -> bool:
        """Check if Azure Cosmos DB is configured"""
        return bool(self.azure_cosmos_endpoint)

    def is_storage_configured(self) -> bool:
        """Check if Azure Storage is configured"""
        return bool(self.azure_storage_connection_string)

    def is_search_configured(self) -> bool:
        """Check if Azure AI Search is configured"""
        return bool(self.azure_search_endpoint)

    def is_servicebus_configured(self) -> bool:
        """Check if Azure Service Bus is configured"""
        return bool(self.azure_servicebus_connection_string or self.azure_servicebus_namespace)

    @model_validator(mode='after')
    def validate_ai_service(self) -> 'AgentConfig':
        """최소 하나의 AI 서비스가 설정되었는지 검증"""
        if not self.azure_foundry_project_endpoint and not self.azure_openai_endpoint:
            import warnings
            warnings.warn(
                "Azure AI Foundry 또는 Azure OpenAI 중 하나 이상이 설정되어야 합니다. "
                "AZURE_FOUNDRY_PROJECT_ENDPOINT 또는 AZURE_OPENAI_ENDPOINT를 설정하세요.",
                UserWarning,
                stacklevel=2
            )
        return self


# 싱글톤 패턴으로 설정 캐싱
@lru_cache(maxsize=1)
def load_config() -> AgentConfig:
    """Load configuration from environment variables (캐시됨)"""
    from dotenv import load_dotenv
    load_dotenv()
    return AgentConfig()


def reload_config() -> AgentConfig:
    """설정 강제 리로드"""
    load_config.cache_clear()
    return load_config()
