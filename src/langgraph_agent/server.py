# Copyright (c) Microsoft. All rights reserved.
"""
FastAPI Web Server for LangGraph-style AI Agent
Container Apps 배포용 HTTP API
"""

from __future__ import annotations

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional, AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, Field

from .workflow import AgentWorkflow
from .config import load_config

logger = structlog.get_logger(__name__)

# ============================================
# Request/Response Models
# ============================================

class AgentRequest(BaseModel):
    """에이전트 요청 모델"""
    request: str = Field(..., description="사용자 요청", min_length=1)
    timeout: float = Field(default=300.0, description="타임아웃(초)", ge=10, le=600)


class AgentResponse(BaseModel):
    """에이전트 응답 모델"""
    success: bool
    result: Optional[str] = None
    error: Optional[str] = None
    session_id: Optional[str] = None


class HealthResponse(BaseModel):
    """헬스체크 응답"""
    status: str
    version: str = "1.0.0"
    environment: str


# ============================================
# Global State
# ============================================

_config = None
_workflow: Optional[AgentWorkflow] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """FastAPI 라이프사이클 관리"""
    global _config, _workflow

    logger.info("server_starting")

    # 설정 로드
    _config = load_config()
    _workflow = AgentWorkflow(_config)

    logger.info("server_ready")
    yield

    # 종료 시 정리
    if _workflow:
        await _workflow.close()
    logger.info("server_stopped")


# ============================================
# FastAPI App
# ============================================

app = FastAPI(
    title="LangGraph AI Agent API",
    description="Planning-Execution-Reflection-Decision 워크플로우 기반 AI 에이전트",
    version="1.0.0",
    lifespan=lifespan,
)

# 정적 파일 디렉토리
STATIC_DIR = Path(__file__).parent / "static"

# CORS 설정 - 환경에 따라 분기
_allowed_origins = os.getenv("CORS_ALLOWED_ORIGINS", "*").split(",")
app.add_middleware(
    CORSMiddleware,
    allow_origins=_allowed_origins if _allowed_origins != ["*"] else ["*"],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type", "Authorization"],
    max_age=600,  # preflight 캐시 10분
)

# 정적 파일 마운트 (CSS, JS 등)
if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


# ============================================
# Endpoints
# ============================================

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """헬스체크 엔드포인트"""
    return HealthResponse(
        status="healthy",
        version="1.0.0",
        environment=os.getenv("ENVIRONMENT", "dev")
    )


@app.get("/", response_class=FileResponse)
async def root():
    """메인 UI 페이지"""
    return FileResponse(STATIC_DIR / "index.html")


@app.post("/agent", response_model=AgentResponse)
async def run_agent_api(request: AgentRequest):
    """에이전트 실행 엔드포인트"""
    global _workflow

    if not _workflow:
        raise HTTPException(status_code=503, detail="서비스가 준비되지 않았습니다.")

    try:
        logger.info("agent_request_received", request=request.request[:100])

        # 타임아웃 적용하여 실행
        result = await asyncio.wait_for(
            _workflow.run(request.request),
            timeout=request.timeout
        )

        return AgentResponse(
            success=True,
            result=result,
            session_id=_workflow._current_session_id if hasattr(_workflow, '_current_session_id') else None
        )

    except asyncio.TimeoutError:
        logger.warning("agent_timeout", timeout=request.timeout)
        return AgentResponse(
            success=False,
            error=f"요청이 {request.timeout}초 내에 완료되지 않았습니다."
        )
    except Exception as e:
        logger.exception("agent_error", error=str(e))
        return AgentResponse(
            success=False,
            error=str(e)
        )


async def event_stream(request: str) -> AsyncGenerator[str, None]:
    """SSE 이벤트 스트림 생성"""
    global _workflow

    try:
        async for event in _workflow.run_stream(request):
            yield f"data: {json.dumps(event, ensure_ascii=False)}\n\n"
    except Exception as e:
        logger.exception("stream_error", error=str(e))
        yield f"data: {json.dumps({'type': 'error', 'message': str(e)}, ensure_ascii=False)}\n\n"


@app.post("/agent/stream")
async def run_agent_stream(request: AgentRequest):
    """스트리밍 에이전트 실행 엔드포인트 (SSE)"""
    global _workflow

    if not _workflow:
        raise HTTPException(status_code=503, detail="서비스가 준비되지 않았습니다.")

    logger.info("agent_stream_request_received", request=request.request[:100])

    return StreamingResponse(
        event_stream(request.request),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        }
    )


# ============================================
# Server Entry Point
# ============================================

def run_server(host: str = "0.0.0.0", port: int = 8080):
    """서버 실행"""
    import uvicorn
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    run_server()
