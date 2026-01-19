# Copyright (c) Microsoft. All rights reserved.
"""
Main entry point for the LangGraph-style AI Agent

최적화:
- 시그널 핸들링 (SIGINT, SIGTERM)
- Graceful shutdown
- 컨텍스트 매니저 기반 리소스 관리
- 에러 복구 및 재시도 로직
"""

from __future__ import annotations

import asyncio
import argparse
import signal
import sys
from contextlib import asynccontextmanager
from functools import partial
from typing import AsyncIterator, Optional

import structlog
from rich.console import Console
from rich.panel import Panel
from rich.markdown import Markdown

from .workflow import AgentWorkflow
from .config import load_config

logger = structlog.get_logger(__name__)
console = Console()

# 전역 종료 이벤트
_shutdown_event: Optional[asyncio.Event] = None


def _handle_signal(signum: int, frame, loop: asyncio.AbstractEventLoop) -> None:
    """시그널 핸들러 (Graceful shutdown)"""
    sig_name = signal.Signals(signum).name
    console.print(f"\n[yellow]⚠️ {sig_name} 수신 - 종료 중...[/yellow]")

    if _shutdown_event:
        loop.call_soon_threadsafe(_shutdown_event.set)


@asynccontextmanager
async def managed_workflow(config) -> AsyncIterator[AgentWorkflow]:
    """워크플로우 리소스 관리 (컨텍스트 매니저)"""
    workflow = AgentWorkflow(config)
    try:
        yield workflow
    finally:
        await workflow.close()
        logger.info("workflow_closed")


async def run_agent(user_request: str, timeout: float = 300.0) -> str:
    """에이전트 실행 (타임아웃 및 에러 처리 포함)"""
    global _shutdown_event
    _shutdown_event = asyncio.Event()

    config = load_config()

    console.print(Panel(
        f"[bold blue]LangGraph-style AI Agent[/bold blue]\n\n"
        f"[yellow]사용자 요청:[/yellow] {user_request}",
        title="🤖 AI Agent Started"
    ))

    async with managed_workflow(config) as workflow:
        try:
            # 타임아웃과 종료 이벤트 동시 대기
            task = asyncio.create_task(workflow.run(user_request))
            shutdown_task = asyncio.create_task(_shutdown_event.wait())

            done, pending = await asyncio.wait(
                [task, shutdown_task],
                timeout=timeout,
                return_when=asyncio.FIRST_COMPLETED,
            )

            # 종료 이벤트가 먼저 발생한 경우
            if shutdown_task in done:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                console.print("[yellow]작업이 취소되었습니다.[/yellow]")
                return ""

            # 타임아웃 발생
            if not done:
                for p in pending:
                    p.cancel()
                raise asyncio.TimeoutError(f"작업이 {timeout}초 내에 완료되지 않았습니다.")

            result = task.result()

            console.print(Panel(
                Markdown(result),
                title="✅ 결과",
                border_style="green"
            ))

            return result

        except asyncio.TimeoutError as e:
            console.print(Panel(
                f"[red]타임아웃:[/red] {str(e)}",
                title="⏱️ Timeout",
                border_style="yellow"
            ))
            raise
        except asyncio.CancelledError:
            console.print("[yellow]작업이 취소되었습니다.[/yellow]")
            raise
        except Exception as e:
            console.print(Panel(
                f"[red]에러 발생:[/red] {str(e)}",
                title="❌ Error",
                border_style="red"
            ))
            logger.exception("agent_error", error=str(e))
            raise


async def interactive_mode() -> None:
    """대화형 모드 (개선된 종료 처리)"""
    console.print("[bold]대화형 모드[/bold] - 'quit' 또는 'exit'로 종료\n")

    while True:
        try:
            user_input = await asyncio.get_event_loop().run_in_executor(
                None,
                lambda: console.input("[bold cyan]요청 > [/bold cyan]")
            )

            if user_input.lower() in ("quit", "exit", "q"):
                console.print("[yellow]종료합니다.[/yellow]")
                break

            if user_input.strip():
                try:
                    await run_agent(user_input)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    continue
                except Exception as e:
                    console.print(f"[red]에러: {e}[/red]")
                    continue

        except (KeyboardInterrupt, EOFError):
            console.print("\n[yellow]종료합니다.[/yellow]")
            break


def setup_signal_handlers(loop: asyncio.AbstractEventLoop) -> None:
    """시그널 핸들러 설정"""
    if sys.platform != "win32":
        for sig in (signal.SIGINT, signal.SIGTERM):
            loop.add_signal_handler(
                sig,
                partial(_handle_signal, sig, None, loop)
            )
    else:
        # Windows에서는 signal.signal 사용
        signal.signal(signal.SIGINT, partial(_handle_signal, loop=loop))


def _cleanup_loop(loop: asyncio.AbstractEventLoop) -> None:
    """이벤트 루프 정리 - 개선된 예외 처리"""
    if loop is None or loop.is_closed():
        return

    try:
        # 남은 태스크 취소
        pending = asyncio.all_tasks(loop)
        if pending:
            for task in pending:
                task.cancel()
            # 취소된 태스크들이 완료될 때까지 대기
            loop.run_until_complete(
                asyncio.gather(*pending, return_exceptions=True)
            )
    except Exception as e:
        logger.warning("cleanup_warning", error=str(e))
    finally:
        try:
            loop.run_until_complete(loop.shutdown_asyncgens())
            loop.run_until_complete(loop.shutdown_default_executor())
        except Exception:
            pass
        loop.close()


def main() -> int:
    """CLI entry point (개선된 버전)"""
    parser = argparse.ArgumentParser(
        description="LangGraph-style AI Agent with Planning-Execution-Reflection-Decision workflow"
    )
    parser.add_argument(
        "request",
        type=str,
        nargs="?",
        help="사용자 요청"
    )
    parser.add_argument(
        "--interactive", "-i",
        action="store_true",
        help="대화형 모드 실행"
    )
    parser.add_argument(
        "--timeout", "-t",
        type=float,
        default=300.0,
        help="작업 타임아웃 (초, 기본값: 300)"
    )

    args = parser.parse_args()

    # 이벤트 루프 설정
    try:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        setup_signal_handlers(loop)

        if args.interactive:
            loop.run_until_complete(interactive_mode())
        elif args.request:
            loop.run_until_complete(run_agent(args.request, timeout=args.timeout))
        else:
            # 기본 예제 실행
            example_request = "Python으로 간단한 REST API 서버를 만드는 방법을 단계별로 설명해주세요."
            loop.run_until_complete(run_agent(example_request, timeout=args.timeout))

        return 0

    except KeyboardInterrupt:
        console.print("\n[yellow]종료합니다.[/yellow]")
        return 130
    except Exception as e:
        console.print(f"[red]치명적 에러: {e}[/red]")
        logger.exception("fatal_error", error=str(e))
        return 1
    finally:
        # 비동기 태스크 정리 개선
        _cleanup_loop(loop)


if __name__ == "__main__":
    sys.exit(main())
