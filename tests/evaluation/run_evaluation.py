# Copyright (c) Microsoft. All rights reserved.
"""
Agent Quality Evaluation Runner
Azure AI Evaluation SDK를 사용한 에이전트 품질 평가

100% Azure 기반:
- Azure AI Evaluation SDK (평가)
- Azure AI Foundry (모델)
- Azure Application Insights (결과 로깅)
- Azure Blob Storage (결과 저장)

최적화:
- 병렬 평가 처리 (asyncio.gather)
- 연결 풀링 및 재사용
- 메모리 효율적인 스트리밍 처리
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# Azure SDK imports
from azure.identity import DefaultAzureCredential
from azure.storage.blob import BlobServiceClient
from azure.storage.blob.aio import BlobServiceClient as AsyncBlobServiceClient

# Azure AI Evaluation imports
try:
    from azure.ai.evaluation import (
        GroundednessEvaluator,
        RelevanceEvaluator,
        CoherenceEvaluator,
        FluencyEvaluator,
        SimilarityEvaluator,
        F1ScoreEvaluator,
        evaluate,
    )
    AZURE_EVAL_AVAILABLE = True
except ImportError:
    AZURE_EVAL_AVAILABLE = False
    print("Warning: azure-ai-evaluation not installed. Run: pip install azure-ai-evaluation")

# Azure Monitor for logging
try:
    from azure.monitor.opentelemetry import configure_azure_monitor
    from opentelemetry import trace
    AZURE_MONITOR_AVAILABLE = True
except ImportError:
    AZURE_MONITOR_AVAILABLE = False

# 상수 정의
DEFAULT_METRICS = ("groundedness", "relevance", "coherence", "fluency")
MAX_CONCURRENT_EVALUATIONS = 5
QUALITY_THRESHOLD = 0.7


class AzureEvaluationRunner:
    """Azure AI Evaluation SDK 기반 에이전트 평가 실행기 (최적화 버전)"""

    __slots__ = (
        'credential', 'project_endpoint', 'model_deployment',
        'storage_connection', 'storage_container', 'app_insights_connection',
        'evaluators', 'tracer', '_semaphore', '_executor'
    )

    def __init__(
        self,
        azure_ai_project_endpoint: Optional[str] = None,
        model_deployment: Optional[str] = None,
        storage_connection_string: Optional[str] = None,
        app_insights_connection_string: Optional[str] = None,
        max_concurrent: int = MAX_CONCURRENT_EVALUATIONS,
    ):
        # Azure 자격 증명 (싱글톤 패턴)
        self.credential = DefaultAzureCredential()

        # Azure AI Foundry 설정
        self.project_endpoint = azure_ai_project_endpoint or os.getenv("AZURE_FOUNDRY_PROJECT_ENDPOINT")
        self.model_deployment = model_deployment or os.getenv("AZURE_FOUNDRY_MODEL_DEPLOYMENT", "gpt-4.1")

        # Azure Storage 설정 (결과 저장용)
        self.storage_connection = storage_connection_string or os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.storage_container = os.getenv("AZURE_EVAL_RESULTS_CONTAINER", "evaluation-results")

        # Azure Application Insights 설정 (모니터링)
        self.app_insights_connection = app_insights_connection_string or os.getenv("APPLICATIONINSIGHTS_CONNECTION_STRING")

        # 동시성 제어
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._executor = ThreadPoolExecutor(max_workers=max_concurrent)

        # 평가기 초기화 (지연 로딩)
        self.evaluators = self._initialize_azure_evaluators()

        # Azure Monitor 초기화
        self._setup_azure_monitor()

    def _setup_azure_monitor(self) -> None:
        """Azure Application Insights 모니터링 설정"""
        if AZURE_MONITOR_AVAILABLE and self.app_insights_connection:
            try:
                configure_azure_monitor(
                    connection_string=self.app_insights_connection,
                    enable_live_metrics=True,
                )
                self.tracer = trace.get_tracer(__name__)
                print("✅ Azure Application Insights 연결됨")
            except Exception as e:
                print(f"Warning: Azure Monitor 설정 실패: {e}")
                self.tracer = None
        else:
            self.tracer = None

    def _initialize_azure_evaluators(self) -> Dict[str, Any]:
        """Azure AI Evaluation SDK 평가기 초기화"""
        if not AZURE_EVAL_AVAILABLE:
            raise ImportError("azure-ai-evaluation SDK가 필요합니다. pip install azure-ai-evaluation")

        if not self.project_endpoint:
            raise ValueError("AZURE_FOUNDRY_PROJECT_ENDPOINT 환경 변수가 필요합니다")

        # Azure AI Foundry 모델 설정
        model_config = {
            "azure_endpoint": self.project_endpoint,
            "azure_deployment": self.model_deployment,
            "api_version": "2024-10-01-preview",
        }

        print(f"✅ Azure AI Foundry 연결: {self.project_endpoint}")
        print(f"   모델 배포: {self.model_deployment}")

        return {
            # 품질 평가 지표
            "groundedness": GroundednessEvaluator(model_config=model_config),
            "relevance": RelevanceEvaluator(model_config=model_config),
            "coherence": CoherenceEvaluator(model_config=model_config),
            "fluency": FluencyEvaluator(model_config=model_config),
            # 유사도 평가
            "similarity": SimilarityEvaluator(model_config=model_config),
            # F1 점수 (정답이 있는 경우)
            "f1_score": F1ScoreEvaluator(),
        }

    async def _evaluate_metric(
        self,
        metric: str,
        query: str,
        response: str,
        context: Optional[str],
        ground_truth: Optional[str],
    ) -> Tuple[str, float]:
        """단일 메트릭 평가 (병렬 처리용)"""
        if metric not in self.evaluators:
            return metric, 0.0

        async with self._semaphore:
            try:
                evaluator = self.evaluators[metric]

                # ThreadPool에서 동기 평가기 실행 (블로킹 방지)
                loop = asyncio.get_event_loop()

                if metric == "groundedness" and context:
                    eval_result = await loop.run_in_executor(
                        self._executor,
                        lambda: evaluator(query=query, response=response, context=context)
                    )
                elif metric == "f1_score" and ground_truth:
                    eval_result = await loop.run_in_executor(
                        self._executor,
                        lambda: evaluator(response=response, ground_truth=ground_truth)
                    )
                elif metric == "similarity" and ground_truth:
                    eval_result = await loop.run_in_executor(
                        self._executor,
                        lambda: evaluator(response=response, ground_truth=ground_truth)
                    )
                else:
                    eval_result = await loop.run_in_executor(
                        self._executor,
                        lambda: evaluator(query=query, response=response)
                    )

                score_key = f"gpt_{metric}"
                score = eval_result.get(score_key, eval_result.get(metric, 0.0))
                return metric, float(score)

            except Exception as e:
                print(f"⚠️ {metric} 평가 실패: {e}")
                return metric, 0.0

    async def evaluate_single(
        self,
        query: str,
        response: str,
        context: Optional[str] = None,
        ground_truth: Optional[str] = None,
        metrics: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """단일 응답 평가 (병렬 메트릭 평가)"""
        metrics = metrics or list(DEFAULT_METRICS)

        # 트레이싱 시작
        span_context = None
        if self.tracer:
            span_context = self.tracer.start_span("evaluate_single")
            span_context.set_attribute("query_length", len(query))
            span_context.set_attribute("metrics", ",".join(metrics))

        # 모든 메트릭을 병렬로 평가
        tasks = [
            self._evaluate_metric(metric, query, response, context, ground_truth)
            for metric in metrics
        ]

        metric_results = await asyncio.gather(*tasks, return_exceptions=True)

        # 결과 집계
        results: Dict[str, float] = {}
        for result in metric_results:
            if isinstance(result, tuple):
                metric_name, score = result
                results[metric_name] = score
            elif isinstance(result, Exception):
                print(f"⚠️ 평가 예외 발생: {result}")

        if span_context:
            span_context.end()

        return results

    async def _evaluate_item(
        self,
        index: int,
        item: Dict[str, Any],
        metrics: Optional[List[str]],
    ) -> Dict[str, Any]:
        """단일 테스트 아이템 평가 (배치 병렬 처리용)"""
        result = await self.evaluate_single(
            query=item.get("query", item.get("question", "")),
            response=item.get("response", item.get("answer", "")),
            context=item.get("context"),
            ground_truth=item.get("ground_truth"),
            metrics=metrics,
        )
        result["index"] = index
        result["query"] = item.get("query", item.get("question", ""))[:100]
        return result

    async def evaluate_batch(
        self,
        test_data_path: str,
        output_dir: str,
        metrics: Optional[List[str]] = None,
        upload_to_azure: bool = True,
        batch_size: int = 10,
    ) -> Dict[str, Any]:
        """배치 평가 실행 (병렬 처리 최적화)"""
        # 테스트 데이터 로드
        test_data = self._load_test_data(test_data_path)

        # 결과 저장 디렉토리 생성
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        all_results: List[Dict[str, Any]] = []
        metric_sums: Dict[str, float] = defaultdict(float)
        num_items = len(test_data)

        print(f"\n🔍 평가 시작: {num_items}개 테스트 케이스")
        print(f"   평가 지표: {metrics or 'all'}")
        print(f"   배치 크기: {batch_size}")

        # 배치 단위로 병렬 처리
        for batch_start in range(0, num_items, batch_size):
            batch_end = min(batch_start + batch_size, num_items)
            batch = test_data[batch_start:batch_end]

            print(f"  [{batch_start+1}-{batch_end}/{num_items}] 평가 중...")

            # 배치 내 아이템들을 병렬로 평가
            tasks = [
                self._evaluate_item(batch_start + i, item, metrics)
                for i, item in enumerate(batch)
            ]

            batch_results = await asyncio.gather(*tasks, return_exceptions=True)

            for result in batch_results:
                if isinstance(result, dict):
                    all_results.append(result)
                    for key, score in result.items():
                        if isinstance(score, (int, float)):
                            metric_sums[key] += score
                elif isinstance(result, Exception):
                    print(f"  ⚠️ 배치 평가 예외: {result}")

        # 평균 계산
        averages = {
            metric: round(total / num_items, 3)
            for metric, total in metric_sums.items()
            if metric not in ("index",)
        }

        # 결과 요약
        summary = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "azure_project_endpoint": self.project_endpoint,
            "model_deployment": self.model_deployment,
            "num_items": num_items,
            "averages": averages,
            "quality_passed": all(
                avg >= QUALITY_THRESHOLD
                for key, avg in averages.items()
                if key in DEFAULT_METRICS
            ),
            "results": all_results,
        }

        # 로컬 저장 (비동기 파일 I/O)
        await self._save_results_async(summary, all_results, output_path)

        # Azure Blob Storage에 업로드
        if upload_to_azure and self.storage_connection:
            await self._upload_to_azure_storage(summary, output_path)

        # 결과 출력
        self._print_summary(num_items, averages)

        return summary

    async def _save_results_async(
        self,
        summary: Dict[str, Any],
        results: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """비동기 결과 저장"""
        loop = asyncio.get_event_loop()

        # metrics.json 저장
        metrics_file = output_path / "metrics.json"
        await loop.run_in_executor(
            None,
            lambda: metrics_file.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False),
                encoding="utf-8"
            )
        )

        # detailed_results.jsonl 저장
        detailed_file = output_path / "detailed_results.jsonl"
        lines = [json.dumps(r, ensure_ascii=False) for r in results]
        await loop.run_in_executor(
            None,
            lambda: detailed_file.write_text("\n".join(lines), encoding="utf-8")
        )

    @staticmethod
    def _print_summary(num_items: int, averages: Dict[str, float]) -> None:
        """결과 요약 출력"""
        print(f"\n📊 평가 결과 요약:")
        print(f"   총 항목: {num_items}")
        for metric, avg in averages.items():
            if metric == "index":
                continue
            status = "✅" if avg >= QUALITY_THRESHOLD else "⚠️"
            print(f"   {status} {metric}: {avg}")

    async def _upload_to_azure_storage(self, summary: Dict, output_path: Path) -> None:
        """Azure Blob Storage에 결과 업로드 (비동기)"""
        try:
            # 비동기 Blob 클라이언트 사용
            async with AsyncBlobServiceClient.from_connection_string(
                self.storage_connection
            ) as blob_service:
                container_client = blob_service.get_container_client(self.storage_container)

                # 컨테이너 생성 (없는 경우)
                try:
                    await container_client.create_container()
                except Exception:
                    pass  # 이미 존재

                # 타임스탬프 기반 경로
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")

                # 병렬 업로드
                async def upload_file(filename: str) -> None:
                    blob_client = container_client.get_blob_client(
                        f"evaluations/{timestamp}/{filename}"
                    )
                    file_path = output_path / filename
                    async with blob_client:
                        with open(file_path, "rb") as f:
                            await blob_client.upload_blob(f, overwrite=True)

                await asyncio.gather(
                    upload_file("metrics.json"),
                    upload_file("detailed_results.jsonl"),
                )

                print(f"   ☁️ Azure Blob Storage 업로드 완료: evaluations/{timestamp}/")

        except Exception as e:
            print(f"   ⚠️ Azure Storage 업로드 실패: {e}")

    def _load_test_data(self, path: str) -> List[Dict]:
        """테스트 데이터 로드"""
        test_data = []

        with open(path, "r", encoding="utf-8") as f:
            if path.endswith(".jsonl"):
                for line in f:
                    if line.strip():
                        test_data.append(json.loads(line))
            else:
                test_data = json.load(f)

        return test_data


def main():
    parser = argparse.ArgumentParser(
        description="Azure AI Evaluation SDK 기반 AI 에이전트 평가"
    )
    parser.add_argument(
        "--test-data",
        type=str,
        required=True,
        help="테스트 데이터 파일 경로 (JSONL 또는 JSON)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="평가 결과 저장 디렉토리",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["groundedness", "relevance", "coherence", "fluency"],
        help="평가할 지표 목록",
    )
    parser.add_argument(
        "--azure-project-endpoint",
        type=str,
        default=None,
        help="Azure AI Foundry 프로젝트 엔드포인트",
    )
    parser.add_argument(
        "--no-upload",
        action="store_true",
        help="Azure Storage 업로드 비활성화",
    )

    args = parser.parse_args()

    runner = AzureEvaluationRunner(
        azure_ai_project_endpoint=args.azure_project_endpoint,
    )

    asyncio.run(runner.evaluate_batch(
        test_data_path=args.test_data,
        output_dir=args.output_dir,
        metrics=args.metrics,
        upload_to_azure=not args.no_upload,
    ))


if __name__ == "__main__":
    main()
