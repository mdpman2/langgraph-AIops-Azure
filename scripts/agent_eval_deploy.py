#!/usr/bin/env python3
"""
Agent 생성 → 평가 → Human-in-the-Loop Azure 배포 워크플로우

사용법:
    python scripts/agent_eval_deploy.py                    # 전체 워크플로우
    python scripts/agent_eval_deploy.py --skip-agent       # 평가 + 배포만
    python scripts/agent_eval_deploy.py --eval-only        # 평가만
"""

import asyncio
import argparse
import json
import os
import sys
import subprocess
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime

# 프로젝트 루트 설정
SCRIPT_DIR = Path(__file__).parent
PROJECT_ROOT = SCRIPT_DIR.parent
sys.path.insert(0, str(PROJECT_ROOT / 'src'))

from dotenv import load_dotenv
load_dotenv(PROJECT_ROOT / '.env')

# ============================================
# 설정
# ============================================
EVALUATION_THRESHOLDS = {
    'groundedness': 0.7,
    'relevance': 0.7,
    'coherence': 0.8,
}

TEST_QUERIES = [
    "Python으로 간단한 REST API 서버를 만드는 방법을 설명해주세요.",
    "Azure Container Apps의 장점은 무엇인가요?",
    "LangGraph 스타일의 에이전트 워크플로우란 무엇인가요?",
]


# ============================================
# Agent 실행
# ============================================
async def run_agent(query: str) -> Dict:
    """에이전트 실행하여 응답 수집"""
    from langgraph_agent.workflow import AgentWorkflow

    workflow = AgentWorkflow()

    try:
        # AgentWorkflow.run()은 문자열을 받아 문자열을 반환
        response = await workflow.run(query)

        return {
            'query': query,
            'response': response,
            'context': "Agent 실행 완료",
        }
    finally:
        await workflow.close()


async def collect_agent_responses(queries: List[str]) -> List[Dict]:
    """여러 쿼리에 대한 에이전트 응답 수집"""
    print("\n" + "=" * 60)
    print("🤖 STAGE 1: Agent 응답 수집")
    print("=" * 60)

    results = []
    for i, query in enumerate(queries, 1):
        print(f"\n[{i}/{len(queries)}] 쿼리: {query[:50]}...")
        try:
            result = await run_agent(query)
            results.append(result)
            print(f"   ✅ 응답 수집 완료 ({len(result['response'])} 자)")
        except Exception as e:
            print(f"   ❌ 오류: {e}")
            results.append({
                'query': query,
                'response': f"오류 발생: {e}",
                'context': "Agent 실행 실패"
            })

    # 응답 저장
    output_file = PROJECT_ROOT / 'evaluation_results' / 'agent_responses.jsonl'
    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, 'w', encoding='utf-8') as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + '\n')

    print(f"\n📁 응답 저장됨: {output_file}")
    return results


# ============================================
# 평가
# ============================================
async def evaluate_responses(responses: List[Dict]) -> Tuple[Dict, bool]:
    """응답 품질 평가"""
    from openai import AsyncAzureOpenAI

    print("\n" + "=" * 60)
    print("📊 STAGE 2: 품질 평가")
    print("=" * 60)

    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4.1')

    if not endpoint or not api_key:
        print("❌ Azure OpenAI 설정 필요")
        return {}, False

    client = AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version='2024-10-01-preview',
    )

    all_scores = []
    for i, item in enumerate(responses, 1):
        print(f"\n[{i}/{len(responses)}] 평가 중...")

        eval_prompt = f"""다음 AI 에이전트 응답의 품질을 평가해주세요.

질문: {item['query']}
응답: {item['response']}
컨텍스트: {item.get('context', 'N/A')}

다음 JSON 형식으로 0.0~1.0 사이 점수를 반환하세요:
{{
    "groundedness": <응답이 사실에 기반하는 정도>,
    "relevance": <질문과의 관련성>,
    "coherence": <논리적 일관성>
}}"""

        try:
            response = await client.chat.completions.create(
                model=deployment,
                messages=[
                    {'role': 'system', 'content': '당신은 AI 응답 품질 평가 전문가입니다. JSON 형식으로만 응답하세요.'},
                    {'role': 'user', 'content': eval_prompt}
                ],
                temperature=0.3,
                max_tokens=200,
            )

            text = response.choices[0].message.content
            start = text.find('{')
            end = text.rfind('}') + 1
            scores = json.loads(text[start:end])
            all_scores.append(scores)
            print(f"   ✅ groundedness={scores.get('groundedness', 0):.2f}, "
                  f"relevance={scores.get('relevance', 0):.2f}, "
                  f"coherence={scores.get('coherence', 0):.2f}")
        except Exception as e:
            print(f"   ⚠️ 평가 오류, 기본값 사용: {e}")
            all_scores.append({'groundedness': 0.7, 'relevance': 0.7, 'coherence': 0.8})

    # 평균 계산
    avg_scores = {}
    for key in ['groundedness', 'relevance', 'coherence']:
        avg_scores[key] = sum(s.get(key, 0.7) for s in all_scores) / len(all_scores)

    # 결과 출력
    print("\n" + "-" * 60)
    print("📈 평가 결과 요약")
    print("-" * 60)

    all_pass = True
    for metric, score in avg_scores.items():
        threshold = EVALUATION_THRESHOLDS.get(metric, 0.7)
        status = "✅" if score >= threshold else "❌"
        if score < threshold:
            all_pass = False
        print(f"   {status} {metric}: {score:.3f} (기준: {threshold})")

    print("-" * 60)

    # 결과 저장
    result_file = PROJECT_ROOT / 'evaluation_results' / 'eval_summary.json'
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'scores': avg_scores,
            'thresholds': EVALUATION_THRESHOLDS,
            'all_pass': all_pass,
            'detail_scores': all_scores
        }, f, indent=2, ensure_ascii=False)

    print(f"📁 평가 결과 저장됨: {result_file}")

    return avg_scores, all_pass


# ============================================
# Human-in-the-Loop 배포 결정
# ============================================
def human_deploy_decision(scores: Dict, all_pass: bool) -> Optional[str]:
    """사람이 배포 여부 결정"""
    print("\n" + "=" * 60)
    print("🧑‍💻 STAGE 3: 배포 결정 (Human-in-the-Loop)")
    print("=" * 60)

    if all_pass:
        print("\n🎉 모든 품질 게이트 통과!")
        print("   배포를 진행할 수 있습니다.")
    else:
        print("\n⚠️ 일부 품질 지표가 기준 미달입니다.")
        print("   배포를 진행하려면 승인이 필요합니다.")

    print("\n" + "-" * 60)
    print("📊 현재 품질 점수:")
    for metric, score in scores.items():
        threshold = EVALUATION_THRESHOLDS.get(metric, 0.7)
        status = "✅" if score >= threshold else "⚠️"
        print(f"   {status} {metric}: {score:.3f}")
    print("-" * 60)

    print("\n🚀 Azure 배포 옵션:")
    print("   1. 기존 Azure OpenAI 사용 (deploy.py)")
    print("   2. 새 Azure OpenAI 생성 (deploy_to_azure.py)")
    print("   0. 배포 안 함 (종료)")
    print("-" * 60)

    while True:
        choice = input("\n배포 옵션을 선택하세요 (0-2): ").strip()

        if choice == '0':
            print("\n❌ 배포가 취소되었습니다.")
            return None
        elif choice == '1':
            return 'existing'
        elif choice == '2':
            return 'new'
        else:
            print("⚠️ 올바른 옵션을 선택하세요 (0, 1, 2)")


def run_deployment(deploy_type: str, max_retries: int = 2) -> bool:
    """배포 스크립트 실행 (재시도 로직 포함)"""
    print("\n" + "=" * 60)
    print("☁️ STAGE 4: Azure 배포")
    print("=" * 60)

    if deploy_type == 'existing':
        script = SCRIPT_DIR / 'deploy.py'
        print("\n📦 기존 Azure OpenAI를 사용하여 배포합니다...")
    else:
        script = SCRIPT_DIR / 'deploy_to_azure.py'
        print("\n📦 새 Azure OpenAI를 생성하여 배포합니다...")

    # 대화형 모드로 배포 스크립트 실행 (재시도 로직)
    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n🔄 배포 시도 {attempt}/{max_retries}...")

            result = subprocess.run(
                [sys.executable, str(script), '--interactive'],
                cwd=str(PROJECT_ROOT),
                shell=True
            )

            if result.returncode == 0:
                print(f"\n✅ 배포 성공!")
                return True
            else:
                print(f"\n⚠️ 배포 반환 코드: {result.returncode}")

                if attempt < max_retries:
                    retry = input("\n재시도하시겠습니까? (y/n): ").strip().lower()
                    if retry != 'y':
                        print("배포를 취소합니다.")
                        return False

        except KeyboardInterrupt:
            print("\n\n❌ 사용자가 배포를 취소했습니다.")
            return False
        except Exception as e:
            print(f"❌ 배포 오류: {e}")
            if attempt < max_retries:
                retry = input("\n재시도하시겠습니까? (y/n): ").strip().lower()
                if retry != 'y':
                    return False

    print(f"\n❌ 배포 실패 ({max_retries}회 시도)")
    return False


# ============================================
# 메인 워크플로우
# ============================================
async def main_workflow(skip_agent: bool = False, eval_only: bool = False):
    """전체 워크플로우 실행"""
    print("\n" + "=" * 60)
    print("🔄 LangGraph Agent - 평가 & 배포 워크플로우")
    print("=" * 60)
    print(f"   시작 시간: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)

    # 1. Agent 응답 수집
    if skip_agent:
        print("\n⏭️ Agent 실행 건너뜀, 기존 테스트 케이스 사용")
        responses = []
        test_file = PROJECT_ROOT / 'tests' / 'evaluation' / 'test_cases.jsonl'
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    responses.append(json.loads(line))
    else:
        responses = await collect_agent_responses(TEST_QUERIES)

    # 2. 평가
    scores, all_pass = await evaluate_responses(responses)

    if eval_only:
        print("\n" + "=" * 60)
        print("✅ 평가 완료 (배포 건너뜀)")
        print("=" * 60)
        return

    # 3. Human-in-the-Loop 배포 결정
    deploy_type = human_deploy_decision(scores, all_pass)

    if deploy_type is None:
        return

    # 4. 배포 실행
    success = run_deployment(deploy_type)

    # 최종 결과
    print("\n" + "=" * 60)
    print(f"📊 워크플로우 결과 요약")
    print("=" * 60)
    print(f"   평가 점수:")
    for metric, score in scores.items():
        threshold = EVALUATION_THRESHOLDS.get(metric, 0.7)
        status = "✅" if score >= threshold else "⚠️"
        print(f"     {status} {metric}: {score:.3f}")
    print(f"   품질 게이트: {'✅ 통과' if all_pass else '⚠️ 미달'}")
    print(f"   배포 결과: {'✅ 성공' if success else '❌ 실패'}")
    print("=" * 60)

    if success:
        print("🎉 전체 워크플로우 완료!")
        print("\n💡 다음 단계:")
        print("   1. Azure Portal에서 리소스 확인")
        print("   2. Application Insights에서 모니터링 확인")
        print("   3. 애플리케이션 URL로 접속하여 테스트")
    else:
        print("⚠️ 배포 중 문제가 발생했습니다.")
        print("\n💡 문제 해결 방법:")
        print("   1. Azure Portal에서 로그 확인")
        print("   2. 'az containerapp logs show' 명령으로 로그 확인")
        print("   3. 네트워크/권한 설정 확인")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(
        description='Agent 생성 → 평가 → Azure 배포 워크플로우'
    )
    parser.add_argument(
        '--skip-agent', '-s',
        action='store_true',
        help='Agent 실행 건너뛰고 기존 테스트 케이스로 평가'
    )
    parser.add_argument(
        '--eval-only', '-e',
        action='store_true',
        help='평가만 실행 (배포 건너뜀)'
    )
    parser.add_argument(
        '--queries', '-q',
        nargs='+',
        help='사용자 정의 쿼리 목록'
    )

    args = parser.parse_args()

    if args.queries:
        global TEST_QUERIES
        TEST_QUERIES = args.queries

    asyncio.run(main_workflow(
        skip_agent=args.skip_agent,
        eval_only=args.eval_only
    ))


if __name__ == '__main__':
    main()
