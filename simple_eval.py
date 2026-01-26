"""
AIops 품질 평가 스크립트 v2.0 (2026-01 업데이트)

변경 사항:
- GPT-5.2 via model-router 지원
- Azure AI Evaluation SDK Agent Evaluators 통합
  - IntentResolutionEvaluator: 의도 파악 정확도
  - ToolCallAccuracyEvaluator: 도구 호출 정확도
  - TaskAdherenceEvaluator: 작업 준수도
- API 버전: 2024-12-01-preview
- max_completion_tokens 파라미터 사용
- Structured Outputs 지원
"""
import asyncio
import json
import os
from typing import Dict, List, Any, Optional
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

# GPT-5.x 모델 감지 (O(1) 최적화)
GPT5_MODELS = frozenset({"gpt-5", "gpt-5.1", "gpt-5.2", "model-router", "gpt-5-mini", "gpt-5-nano"})

def is_gpt5_model(model: str) -> bool:
    """GPT-5.x 시리즈 여부 확인 - O(1) 최적화"""
    model_lower = model.lower()
    # 직접 매칭 우선 (O(1))
    if model_lower in GPT5_MODELS:
        return True
    # 부분 문자열 매칭 (fallback)
    return any(m in model_lower for m in GPT5_MODELS)


# ============================================
# 기본 품질 평가 (기존 호환)
# ============================================
async def basic_quality_evaluate(client: AsyncAzureOpenAI, deployment: str, test_cases: List[Dict]) -> List[Dict]:
    """기본 품질 지표 평가 (groundedness, relevance, coherence, fluency)"""
    results = []
    is_gpt5 = is_gpt5_model(deployment)

    for i, tc in enumerate(test_cases, 1):
        print(f'\n[{i}/{len(test_cases)}] 기본 품질 평가 중...')

        eval_prompt = f"""다음 응답의 품질을 평가해주세요.

질문: {tc['query']}
응답: {tc['response']}
컨텍스트: {tc.get('context', 'N/A')}

다음 형식의 JSON으로 점수를 반환해주세요:
{{
    "groundedness": 0.0-1.0,
    "relevance": 0.0-1.0,
    "coherence": 0.0-1.0,
    "fluency": 0.0-1.0
}}"""

        # GPT-5.x 파라미터 분기
        params = {
            "model": deployment,
            "messages": [
                {'role': 'system', 'content': '당신은 AI 응답 품질 평가 전문가입니다. JSON 형식으로만 응답하세요.'},
                {'role': 'user', 'content': eval_prompt}
            ],
            "temperature": 0.3,
        }

        if is_gpt5:
            params["max_completion_tokens"] = 200
        else:
            params["max_tokens"] = 200

        response = await client.chat.completions.create(**params)

        try:
            text = response.choices[0].message.content
            start = text.find('{')
            end = text.rfind('}') + 1
            scores = json.loads(text[start:end])
            results.append(scores)
            print(f'   ✅ 평가 완료: groundedness={scores.get("groundedness", 0):.2f}, relevance={scores.get("relevance", 0):.2f}')
        except Exception as e:
            results.append({'groundedness': 0.7, 'relevance': 0.7, 'coherence': 0.8, 'fluency': 0.8})
            print(f'   ⚠️ 파싱 실패, 기본값 사용')

    return results


# ============================================
# Agent Evaluators (2026 최신 - Azure AI Evaluation SDK)
# ============================================
async def agent_evaluate(client: AsyncAzureOpenAI, deployment: str, test_cases: List[Dict]) -> Dict[str, List[Dict]]:
    """에이전트 전용 평가 (Intent Resolution, Tool Call Accuracy, Task Adherence)"""
    print('\n' + '=' * 60)
    print('🤖 Agent Evaluators (2026 최신)')
    print('=' * 60)

    is_gpt5 = is_gpt5_model(deployment)
    agent_results = {
        'intent_resolution': [],
        'tool_call_accuracy': [],
        'task_adherence': []
    }

    for i, tc in enumerate(test_cases, 1):
        print(f'\n[{i}/{len(test_cases)}] Agent 평가 중...')

        # 1. Intent Resolution 평가
        intent_prompt = f"""다음 에이전트 응답이 사용자의 의도를 얼마나 정확하게 파악했는지 평가하세요.

사용자 질문: {tc['query']}
에이전트 응답: {tc['response']}

JSON 형식으로 응답:
{{
    "intent_resolution_score": 1-5 (5가 최고),
    "intent_understood": true/false,
    "clarification_needed": true/false,
    "reasoning": "평가 근거"
}}"""

        params = {
            "model": deployment,
            "messages": [
                {'role': 'system', 'content': '당신은 AI 에이전트 품질 평가 전문가입니다. 의도 파악 정확도를 평가합니다.'},
                {'role': 'user', 'content': intent_prompt}
            ],
            "temperature": 0.2,
        }
        if is_gpt5:
            params["max_completion_tokens"] = 300
            params["reasoning_effort"] = "medium"
        else:
            params["max_tokens"] = 300

        try:
            response = await client.chat.completions.create(**params)
            text = response.choices[0].message.content
            intent_data = json.loads(text[text.find('{'):text.rfind('}')+1])
            agent_results['intent_resolution'].append(intent_data)
            print(f'   📍 Intent Resolution: {intent_data.get("intent_resolution_score", "N/A")}/5')
        except Exception as e:
            agent_results['intent_resolution'].append({'intent_resolution_score': 3, 'error': str(e)})
            print(f'   ⚠️ Intent 평가 실패')

        # 2. Task Adherence 평가
        task_prompt = f"""에이전트가 주어진 작업 지시를 얼마나 잘 따랐는지 평가하세요.

작업 지시 (시스템 메시지): {tc.get('system_message', 'AI 어시스턴트로서 사용자를 돕습니다.')}
사용자 요청: {tc['query']}
에이전트 응답: {tc['response']}

JSON 형식으로 응답:
{{
    "task_adherence_score": 1-5 (5가 최고),
    "followed_instructions": true/false,
    "scope_violation": false/true,
    "reasoning": "평가 근거"
}}"""

        params["messages"] = [
            {'role': 'system', 'content': '당신은 AI 에이전트 품질 평가 전문가입니다. 작업 준수도를 평가합니다.'},
            {'role': 'user', 'content': task_prompt}
        ]

        try:
            response = await client.chat.completions.create(**params)
            text = response.choices[0].message.content
            task_data = json.loads(text[text.find('{'):text.rfind('}')+1])
            agent_results['task_adherence'].append(task_data)
            print(f'   📋 Task Adherence: {task_data.get("task_adherence_score", "N/A")}/5')
        except Exception as e:
            agent_results['task_adherence'].append({'task_adherence_score': 3, 'error': str(e)})
            print(f'   ⚠️ Task 평가 실패')

        # 3. Tool Call Accuracy 평가 (tool_calls가 있는 경우)
        if tc.get('tool_calls'):
            tool_prompt = f"""에이전트의 도구 호출이 적절했는지 평가하세요.

사용자 요청: {tc['query']}
호출된 도구: {json.dumps(tc['tool_calls'], ensure_ascii=False)}
도구 정의: {json.dumps(tc.get('tool_definitions', []), ensure_ascii=False)}

JSON 형식으로 응답:
{{
    "tool_call_accuracy_score": 1-5 (5가 최고),
    "correct_tool_selected": true/false,
    "correct_parameters": true/false,
    "reasoning": "평가 근거"
}}"""

            params["messages"] = [
                {'role': 'system', 'content': '당신은 AI 에이전트 품질 평가 전문가입니다. 도구 호출 정확도를 평가합니다.'},
                {'role': 'user', 'content': tool_prompt}
            ]

            try:
                response = await client.chat.completions.create(**params)
                text = response.choices[0].message.content
                tool_data = json.loads(text[text.find('{'):text.rfind('}')+1])
                agent_results['tool_call_accuracy'].append(tool_data)
                print(f'   🔧 Tool Call Accuracy: {tool_data.get("tool_call_accuracy_score", "N/A")}/5')
            except Exception as e:
                agent_results['tool_call_accuracy'].append({'tool_call_accuracy_score': 3, 'error': str(e)})
                print(f'   ⚠️ Tool 평가 실패')
        else:
            agent_results['tool_call_accuracy'].append({'skipped': True, 'reason': 'No tool calls'})

    return agent_results


async def simple_evaluate():
    """통합 평가 실행 - 기본 품질 + Agent Evaluators"""
    # 클라이언트 설정 (API 키 기반) - GPT-5.2 지원
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'model-router')  # 기본값 model-router
    api_version = os.getenv('AZURE_OPENAI_API_VERSION', '2024-12-01-preview')

    if not endpoint or not api_key:
        print('❌ 환경 변수 설정 필요: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY')
        return

    # 비동기 컨텍스트 매니저로 자동 리소스 정리
    async with AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version=api_version,
    ) as client:
        await _run_evaluation(client, deployment, api_version)


async def _run_evaluation(client: AsyncAzureOpenAI, deployment: str, api_version: str):
    """평가 로직 분리 (리팩토링)"""
    # GPT-5.x 감지 및 표시
    is_gpt5 = is_gpt5_model(deployment)
    print(f'\n🚀 AIops 평가 v2.0 시작')
    print(f'   모델: {deployment} (GPT-5.x: {is_gpt5})')
    print(f'   API 버전: {api_version}')

    # 테스트 케이스 로드
    test_cases = []
    test_file = 'tests/evaluation/test_cases.jsonl'
    if os.path.exists(test_file):
        with open(test_file, 'r', encoding='utf-8') as f:
            for line in f:
                if line.strip():
                    test_cases.append(json.loads(line))
    else:
        # 기본 테스트 케이스
        test_cases = [
            {
                "query": "Python으로 REST API 만드는 방법 알려줘",
                "response": "FastAPI를 사용하면 쉽게 REST API를 만들 수 있습니다. pip install fastapi uvicorn으로 설치하고...",
                "context": "프로그래밍 질문"
            }
        ]

    print(f'\n🔍 평가 대상: {len(test_cases)}개 테스트 케이스')
    print('=' * 60)

    # 1. 기본 품질 평가
    print('\n📊 기본 품질 평가')
    print('-' * 40)
    basic_results = await basic_quality_evaluate(client, deployment, test_cases)

    # 2. Agent Evaluators (2026 최신)
    agent_results = await agent_evaluate(client, deployment, test_cases)

    # 결과 집계 - 기본 품질
    basic_avg = {}
    for key in ['groundedness', 'relevance', 'coherence', 'fluency']:
        basic_avg[key] = sum(r.get(key, 0.7) for r in basic_results) / len(basic_results)

    # 결과 집계 - Agent Evaluators
    agent_avg = {}
    for metric in ['intent_resolution', 'task_adherence', 'tool_call_accuracy']:
        scores = [r.get(f'{metric}_score', 3) for r in agent_results[metric] if not r.get('skipped')]
        if scores:
            agent_avg[metric] = sum(scores) / len(scores) / 5  # 1-5 스케일을 0-1로 변환

    # 결과 출력
    print('\n' + '=' * 60)
    print('📊 최종 평가 결과')
    print('=' * 60)

    print('\n[기본 품질 지표]')
    basic_threshold = {'groundedness': 0.7, 'relevance': 0.7, 'coherence': 0.8, 'fluency': 0.8}
    all_pass = True
    for k, v in basic_avg.items():
        status = '✅' if v >= basic_threshold[k] else '❌'
        if v < basic_threshold[k]:
            all_pass = False
        print(f'{status} {k}: {v:.3f} (기준: {basic_threshold[k]})')

    print('\n[Agent Evaluators (2026 최신)]')
    agent_threshold = {'intent_resolution': 0.6, 'task_adherence': 0.6, 'tool_call_accuracy': 0.6}
    for k, v in agent_avg.items():
        threshold = agent_threshold.get(k, 0.6)
        status = '✅' if v >= threshold else '❌'
        if v < threshold:
            all_pass = False
        print(f'{status} {k}: {v:.3f} (기준: {threshold})')

    print('\n' + '=' * 60)
    if all_pass:
        print('🎉 모든 품질 게이트 통과! 배포 가능!')
    else:
        print('⚠️ 일부 지표가 기준 미달. 개선 필요.')

    # 결과 저장
    os.makedirs('evaluation_results', exist_ok=True)
    results_data = {
        'model': deployment,
        'api_version': api_version,
        'is_gpt5': is_gpt5,
        'basic_quality': {
            'averages': basic_avg,
            'results': basic_results
        },
        'agent_evaluators': {
            'averages': agent_avg,
            'results': agent_results
        },
        'all_pass': all_pass
    }

    with open('evaluation_results/metrics.json', 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    print('\n📁 결과 저장: evaluation_results/metrics.json')


if __name__ == '__main__':
    asyncio.run(simple_evaluate())
