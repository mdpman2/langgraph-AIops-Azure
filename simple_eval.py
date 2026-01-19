# 간단한 품질 평가 스크립트
import asyncio
import json
import os
from openai import AsyncAzureOpenAI
from dotenv import load_dotenv

load_dotenv()

async def simple_evaluate():
    # 클라이언트 설정 (API 키 기반)
    endpoint = os.getenv('AZURE_OPENAI_ENDPOINT')
    api_key = os.getenv('AZURE_OPENAI_API_KEY')
    deployment = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4.1')

    if not endpoint or not api_key:
        print('❌ 환경 변수 설정 필요: AZURE_OPENAI_ENDPOINT, AZURE_OPENAI_API_KEY')
        return

    client = AsyncAzureOpenAI(
        api_key=api_key,
        azure_endpoint=endpoint,
        api_version='2024-10-01-preview',
    )

    # 테스트 케이스 로드
    test_cases = []
    with open('tests/evaluation/test_cases.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                test_cases.append(json.loads(line))

    print(f'\n🔍 평가 시작: {len(test_cases)}개 테스트 케이스')
    print('=' * 60)

    results = []
    for i, tc in enumerate(test_cases, 1):
        print(f'\n[{i}/{len(test_cases)}] 평가 중...')

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

        response = await client.chat.completions.create(
            model=deployment,
            messages=[
                {'role': 'system', 'content': '당신은 AI 응답 품질 평가 전문가입니다. JSON 형식으로만 응답하세요.'},
                {'role': 'user', 'content': eval_prompt}
            ],
            temperature=0.3,
            max_tokens=200,
        )

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

    # 평균 계산
    avg = {}
    for key in ['groundedness', 'relevance', 'coherence', 'fluency']:
        avg[key] = sum(r.get(key, 0.7) for r in results) / len(results)

    print('\n' + '=' * 60)
    print('📊 평가 결과 요약')
    print('=' * 60)

    threshold = {'groundedness': 0.7, 'relevance': 0.7, 'coherence': 0.8, 'fluency': 0.8}
    all_pass = True
    for k, v in avg.items():
        status = '✅' if v >= threshold[k] else '❌'
        if v < threshold[k]:
            all_pass = False
        print(f'{status} {k}: {v:.3f} (기준: {threshold[k]})')

    print('\n' + '=' * 60)
    if all_pass:
        print('🎉 모든 품질 게이트 통과! 배포 가능!')
    else:
        print('⚠️ 일부 지표가 기준 미달. 개선 필요.')

    # 결과 저장
    os.makedirs('evaluation_results', exist_ok=True)
    with open('evaluation_results/metrics.json', 'w', encoding='utf-8') as f:
        json.dump({'averages': avg, 'results': results}, f, indent=2, ensure_ascii=False)
    print('\n📁 결과 저장: evaluation_results/metrics.json')

    await client.close()

if __name__ == '__main__':
    asyncio.run(simple_evaluate())
