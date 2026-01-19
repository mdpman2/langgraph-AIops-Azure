#!/usr/bin/env python3
"""
Azure 지역별 GPT-4.1 모델 가용성 확인 스크립트
"""

import subprocess
import json
import sys
from typing import Optional

# GPT-4.1 지원 지역 목록 (2025년 기준)
SUPPORTED_REGIONS = [
    'eastus',
    'eastus2',
    'westus',
    'westus2',
    'westus3',
    'northcentralus',
    'southcentralus',
    'swedencentral',
    'uksouth',
    'francecentral',
    'germanywestcentral',
    'japaneast',
    'australiaeast',
    'canadaeast',
]

def check_az_login() -> bool:
    """Azure CLI 로그인 상태 확인"""
    try:
        result = subprocess.run(
            ['az', 'account', 'show', '--output', 'json'],
            capture_output=True, text=True, timeout=30
        )
        return result.returncode == 0
    except Exception:
        return False

def check_model_availability(region: str, model_name: str = 'gpt-4.1') -> dict:
    """특정 지역에서 모델 가용성 확인"""
    try:
        result = subprocess.run(
            ['az', 'cognitiveservices', 'model', 'list',
             '--location', region,
             '--query', f"[?model.name=='{model_name}'].{{name:model.name, version:model.version, kind:kind}}",
             '--output', 'json'],
            capture_output=True, text=True, timeout=60
        )

        if result.returncode == 0 and result.stdout.strip():
            models = json.loads(result.stdout)
            if models:
                return {
                    'region': region,
                    'available': True,
                    'models': models
                }

        return {'region': region, 'available': False, 'models': []}

    except Exception as e:
        return {'region': region, 'available': False, 'error': str(e)}

def find_best_region(preferred_regions: list = None) -> Optional[str]:
    """GPT-4.1 사용 가능한 최적 지역 찾기"""
    regions_to_check = preferred_regions or SUPPORTED_REGIONS

    print("🔍 GPT-4.1 모델 가용성 확인 중...")
    print("=" * 50)

    available_regions = []

    for region in regions_to_check:
        result = check_model_availability(region, 'gpt-4.1')

        if result['available']:
            print(f"✅ {region}: GPT-4.1 사용 가능")
            available_regions.append(region)
        else:
            print(f"❌ {region}: GPT-4.1 사용 불가")

    print("=" * 50)

    if available_regions:
        best = available_regions[0]
        print(f"\n🎯 추천 지역: {best}")
        return best
    else:
        print("\n⚠️ GPT-4.1 사용 가능한 지역이 없습니다.")
        return None

def main():
    """메인 함수"""
    print("\n" + "=" * 50)
    print("🚀 Azure GPT-4.1 지역 가용성 확인 도구")
    print("=" * 50 + "\n")

    # Azure 로그인 확인
    if not check_az_login():
        print("❌ Azure CLI 로그인이 필요합니다.")
        print("   'az login' 명령을 먼저 실행하세요.")
        sys.exit(1)

    print("✅ Azure CLI 로그인 확인됨\n")

    # 지역 가용성 확인
    best_region = find_best_region()

    if best_region:
        print(f"\n📋 배포 명령어:")
        print(f"   python scripts/deploy_to_azure.py --region {best_region}")

    return best_region

if __name__ == '__main__':
    main()
