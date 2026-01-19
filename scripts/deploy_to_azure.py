#!/usr/bin/env python3
"""
Azure에 LangGraph Agent를 자동 배포하는 스크립트
- GPT-4.1 모델 가용성 확인
- 리소스 그룹 및 인프라 배포
- 배포 결과 검증
"""

import subprocess
import json
import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Tuple

# 기본 설정
DEFAULT_RESOURCE_PREFIX = 'langgraph-agent'
DEFAULT_ENVIRONMENT = 'dev'
DEFAULT_MODEL = 'gpt-4.1'
MODEL_VERSION = '2025-04-14'

# GPT-4.1 지원 지역 (우선순위 순)
GPT41_SUPPORTED_REGIONS = [
    'eastus',
    'eastus2',
    'westus',
    'westus3',
    'swedencentral',
    'northcentralus',
    'southcentralus',
    'uksouth',
]

# 한글 지역명 매핑
REGION_NAMES_KO = {
    'eastus': '미국 동부 (East US)',
    'eastus2': '미국 동부 2 (East US 2)',
    'westus': '미국 서부 (West US)',
    'westus3': '미국 서부 3 (West US 3)',
    'swedencentral': '스웨덴 중부 (Sweden Central)',
    'northcentralus': '미국 북중부 (North Central US)',
    'southcentralus': '미국 남중부 (South Central US)',
    'uksouth': '영국 남부 (UK South)',
    'koreacentral': '한국 중부 (Korea Central)',
    'japaneast': '일본 동부 (Japan East)',
}


def select_region_interactive(available_regions: list) -> str:
    """대화형으로 지역 선택"""
    print("\n" + "=" * 50)
    print("🌍 배포 지역 선택")
    print("=" * 50)

    for i, region in enumerate(available_regions, 1):
        region_name = REGION_NAMES_KO.get(region, region)
        print(f"  {i}. {region_name}")

    print(f"  0. 자동 선택 (최적 지역)")
    print("=" * 50)

    while True:
        try:
            choice = input("\n지역 번호를 입력하세요 (0-{0}): ".format(len(available_regions)))
            choice = int(choice)

            if choice == 0:
                return None  # 자동 선택
            elif 1 <= choice <= len(available_regions):
                return available_regions[choice - 1]
            else:
                print("❌ 잘못된 번호입니다. 다시 입력하세요.")
        except ValueError:
            print("❌ 숫자를 입력해주세요.")
        except KeyboardInterrupt:
            print("\n\n취소되었습니다.")
            return None


class AzureDeployer:
    """Azure 배포 클래스"""

    def __init__(self, region: str = None, environment: str = DEFAULT_ENVIRONMENT,
                 resource_prefix: str = DEFAULT_RESOURCE_PREFIX, interactive: bool = False):
        self.region = region
        self.environment = environment
        self.resource_prefix = resource_prefix
        self.resource_group = f"rg-{resource_prefix}-{environment}"
        self.script_dir = Path(__file__).parent
        self.infra_dir = self.script_dir.parent / 'infra'
        self.interactive = interactive

    def run_command(self, cmd: list, timeout: int = 300) -> Tuple[int, str, str]:
        """명령어 실행"""
        try:
            # Windows에서 az CLI가 제대로 동작하도록 shell=True 사용
            import platform
            use_shell = platform.system() == 'Windows'

            result = subprocess.run(
                cmd if not use_shell else ' '.join(cmd),
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=use_shell,
                env=os.environ.copy()  # 환경변수 상속
            )
            return result.returncode, result.stdout, result.stderr
        except subprocess.TimeoutExpired:
            return -1, '', 'Command timed out'
        except Exception as e:
            return -1, '', str(e)

    def check_login(self) -> bool:
        """Azure 로그인 확인"""
        print("🔐 Azure 로그인 상태 확인...")
        code, out, err = self.run_command(['az', 'account', 'show', '--output', 'json'])

        if code != 0:
            print("❌ Azure 로그인이 필요합니다. 'az login' 실행하세요.")
            return False

        try:
            account = json.loads(out)
            print(f"✅ 로그인됨: {account.get('name', 'Unknown')}")
        except:
            print("✅ Azure 로그인 확인됨")
        return True

    def check_model_availability(self, region: str) -> bool:
        """특정 지역에서 GPT-4.1 가용성 확인"""
        print(f"🔍 {region}에서 GPT-4.1 가용성 확인...")

        code, out, err = self.run_command([
            'az', 'cognitiveservices', 'model', 'list',
            '--location', region,
            '--query', "[?model.name=='gpt-4.1'].model.name",
            '--output', 'json'
        ], timeout=60)

        if code == 0 and out.strip():
            try:
                models = json.loads(out)
                if models:
                    print(f"✅ {region}: GPT-4.1 사용 가능")
                    return True
            except:
                pass

        print(f"❌ {region}: GPT-4.1 사용 불가")
        return False

    def find_available_region(self, max_results: int = 1, skip_regions: list = None) -> Optional[str]:
        """GPT-4.1 사용 가능한 지역 찾기 (찾으면 즉시 반환)"""
        print("\n📍 GPT-4.1 사용 가능 지역 검색...")
        print("=" * 50)

        skip_regions = skip_regions or []
        found_regions = []

        for region in GPT41_SUPPORTED_REGIONS:
            if region in skip_regions:
                continue
            if self.check_model_availability(region):
                found_regions.append(region)
                if len(found_regions) >= max_results:
                    print(f"\n✅ {max_results}개 지역 발견, 검색 종료")
                    break

        if found_regions:
            print(f"\n🎯 선택된 배포 지역: {found_regions[0]}")
            return found_regions[0]

        print("\n⚠️ GPT-4.1 사용 가능한 지역을 찾을 수 없습니다.")
        return None

    def find_available_regions(self, max_results: int = 3, skip_regions: list = None) -> list:
        """GPT-4.1 사용 가능한 지역 목록 찾기 (대화형 모드용)"""
        print("\n📍 GPT-4.1 사용 가능 지역 검색 중...")
        print("=" * 50)

        skip_regions = skip_regions or []
        found_regions = []

        for region in GPT41_SUPPORTED_REGIONS:
            if region in skip_regions:
                continue
            if self.check_model_availability(region):
                found_regions.append(region)
                print(f"   → {len(found_regions)}개 지역 발견")
                if len(found_regions) >= max_results:
                    print(f"\n✅ {max_results}개 지역 발견, 검색 종료 (--max-regions로 조정 가능)")
                    break

        print("=" * 50)
        return found_regions

    def create_resource_group(self) -> bool:
        """리소스 그룹 생성"""
        print(f"\n📦 리소스 그룹 생성: {self.resource_group}")

        code, out, err = self.run_command([
            'az', 'group', 'create',
            '--name', self.resource_group,
            '--location', self.region,
            '--output', 'json'
        ])

        if code != 0:
            print(f"❌ 리소스 그룹 생성 실패: {err}")
            return False

        print(f"✅ 리소스 그룹 생성됨: {self.resource_group}")
        return True

    def deploy_bicep(self) -> bool:
        """Bicep 템플릿 배포"""
        print(f"\n🚀 Azure 인프라 배포 시작...")
        print(f"   지역: {self.region}")
        print(f"   리소스 그룹: {self.resource_group}")
        print(f"   모델: {DEFAULT_MODEL}")

        bicep_file = self.infra_dir / 'main-gpt41.bicep'

        # what-if로 먼저 검증
        print("\n📋 배포 계획 검증 (what-if)...")
        code, out, err = self.run_command([
            'az', 'deployment', 'group', 'what-if',
            '--name', f'{self.resource_prefix}-deploy',
            '--resource-group', self.resource_group,
            '--template-file', str(bicep_file),
            '--parameters', f'location={self.region}',
            '--parameters', f'environment={self.environment}',
            '--parameters', f'baseName={self.resource_prefix}',
            '--output', 'table'
        ], timeout=120)

        if code != 0:
            print(f"⚠️ 배포 검증 경고: {err}")

        # 실제 배포
        print("\n⏳ 배포 실행 중 (약 5-10분 소요)...")
        code, out, err = self.run_command([
            'az', 'deployment', 'group', 'create',
            '--name', f'{self.resource_prefix}-deploy',
            '--resource-group', self.resource_group,
            '--template-file', str(bicep_file),
            '--parameters', f'location={self.region}',
            '--parameters', f'environment={self.environment}',
            '--parameters', f'baseName={self.resource_prefix}',
            '--output', 'json'
        ], timeout=600)

        if code != 0:
            print(f"❌ 배포 실패: {err}")
            return False

        try:
            result = json.loads(out)
            if result.get('properties', {}).get('provisioningState') == 'Succeeded':
                print("✅ 인프라 배포 성공!")

                # 출력값 표시
                outputs = result.get('properties', {}).get('outputs', {})
                if outputs:
                    print("\n📊 배포 결과:")
                    for key, val in outputs.items():
                        print(f"   {key}: {val.get('value', 'N/A')}")

                return True
            else:
                print(f"❌ 배포 상태: {result.get('properties', {}).get('provisioningState', 'Unknown')}")
                return False
        except:
            print("✅ 배포 명령 완료")
            return True

    def verify_deployment(self) -> bool:
        """배포 검증"""
        print("\n🔍 배포 검증 중...")

        # Container App 확인
        ca_name = f"ca-{self.resource_prefix}-{self.environment}"
        code, out, err = self.run_command([
            'az', 'containerapp', 'show',
            '--name', ca_name,
            '--resource-group', self.resource_group,
            '--query', '{name:name, url:properties.configuration.ingress.fqdn, status:properties.runningStatus}',
            '--output', 'json'
        ])

        if code == 0:
            try:
                info = json.loads(out)
                print(f"✅ Container App: {info.get('name')}")
                print(f"   URL: https://{info.get('url')}")
                print(f"   상태: {info.get('status')}")
                return True
            except:
                pass

        print("⚠️ Container App 확인 실패")
        return False

    def deploy(self) -> bool:
        """전체 배포 프로세스"""
        print("\n" + "=" * 60)
        print("🚀 LangGraph Agent Azure 배포 (GPT-4.1)")
        print("=" * 60)

        # 1. 로그인 확인
        if not self.check_login():
            return False

        # 2. 지역 확인/선택
        if self.interactive:
            # 대화형 모드: 사용 가능한 지역 확인 (최대 3개만 검색)
            print("\n🔍 GPT-4.1 사용 가능 지역 확인 중...")
            available_regions = self.find_available_regions(max_results=3)

            if not available_regions:
                print("❌ GPT-4.1 사용 가능한 지역이 없습니다.")
                return False

            # 사용자 선택
            selected = select_region_interactive(available_regions)
            if selected is None:
                # 자동 선택
                self.region = available_regions[0]
                print(f"\n🎯 자동 선택된 지역: {self.region}")
            else:
                self.region = selected
                print(f"\n🎯 선택된 지역: {self.region}")

        elif not self.region:
            # 지역 미지정: 자동 선택
            self.region = self.find_available_region()
            if not self.region:
                return False
        else:
            # 지역 지정됨: 해당 지역 확인 후 불가능시 대체 지역 검색
            if not self.check_model_availability(self.region):
                print(f"\n⚠️ {self.region}에서 GPT-4.1 사용 불가.")

                if self.interactive:
                    # 대화형: 대체 지역 선택 제안 (최대 3개만 검색)
                    print("다른 지역을 검색합니다...")
                    available_regions = self.find_available_regions(max_results=3, skip_regions=[self.region])

                    if not available_regions:
                        print("❌ 대체 가능한 지역이 없습니다.")
                        return False

                    print(f"\n대체 가능한 지역을 찾았습니다:")
                    selected = select_region_interactive(available_regions)
                    if selected is None:
                        self.region = available_regions[0]
                    else:
                        self.region = selected
                    print(f"🎯 대체 지역: {self.region}")
                else:
                    # 비대화형: 자동 대체 지역 검색
                    print("대체 지역 자동 검색 중...")
                    self.region = self.find_available_region()
                    if not self.region:
                        return False
                    print(f"🎯 대체 지역 선택: {self.region}")

        # 리소스 그룹명 업데이트 (지역 포함)
        self.resource_group = f"rg-{self.resource_prefix}-{self.environment}-{self.region}"

        # 3. 리소스 그룹 생성
        if not self.create_resource_group():
            return False

        # 4. Bicep 배포
        if not self.deploy_bicep():
            return False

        # 5. 검증
        self.verify_deployment()

        print("\n" + "=" * 60)
        print("🎉 배포 완료!")
        print("=" * 60)
        print(f"\n📌 Azure Portal에서 확인:")
        print(f"   https://portal.azure.com")
        print(f"   리소스 그룹: {self.resource_group}")

        return True


def main():
    parser = argparse.ArgumentParser(description='Azure에 LangGraph Agent 배포 (GPT-4.1)')
    parser.add_argument('--region', '-r', help='배포 지역 (기본: 자동 선택)')
    parser.add_argument('--environment', '-e', default='dev',
                        choices=['dev', 'staging', 'prod'], help='환경')
    parser.add_argument('--prefix', '-p', default='langgraph-agent',
                        help='리소스 이름 접두사')
    parser.add_argument('--check-only', action='store_true',
                        help='지역 가용성만 확인 (배포 안함)')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='대화형 모드: 지역을 수동으로 선택')
    parser.add_argument('--auto-fallback', '-a', action='store_true', default=True,
                        help='선택한 지역 불가시 자동 대체 지역 검색 (기본: 활성화)')

    args = parser.parse_args()

    deployer = AzureDeployer(
        region=args.region,
        environment=args.environment,
        resource_prefix=args.prefix,
        interactive=args.interactive
    )

    if args.check_only:
        deployer.check_login()
        deployer.find_available_region()
        return

    success = deployer.deploy()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
