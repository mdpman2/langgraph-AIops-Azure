#!/usr/bin/env python3
"""
Azure 지역별 서비스 가용성 확인 및 배포 스크립트
- 기존 Azure OpenAI 리소스 사용
- Container Apps, Application Insights 등 인프라만 배포
"""

import subprocess
import json
import sys
import os
import argparse
from pathlib import Path
from typing import Optional, Tuple, Dict

# ============================================
# 설정
# ============================================

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv()

# 기존 Azure OpenAI 설정 (환경변수 또는 기본값)
EXISTING_AOAI_ENDPOINT = os.getenv('AZURE_OPENAI_ENDPOINT', '')
EXISTING_AOAI_KEY = os.getenv('AZURE_OPENAI_API_KEY', '')  # .env에서 API 키 직접 읽기
EXISTING_AOAI_RG = os.getenv('AZURE_OPENAI_RG', '')
EXISTING_AOAI_NAME = os.getenv('AZURE_OPENAI_NAME', '')
EXISTING_AOAI_DEPLOYMENT = os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME', 'gpt-4.1')

# 배포 설정
DEFAULT_RESOURCE_PREFIX = 'langgraph-agent'
DEFAULT_ENVIRONMENT = 'dev'

# Container Apps 지원 지역
CONTAINER_APPS_REGIONS = [
    'eastus',
    'eastus2',
    'westus',
    'westus2',
    'westus3',
    'centralus',
    'northcentralus',
    'southcentralus',
    'westeurope',
    'northeurope',
    'swedencentral',
    'uksouth',
    'japaneast',
    'australiaeast',
    'koreacentral',
]

# 한글 지역명 매핑
REGION_NAMES_KO = {
    'eastus': '미국 동부 (East US)',
    'eastus2': '미국 동부 2 (East US 2)',
    'westus': '미국 서부 (West US)',
    'westus2': '미국 서부 2 (West US 2)',
    'westus3': '미국 서부 3 (West US 3)',
    'centralus': '미국 중부 (Central US)',
    'northcentralus': '미국 북중부 (North Central US)',
    'southcentralus': '미국 남중부 (South Central US)',
    'westeurope': '서유럽 (West Europe)',
    'northeurope': '북유럽 (North Europe)',
    'swedencentral': '스웨덴 중부 (Sweden Central)',
    'uksouth': '영국 남부 (UK South)',
    'japaneast': '일본 동부 (Japan East)',
    'australiaeast': '호주 동부 (Australia East)',
    'koreacentral': '한국 중부 (Korea Central)',
}


def select_region_interactive(available_regions: list) -> Optional[str]:
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


class ServiceChecker:
    """Azure 서비스 가용성 확인 클래스"""

    @staticmethod
    def run_az_command(cmd: list, timeout: int = 60) -> Tuple[int, str, str]:
        """Azure CLI 명령 실행"""
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

    @staticmethod
    def check_login() -> Tuple[bool, Dict]:
        """Azure 로그인 확인"""
        code, out, err = ServiceChecker.run_az_command(
            ['az', 'account', 'show', '--output', 'json']
        )
        if code == 0:
            try:
                return True, json.loads(out)
            except:
                return True, {}
        return False, {}

    @staticmethod
    def check_container_apps_availability(region: str) -> bool:
        """Container Apps 서비스 가용성 확인"""
        code, out, err = ServiceChecker.run_az_command([
            'az', 'provider', 'show',
            '--namespace', 'Microsoft.App',
            '--query', 'registrationState',
            '--output', 'tsv'
        ])
        return code == 0 and 'Registered' in out

    @staticmethod
    def check_existing_openai() -> Tuple[bool, Dict]:
        """기존 Azure OpenAI 리소스 확인"""
        code, out, err = ServiceChecker.run_az_command([
            'az', 'cognitiveservices', 'account', 'show',
            '--name', EXISTING_AOAI_NAME,
            '--resource-group', EXISTING_AOAI_RG,
            '--output', 'json'
        ])
        if code == 0:
            try:
                return True, json.loads(out)
            except:
                return True, {}
        return False, {}

    @staticmethod
    def get_openai_key() -> Optional[str]:
        """Azure OpenAI API 키 조회"""
        code, out, err = ServiceChecker.run_az_command([
            'az', 'cognitiveservices', 'account', 'keys', 'list',
            '--name', EXISTING_AOAI_NAME,
            '--resource-group', EXISTING_AOAI_RG,
            '--query', 'key1',
            '--output', 'tsv'
        ])
        return out.strip() if code == 0 else None

    @staticmethod
    def find_available_regions() -> list:
        """Container Apps 배포 가능 지역 찾기"""
        available = []
        print("\n🔍 서비스 배포 가능 지역 확인 중...")
        print("=" * 60)

        for region in CONTAINER_APPS_REGIONS:
            # 간단한 검증: 지역이 구독에서 사용 가능한지 확인
            code, out, err = ServiceChecker.run_az_command([
                'az', 'account', 'list-locations',
                '--query', f"[?name=='{region}'].name",
                '--output', 'tsv'
            ])

            if code == 0 and region in out:
                print(f"✅ {region}: 사용 가능")
                available.append(region)
            else:
                print(f"❌ {region}: 사용 불가")

        print("=" * 60)
        return available


class AzureDeployer:
    """Azure 배포 클래스 (기존 OpenAI 사용)"""

    def __init__(self, region: str = None, environment: str = DEFAULT_ENVIRONMENT,
                 resource_prefix: str = DEFAULT_RESOURCE_PREFIX, interactive: bool = False,
                 auto_fallback: bool = True):
        self.region = region
        self.environment = environment
        self.resource_prefix = resource_prefix
        self.resource_group = f"rg-{resource_prefix}-{environment}-{region}" if region else None
        self.script_dir = Path(__file__).parent
        self.infra_dir = self.script_dir.parent / 'infra'
        self.aoai_key = None
        self.interactive = interactive
        self.auto_fallback = auto_fallback

    def run_command(self, cmd: list, timeout: int = 300) -> Tuple[int, str, str]:
        """명령어 실행 (Windows 호환)"""
        try:
            # Windows에서 az CLI 실행을 위해 shell=True 필요
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                shell=True,
                env=os.environ.copy()
            )
            return result.returncode, result.stdout, result.stderr
        except Exception as e:
            return -1, '', str(e)

    def prepare(self) -> bool:
        """배포 준비"""
        print("\n📋 배포 준비 중...")

        # 1. Azure 로그인 확인
        logged_in, account = ServiceChecker.check_login()
        if not logged_in:
            print("❌ Azure 로그인이 필요합니다. 'az login' 실행하세요.")
            return False
        print(f"✅ Azure 로그인됨: {account.get('name', 'Unknown')}")

        # 2. Azure OpenAI 설정 확인
        # .env에 API 키가 있으면 직접 사용 (Azure 리소스 확인 생략)
        if EXISTING_AOAI_KEY and EXISTING_AOAI_ENDPOINT:
            print(f"✅ Azure OpenAI 설정 확인됨 (.env)")
            print(f"   엔드포인트: {EXISTING_AOAI_ENDPOINT[:50]}...")
            print(f"   모델: {EXISTING_AOAI_DEPLOYMENT}")
            self.aoai_key = EXISTING_AOAI_KEY
            return True

        # .env에 API 키가 없으면 Azure 리소스에서 조회 시도
        if not EXISTING_AOAI_NAME or not EXISTING_AOAI_RG:
            print("❌ Azure OpenAI 설정이 불완전합니다.")
            print("   .env 파일에 다음 중 하나를 설정하세요:")
            print("   1. AZURE_OPENAI_ENDPOINT + AZURE_OPENAI_API_KEY (권장)")
            print("   2. AZURE_OPENAI_NAME + AZURE_OPENAI_RG (리소스에서 키 조회)")
            return False

        # Azure 리소스에서 확인
        exists, aoai_info = ServiceChecker.check_existing_openai()
        if not exists:
            print(f"❌ 기존 Azure OpenAI를 찾을 수 없습니다: {EXISTING_AOAI_NAME}")
            print(f"   리소스 그룹: {EXISTING_AOAI_RG}")
            return False
        print(f"✅ 기존 Azure OpenAI 확인됨: {aoai_info.get('name', EXISTING_AOAI_NAME)}")
        print(f"   엔드포인트: {aoai_info.get('properties', {}).get('endpoint', EXISTING_AOAI_ENDPOINT)}")

        # 3. API 키 조회
        self.aoai_key = ServiceChecker.get_openai_key()
        if not self.aoai_key:
            print("❌ Azure OpenAI API 키를 가져올 수 없습니다.")
            return False
        print("✅ Azure OpenAI API 키 확인됨")

        return True

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

        print(f"✅ 리소스 그룹 생성됨")
        return True

    def deploy_infrastructure(self) -> bool:
        """인프라 배포 (기존 OpenAI 사용)"""
        print(f"\n🚀 인프라 배포 시작...")
        print(f"   지역: {self.region}")
        print(f"   리소스 그룹: {self.resource_group}")
        print(f"   기존 Azure OpenAI: {EXISTING_AOAI_NAME}")

        bicep_file = self.infra_dir / 'deploy-with-existing-aoai.bicep'

        # 실제 배포
        print("\n⏳ 배포 실행 중 (약 3-5분 소요)...")
        code, out, err = self.run_command([
            'az', 'deployment', 'group', 'create',
            '--name', f'{self.resource_prefix}-deploy-{self.region}',
            '--resource-group', self.resource_group,
            '--template-file', str(bicep_file),
            '--parameters', f'location={self.region}',
            '--parameters', f'environment={self.environment}',
            '--parameters', f'baseName={self.resource_prefix}',
            '--parameters', f'existingAoaiEndpoint={EXISTING_AOAI_ENDPOINT}',
            '--parameters', f'existingAoaiKey={self.aoai_key}',
            '--parameters', f'modelDeploymentName={EXISTING_AOAI_DEPLOYMENT}',
            '--output', 'json'
        ], timeout=600)

        if code != 0:
            print(f"❌ 배포 실패: {err}")
            return False

        try:
            result = json.loads(out)
            if result.get('properties', {}).get('provisioningState') == 'Succeeded':
                print("✅ 인프라 배포 성공!")

                outputs = result.get('properties', {}).get('outputs', {})
                if outputs:
                    print("\n📊 배포 결과:")
                    for key, val in outputs.items():
                        print(f"   {key}: {val.get('value', 'N/A')}")
                return True
        except:
            pass

        print("✅ 배포 명령 완료")
        return True

    def create_acr_and_build_image(self, max_retries: int = 3) -> Optional[str]:
        """ACR 생성 및 이미지 빌드 (재시도 로직 포함)"""
        import time

        # ACR 이름 생성 (영문자와 숫자만, 최대 50자)
        acr_name = f"acr{self.resource_prefix.replace('-', '')}{self.region.replace('-', '')}"[:50]

        print(f"\n🐳 Container Registry 설정: {acr_name}")

        # ACR 존재 여부 확인
        code, out, err = self.run_command([
            'az', 'acr', 'show',
            '--name', acr_name,
            '--query', 'loginServer',
            '--output', 'tsv'
        ])

        acr_created_now = False
        if code != 0:
            # ACR 생성
            print(f"   ACR 생성 중...")
            code, out, err = self.run_command([
                'az', 'acr', 'create',
                '--resource-group', self.resource_group,
                '--name', acr_name,
                '--sku', 'Basic',
                '--admin-enabled', 'true'
            ])

            if code != 0:
                print(f"❌ ACR 생성 실패: {err}")
                return None
            print(f"   ✅ ACR 생성됨")
            acr_created_now = True
            # 새로 생성된 ACR의 DNS 전파 대기
            print(f"   ⏳ DNS 전파 대기 중 (60초)...")
            time.sleep(60)
        else:
            print(f"   ✅ 기존 ACR 사용")

        # 로그인 서버 조회
        code, login_server, _ = self.run_command([
            'az', 'acr', 'show',
            '--name', acr_name,
            '--query', 'loginServer',
            '--output', 'tsv'
        ])
        login_server = login_server.strip()

        # 이미지 빌드 (재시도 로직)
        image_name = f"{login_server}/langgraph-agent:v1"
        print(f"\n🔨 Docker 이미지 빌드 중...")
        print(f"   이미지: {image_name}")
        print(f"   (약 2-3분 소요)")

        build_success = False
        last_error = ""

        for attempt in range(1, max_retries + 1):
            print(f"   📦 빌드 시도 {attempt}/{max_retries}...")

            code, out, err = self.run_command([
                'az', 'acr', 'build',
                '--registry', acr_name,
                '--image', 'langgraph-agent:v1',
                '--file', 'Dockerfile',
                '--target', 'production',
                '.'
            ], timeout=900)  # 15분 타임아웃

            if code == 0:
                build_success = True
                print(f"   ✅ 이미지 빌드 완료")
                break

            last_error = err

            # DNS 관련 오류인지 확인
            if 'no such host' in err.lower() or 'dns' in err.lower() or 'unauthorized' in err.lower():
                wait_time = 30 * attempt
                print(f"   ⚠️ DNS/인증 오류, {wait_time}초 대기 후 재시도...")
                time.sleep(wait_time)
            else:
                print(f"   ⚠️ 빌드 오류: {err[:200]}")
                if attempt < max_retries:
                    time.sleep(15)

        if not build_success:
            print(f"❌ 이미지 빌드 실패 ({max_retries}회 시도): {last_error[:300]}")
            return None

        # Container App에 ACR 권한 부여
        self._grant_acr_permissions(acr_name)

        return image_name

    def _grant_acr_permissions(self, acr_name: str) -> bool:
        """Container App에 ACR 권한 부여 (확인 후 부여)"""
        print(f"\n🔑 Container App ACR 권한 설정 중...")
        ca_name = f"ca-{self.resource_prefix}-{self.environment}"

        # Container App의 Principal ID 조회
        code, principal_id, _ = self.run_command([
            'az', 'containerapp', 'show',
            '--name', ca_name,
            '--resource-group', self.resource_group,
            '--query', 'identity.principalId',
            '--output', 'tsv'
        ])
        principal_id = principal_id.strip()

        if not principal_id:
            print(f"   ⚠️ Container App Managed Identity 없음")
            return False

        # ACR ID 조회
        code, acr_id, _ = self.run_command([
            'az', 'acr', 'show',
            '--name', acr_name,
            '--query', 'id',
            '--output', 'tsv'
        ])
        acr_id = acr_id.strip()

        if not acr_id:
            print(f"   ⚠️ ACR ID 조회 실패")
            return False

        # 기존 권한 확인
        code, existing_roles, _ = self.run_command([
            'az', 'role', 'assignment', 'list',
            '--assignee', principal_id,
            '--scope', acr_id,
            '--query', "[?roleDefinitionName=='AcrPull'].roleDefinitionName",
            '--output', 'tsv'
        ])

        if 'AcrPull' in existing_roles:
            print(f"   ✅ ACR Pull 권한 이미 존재")
            return True

        # AcrPull 권한 부여
        code, _, err = self.run_command([
            'az', 'role', 'assignment', 'create',
            '--assignee', principal_id,
            '--scope', acr_id,
            '--role', 'AcrPull'
        ])

        if code == 0:
            print(f"   ✅ ACR Pull 권한 부여됨")
            # 권한 전파 대기
            import time
            print(f"   ⏳ 권한 전파 대기 (30초)...")
            time.sleep(30)
            return True
        else:
            print(f"   ⚠️ 권한 부여 실패: {err[:100]}")
            return False

        return True

    def update_container_app_image(self, image_name: str, max_retries: int = 5) -> bool:
        """Container App 이미지 업데이트 (강화된 재시도 로직)"""
        import time

        ca_name = f"ca-{self.resource_prefix}-{self.environment}"
        acr_name = f"acr{self.resource_prefix.replace('-', '')}{self.region.replace('-', '')}"[:50]

        print(f"\n🚀 Container App 이미지 업데이트 중...")
        print(f"   새 이미지: {image_name}")

        # 레지스트리 설정
        print(f"   레지스트리 설정 중...")
        code, _, err = self.run_command([
            'az', 'containerapp', 'registry', 'set',
            '--name', ca_name,
            '--resource-group', self.resource_group,
            '--server', f'{acr_name}.azurecr.io',
            '--identity', 'system'
        ])

        if code != 0:
            print(f"   ⚠️ 레지스트리 설정 실패, 재시도...")
            time.sleep(10)
            self.run_command([
                'az', 'containerapp', 'registry', 'set',
                '--name', ca_name,
                '--resource-group', self.resource_group,
                '--server', f'{acr_name}.azurecr.io',
                '--identity', 'system'
            ])

        # 이미지 업데이트 (최대 5회 재시도)
        for attempt in range(1, max_retries + 1):
            print(f"   🔄 업데이트 시도 {attempt}/{max_retries}...")

            code, out, err = self.run_command([
                'az', 'containerapp', 'update',
                '--name', ca_name,
                '--resource-group', self.resource_group,
                '--image', image_name
            ], timeout=300)

            if code == 0:
                print(f"   ✅ Container App 업데이트 완료")
                return True

            # 오류 유형별 처리
            err_lower = err.lower()

            if 'another operation is in progress' in err_lower or 'conflict' in err_lower:
                wait_time = 30 * attempt
                print(f"   ⏳ 다른 작업 진행 중, {wait_time}초 대기 후 재시도...")
                time.sleep(wait_time)
            elif 'unauthorized' in err_lower or 'authentication' in err_lower:
                print(f"   ⚠️ 인증 오류 - ACR 권한 재확인 중...")
                self._grant_acr_permissions(acr_name)
                time.sleep(30)
            elif 'not found' in err_lower:
                print(f"   ⚠️ 이미지를 찾을 수 없음 - 대기 후 재시도...")
                time.sleep(60)
            else:
                print(f"   ⚠️ 오류: {err[:200]}")
                if attempt < max_retries:
                    time.sleep(20 * attempt)

        print(f"   ❌ 최대 재시도 횟수 초과")
        return False

    def verify_deployment(self) -> Dict:
        """배포 검증"""
        print("\n🔍 배포 검증 중...")

        result = {'success': False}

        # Container App 확인
        ca_name = f"ca-{self.resource_prefix}-{self.environment}"
        code, out, err = self.run_command([
            'az', 'containerapp', 'show',
            '--name', ca_name,
            '--resource-group', self.resource_group,
            '--query', '{name:name, url:properties.configuration.ingress.fqdn, status:properties.runningStatus, image:properties.template.containers[0].image}',
            '--output', 'json'
        ])

        if code == 0:
            try:
                info = json.loads(out)
                result['container_app'] = info
                result['url'] = f"https://{info.get('url')}"
                result['image'] = info.get('image', 'Unknown')
                result['success'] = True
                print(f"✅ Container App: {info.get('name')}")
                print(f"   URL: {result['url']}")
                print(f"   이미지: {result['image']}")
                print(f"   상태: {info.get('status')}")
            except:
                print("⚠️ Container App 응답 파싱 실패")
        else:
            print("⚠️ Container App 확인 실패")

        return result

    def health_check(self, url: str, max_retries: int = 10, wait_seconds: int = 15) -> bool:
        """헬스 체크 수행"""
        import time
        import urllib.request
        import urllib.error
        import ssl

        health_url = f"{url}/health"
        print(f"\n🏥 헬스 체크 시작: {health_url}")
        print(f"   (최대 {max_retries}회 시도, 각 {wait_seconds}초 간격)")

        # SSL 인증서 검증 무시 (테스트용)
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        for attempt in range(1, max_retries + 1):
            try:
                print(f"   🔄 시도 {attempt}/{max_retries}...", end=" ")

                req = urllib.request.Request(health_url, method='GET')
                req.add_header('User-Agent', 'LangGraph-Deploy/1.0')

                with urllib.request.urlopen(req, timeout=30, context=ssl_context) as response:
                    status_code = response.getcode()
                    body = response.read().decode('utf-8')

                    if status_code == 200:
                        print("✅ 성공!")
                        try:
                            health_data = json.loads(body)
                            print(f"   📊 응답: {json.dumps(health_data, ensure_ascii=False)}")
                        except:
                            print(f"   📊 응답: {body[:200]}")
                        return True
                    else:
                        print(f"⚠️ 상태 코드: {status_code}")

            except urllib.error.HTTPError as e:
                print(f"⚠️ HTTP 오류: {e.code}")
            except urllib.error.URLError as e:
                print(f"⚠️ 연결 오류: {e.reason}")
            except Exception as e:
                print(f"⚠️ 오류: {type(e).__name__}: {str(e)[:50]}")

            if attempt < max_retries:
                print(f"   ⏳ {wait_seconds}초 대기...")
                time.sleep(wait_seconds)

        print(f"   ❌ 헬스 체크 실패 ({max_retries}회 시도)")
        return False

    def full_health_check(self, url: str) -> Dict:
        """전체 헬스 체크 (여러 엔드포인트)"""
        import urllib.request
        import ssl

        print(f"\n🏥 전체 엔드포인트 검증")
        print("=" * 50)

        results = {}
        endpoints = [
            ('/health', 'Health API'),
            ('/', 'Web UI'),
        ]

        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE

        for path, name in endpoints:
            endpoint_url = f"{url}{path}"
            try:
                req = urllib.request.Request(endpoint_url, method='GET')
                req.add_header('User-Agent', 'LangGraph-Deploy/1.0')

                with urllib.request.urlopen(req, timeout=30, context=ssl_context) as response:
                    status_code = response.getcode()
                    results[name] = {
                        'url': endpoint_url,
                        'status': status_code,
                        'ok': status_code == 200
                    }
                    status = "✅" if status_code == 200 else "⚠️"
                    print(f"   {status} {name}: {status_code}")
            except Exception as e:
                results[name] = {
                    'url': endpoint_url,
                    'status': 'error',
                    'ok': False,
                    'error': str(e)
                }
                print(f"   ❌ {name}: {type(e).__name__}")

        print("=" * 50)
        all_ok = all(r.get('ok', False) for r in results.values())
        results['all_ok'] = all_ok

        return results

    def deploy(self) -> bool:
        """전체 배포 프로세스"""
        print("\n" + "=" * 60)
        print("🚀 LangGraph Agent Azure 배포")
        print("   (기존 Azure OpenAI 사용)")
        print("=" * 60)

        # 1. 준비
        if not self.prepare():
            return False

        # 2. 지역 선택/확인
        available_regions = ServiceChecker.find_available_regions()

        if not available_regions:
            print("❌ 배포 가능한 지역이 없습니다.")
            return False

        if self.interactive:
            # 대화형 모드: 사용자가 지역 선택
            selected = select_region_interactive(available_regions)
            if selected is None:
                # 자동 선택
                self.region = available_regions[0]
                print(f"\n🎯 자동 선택된 지역: {self.region}")
            else:
                self.region = selected
                print(f"\n🎯 선택된 지역: {self.region}")
        elif not self.region:
            # 지역 미지정: 첫 번째 사용 가능 지역 자동 선택
            self.region = available_regions[0]
            print(f"\n🎯 자동 선택된 지역: {self.region}")
        else:
            # 지정된 지역 확인
            if self.region not in available_regions:
                print(f"\n⚠️ {self.region} 지역은 배포 불가능합니다.")

                if self.auto_fallback:
                    if self.interactive:
                        # 대화형: 대체 지역 선택
                        print("대체 가능한 지역:")
                        selected = select_region_interactive(available_regions)
                        if selected is None:
                            self.region = available_regions[0]
                        else:
                            self.region = selected
                    else:
                        # 비대화형: 자동 대체
                        self.region = available_regions[0]
                    print(f"🎯 대체 지역: {self.region}")
                else:
                    print("❌ 자동 대체 기능이 비활성화되어 있습니다.")
                    return False

        # 리소스 그룹명 업데이트
        self.resource_group = f"rg-{self.resource_prefix}-{self.environment}-{self.region}"

        # 3. 리소스 그룹 생성
        if not self.create_resource_group():
            return False

        # 4. 인프라 배포
        if not self.deploy_infrastructure():
            return False

        # 5. ACR 생성 및 이미지 빌드
        image_name = self.create_acr_and_build_image()
        if not image_name:
            print("⚠️ 이미지 빌드 실패, 기본 이미지로 배포됨")
        else:
            # 6. Container App 이미지 업데이트
            if not self.update_container_app_image(image_name):
                print("⚠️ 이미지 업데이트 실패")

        # 7. 검증
        result = self.verify_deployment()

        # 8. 헬스 체크
        health_ok = False
        if result.get('url'):
            health_ok = self.health_check(result['url'])
            if health_ok:
                # 전체 엔드포인트 검증
                health_results = self.full_health_check(result['url'])
                result['health_check'] = health_results

        print("\n" + "=" * 60)
        if result['success'] and health_ok:
            print("🎉 배포 및 헬스 체크 완료!")
            print(f"\n🔗 애플리케이션 URL: {result.get('url', 'N/A')}")
            print(f"   Health: {result.get('url', '')}/health")
        elif result['success']:
            print("⚠️ 배포 완료 (헬스 체크 실패 - 앱 시작 중일 수 있음)")
            print(f"\n🔗 애플리케이션 URL: {result.get('url', 'N/A')}")
        else:
            print("⚠️ 배포 완료 (검증 필요)")
        print("=" * 60)

        print(f"\n📌 Azure Portal:")
        print(f"   https://portal.azure.com")
        print(f"   리소스 그룹: {self.resource_group}")

        return result['success'] and health_ok


def check_regions():
    """지역 가용성만 확인"""
    print("\n" + "=" * 60)
    print("🌍 Azure 서비스 배포 가능 지역 확인")
    print("=" * 60)

    # 로그인 확인
    logged_in, account = ServiceChecker.check_login()
    if not logged_in:
        print("❌ Azure 로그인이 필요합니다. 'az login' 실행하세요.")
        return
    print(f"✅ Azure 로그인됨: {account.get('name', 'Unknown')}")

    # 기존 Azure OpenAI 확인
    exists, aoai_info = ServiceChecker.check_existing_openai()
    if exists:
        print(f"\n📌 기존 Azure OpenAI 정보:")
        print(f"   이름: {EXISTING_AOAI_NAME}")
        print(f"   엔드포인트: {aoai_info.get('properties', {}).get('endpoint', EXISTING_AOAI_ENDPOINT)}")
        print(f"   지역: {aoai_info.get('location', 'Unknown')}")
    else:
        print(f"\n⚠️ 기존 Azure OpenAI를 찾을 수 없습니다: {EXISTING_AOAI_NAME}")

    # 지역 확인
    available = ServiceChecker.find_available_regions()

    print(f"\n📋 배포 가능 지역 ({len(available)}개):")
    for r in available:
        print(f"   • {r}")

    print(f"\n💡 배포 명령어:")
    if available:
        print(f"   python scripts/deploy.py --region {available[0]}")


def main():
    parser = argparse.ArgumentParser(
        description='Azure에 LangGraph Agent 배포 (기존 Azure OpenAI 사용)'
    )
    parser.add_argument('--region', '-r', help='배포 지역 (지정하지 않으면 자동 선택)')
    parser.add_argument('--environment', '-e', default='dev',
                        choices=['dev', 'staging', 'prod'], help='환경')
    parser.add_argument('--prefix', '-p', default='langgraph-agent',
                        help='리소스 이름 접두사')
    parser.add_argument('--check-regions', action='store_true',
                        help='배포 가능 지역만 확인')
    parser.add_argument('--interactive', '-i', action='store_true',
                        help='대화형 모드: 지역을 수동으로 선택')
    parser.add_argument('--no-auto-fallback', action='store_true',
                        help='선택한 지역 불가시 자동 대체 비활성화')

    args = parser.parse_args()

    if args.check_regions:
        check_regions()
        return

    deployer = AzureDeployer(
        region=args.region,
        environment=args.environment,
        resource_prefix=args.prefix,
        interactive=args.interactive,
        auto_fallback=not args.no_auto_fallback
    )

    success = deployer.deploy()
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()
