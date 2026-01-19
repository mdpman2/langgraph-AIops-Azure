# Copyright (c) Microsoft. All rights reserved.
"""
Evaluation Threshold Checker
CI/CD 파이프라인에서 평가 결과가 임계값을 충족하는지 확인
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict


def check_thresholds(
    results_path: str,
    min_groundedness: float = 0.7,
    min_relevance: float = 0.7,
    min_coherence: float = 0.8,
    min_fluency: float = 0.8,
) -> bool:
    """평가 결과가 임계값을 충족하는지 확인"""

    # 결과 파일 로드
    with open(results_path, "r", encoding="utf-8") as f:
        results = json.load(f)

    averages = results.get("averages", {})

    # 임계값 정의
    thresholds = {
        "groundedness": min_groundedness,
        "relevance": min_relevance,
        "coherence": min_coherence,
        "fluency": min_fluency,
    }

    # 결과 확인
    all_passed = True
    print("\n" + "=" * 50)
    print("Evaluation Threshold Check")
    print("=" * 50)

    for metric, threshold in thresholds.items():
        actual = averages.get(metric, 0.0)
        passed = actual >= threshold
        status = "✅ PASS" if passed else "❌ FAIL"

        print(f"{metric:15} | Threshold: {threshold:.2f} | Actual: {actual:.2f} | {status}")

        if not passed:
            all_passed = False

    print("=" * 50)

    if all_passed:
        print("\n✅ All thresholds met! Evaluation PASSED.\n")
    else:
        print("\n❌ Some thresholds not met. Evaluation FAILED.\n")

    return all_passed


def main():
    parser = argparse.ArgumentParser(description="Check evaluation thresholds")
    parser.add_argument(
        "--results",
        type=str,
        required=True,
        help="Path to evaluation results JSON file",
    )
    parser.add_argument(
        "--min-groundedness",
        type=float,
        default=0.7,
        help="Minimum groundedness threshold",
    )
    parser.add_argument(
        "--min-relevance",
        type=float,
        default=0.7,
        help="Minimum relevance threshold",
    )
    parser.add_argument(
        "--min-coherence",
        type=float,
        default=0.8,
        help="Minimum coherence threshold",
    )
    parser.add_argument(
        "--min-fluency",
        type=float,
        default=0.8,
        help="Minimum fluency threshold",
    )

    args = parser.parse_args()

    passed = check_thresholds(
        results_path=args.results,
        min_groundedness=args.min_groundedness,
        min_relevance=args.min_relevance,
        min_coherence=args.min_coherence,
        min_fluency=args.min_fluency,
    )

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
