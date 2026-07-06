"""End-to-end calibration CLI for the CSIC 2010 dataset.

Wires the whole pipeline with real components:

    train     CsicTrafficSource -> CalibrateModel -> HmmDetector -> PickleModelRepository
    evaluate  load model -> score normal vs anomalous packets -> IDS metrics

Usage:
    python learning_data/calibrate_cli.py train    [--normal-csv F] [--model F] [--limit N] ...
    python learning_data/calibrate_cli.py evaluate [--eval-csv F]   [--model F] [--limit N] ...
    python learning_data/calibrate_cli.py run      # train then evaluate
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
from csic_to_packets import iter_balanced_packets
from csic_traffic_source import CsicTrafficSource, record_to_flow

from waf.application.calibrate_model import CalibrateModel, CalibrationReport
from waf.infrastructure.adapter.persistence import PickleModelRepository
from waf.infrastructure.hmm import HmmDetector, evaluate, extractor_by_name
from waf.infrastructure.hmm.benchmark import ModelBenchmark, VariantResult, render_comparison
from waf.infrastructure.hmm.evaluation import (
    ClassificationMetrics,
    EvaluationReport,
    classification_metrics,
)
from waf.infrastructure.hmm.model_variant import ABLATION_MATRIX, ModelVariant

_DATA_DIR = Path(__file__).parent
_DEFAULT_NORMAL = _DATA_DIR / "csic_database_normal.csv"
_DEFAULT_EVAL = _DATA_DIR / "csic_database.csv"
_DEFAULT_MODEL = _DATA_DIR / "hmm_model.pkl"


def train_model(
    normal_csv: str | Path,
    model_path: str | Path,
    *,
    validation_ratio: float = 0.2,
    extractor: str = "byte_class",
    n_states: int = 8,
    window: int = 0,
    ensemble_size: int = 1,
    max_fit_windows: int | None = None,
    target_fpr: float = 0.01,
    n_iter: int = 100,
    limit: int | None = None,
) -> CalibrationReport:
    source = CsicTrafficSource(normal_csv, validation_ratio=validation_ratio, limit=limit)
    repository = PickleModelRepository(model_path)
    detector = HmmDetector(
        extractor=extractor_by_name(extractor),
        n_states=n_states,
        window=window,
        ensemble_size=ensemble_size,
        max_fit_windows=max_fit_windows,
        target_fpr=target_fpr,
        n_iter=n_iter,
    )
    return CalibrateModel(source, repository)(detector)


@dataclass(frozen=True, slots=True)
class ScoreStats:
    """Per-class summary of the per-symbol log-likelihood ("likelihood 분포")."""

    count: int
    mean: float
    std: float
    minimum: float
    maximum: float


def _score_stats(scores: list[float]) -> ScoreStats:
    arr = np.asarray(scores, dtype=float)
    if arr.size == 0:
        return ScoreStats(0, 0.0, 0.0, 0.0, 0.0)
    return ScoreStats(
        int(arr.size), float(arr.mean()), float(arr.std()), float(arr.min()), float(arr.max())
    )


@dataclass(frozen=True, slots=True)
class EvaluationOutcome:
    """Everything the report (content.md) needs, in one place."""

    detector: HmmDetector
    report: EvaluationReport | None  # None if a class is empty
    classification: ClassificationMetrics | None
    normal_stats: ScoreStats
    attack_stats: ScoreStats
    covered: int
    uncovered: int
    target_fpr: float


def evaluate_model(
    eval_csv: str | Path,
    model_path: str | Path,
    *,
    target_fpr: float = 0.01,
    limit: int | None = None,
) -> EvaluationOutcome:
    """Score the eval set and gather every value the report needs.

    Classification metrics come from the model's *realized* per-partition decisions
    (`assessment.blocked`), i.e. its true operating point — not a re-thresholding of
    the score arrays. Ranking metrics (ROC/PR/AUCp) come from the scores via evaluate().
    """
    detector = cast(HmmDetector, PickleModelRepository(model_path).load())

    normal_scores: list[float] = []
    attack_scores: list[float] = []
    tp = fp = fn = tn = 0  # positive class = attack
    covered = 0
    uncovered = 0
    for record in iter_balanced_packets(eval_csv, limit):
        assessment = detector.assess_flow(record_to_flow(record))
        if not assessment.covered:
            uncovered += 1
            continue
        covered += 1
        if record.is_normal:
            normal_scores.append(assessment.score)
            fp += assessment.blocked
            tn += not assessment.blocked
        elif record.is_anomalous:
            attack_scores.append(assessment.score)
            tp += assessment.blocked
            fn += not assessment.blocked

    report: EvaluationReport | None = None
    classification: ClassificationMetrics | None = None
    if normal_scores and attack_scores:
        report = evaluate(normal_scores, attack_scores, target_fpr=target_fpr)
        classification = classification_metrics(tp=tp, fp=fp, fn=fn, tn=tn)

    return EvaluationOutcome(
        detector=detector,
        report=report,
        classification=classification,
        normal_stats=_score_stats(normal_scores),
        attack_stats=_score_stats(attack_scores),
        covered=covered,
        uncovered=uncovered,
        target_fpr=target_fpr,
    )


def compare_models(
    normal_csv: str | Path,
    eval_csv: str | Path,
    variants: Sequence[ModelVariant],
    *,
    validation_ratio: float = 0.2,
    limit: int | None = None,
) -> list[VariantResult]:
    """여러 모델 버전을 같은 CSIC 데이터로 학습·평가해 비교 결과를 모은다.

    CSIC 어댑터(학습 트래픽 소스 + 패킷 파서)를 도메인-순수 ModelBenchmark에 주입하는
    얇은 배선층. 모든 변형이 동일한 정상/검증/평가 데이터를 보도록 한 번만 로드한다.
    """
    source = CsicTrafficSource(normal_csv, validation_ratio=validation_ratio, limit=limit)
    normal = list(source.normal())
    validation = list(source.validation())
    labeled = [
        (record_to_flow(record), record.is_anomalous)
        for record in iter_balanced_packets(eval_csv, limit)
        if record.is_normal or record.is_anomalous
    ]
    benchmark = ModelBenchmark(normal, validation, labeled)
    return benchmark.run(variants)


def _print_compare(results: Sequence[VariantResult]) -> None:
    print(render_comparison(results))


def _print_train(report: CalibrationReport) -> None:
    print(
        f"[train] fit on {report.normal_count} normal, "
        f"calibrated on {report.validation_count} validation packets"
    )


def _fmt_count(value: int | None) -> str:
    return str(value) if value is not None else "— (모델에 미기록; 재학습 시 기록됨)"


def _print_eval(outcome: EvaluationOutcome, model_path: str | Path) -> None:
    """Render a report card whose rows map 1:1 to reports/content.md 요구 수치."""
    d = outcome.detector
    fpr = outcome.target_fpr
    rule = "═" * 70

    print(rule)
    print(f" CSIC 2010 평가 리포트   (model: {Path(model_path).name})")
    print(rule)

    print("[모델 구조]")
    print(f"  추출기(관측 심볼화)   : {d.extractor_name}")
    print(f"  상태 수 (n_states)    : {d.n_states}")
    print(f"  관측 기호 수          : {d.n_symbols}")
    print(f"  전이행렬 크기 (S×S)   : {d.n_states} × {d.n_states}")
    print(f"  관측행렬 크기 (S×O)   : {d.n_states} × {d.n_symbols}")
    print(f"  슬라이딩 윈도우       : {d.window if d.window > 0 else 'off'}")
    print(f"  앙상블 크기 (K)       : {d.ensemble_size}")
    print(f"  파티션(모델) 수       : {d.num_partitions}")

    print("\n[학습 데이터]   (one-class: 정상 트래픽만 학습)")
    print(f"  정상 학습 패킷        : {_fmt_count(d.train_count)}")
    print(f"  검증(보정) 패킷       : {_fmt_count(d.validation_count)}")
    print("  공격 학습 패킷        : 0  (one-class 설계: 공격 데이터로 학습하지 않음)")

    n_normal = outcome.normal_stats.count
    n_attack = outcome.attack_stats.count
    print("\n[테스트 데이터]")
    print(f"  전체 평가 패킷        : {outcome.covered + outcome.uncovered}")
    print(f"  파티션 커버           : {outcome.covered} covered / {outcome.uncovered} uncovered")
    print(f"  정상 / 공격           : {n_normal} / {n_attack}")

    if outcome.report is None or outcome.classification is None:
        print("\n[지표] 두 클래스 모두에 충분한 표본이 없어 지표를 계산할 수 없음")
        print(rule)
        return

    r = outcome.report
    c = outcome.classification
    print("\n[탐지 성능 (ranking)]")
    print(f"  {'ROC-AUC':<19} : {r.roc_auc:.3f}")
    print(f"  {'PR-AUC':<19} : {r.pr_auc:.3f}")
    print(f"  {f'AUCp@{fpr:.0%}':<19} : {r.partial_auc:.3f}")
    print(f"  {f'TPR@{fpr:.0%}FPR':<19} : {r.tpr_at_fpr:.3f}   (anomaly thr={r.threshold_at_fpr:.3f})")

    print(f"\n[분류 성능 @ 보정 임계치(목표 FPR={fpr:.0%}, 파티션별)]")
    print(f"  Accuracy              : {c.accuracy:.3f}")
    print(f"  Precision             : {c.precision:.3f}")
    print(f"  Recall                : {c.recall:.3f}")
    print(f"  F1-score              : {c.f1:.3f}")
    print(f"  혼동행렬              : TP={c.tp}  FP={c.fp}  FN={c.fn}  TN={c.tn}")

    ns, ats = outcome.normal_stats, outcome.attack_stats
    print("\n[클래스별 likelihood]  (per-symbol 로그우도, 높을수록 정상)")
    print(f"  정상  평균±표준편차   : {ns.mean:.3f} ± {ns.std:.3f}   (min {ns.minimum:.3f} ~ max {ns.maximum:.3f})")
    print(f"  공격  평균±표준편차   : {ats.mean:.3f} ± {ats.std:.3f}   (min {ats.minimum:.3f} ~ max {ats.maximum:.3f})")
    print(rule)


def main() -> None:
    # The report card uses box-drawing / ×·± glyphs and Korean labels; force UTF-8
    # so output doesn't crash on a legacy Windows console codepage (e.g. cp949).
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")

    parser = argparse.ArgumentParser(description="CSIC 2010 HMM calibration pipeline")
    sub = parser.add_subparsers(dest="command", required=True)

    for name in ("train", "evaluate", "run", "compare"):
        p = sub.add_parser(name)
        p.add_argument("--normal-csv", default=str(_DEFAULT_NORMAL))
        p.add_argument("--eval-csv", default=str(_DEFAULT_EVAL))
        p.add_argument("--model", default=str(_DEFAULT_MODEL))
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--validation-ratio", type=float, default=0.2)
        p.add_argument(
            "--extractor",
            default="byte_class",
            choices=["byte_class", "raw_byte", "window_token"],
        )
        p.add_argument("--n-states", type=int, default=8)
        p.add_argument("--window", type=int, default=0, help="HMMPayl sliding-window width (0=off)")
        p.add_argument("--ensemble", type=int, default=1, help="number of HMMs in the ensemble")
        p.add_argument(
            "--max-fit-windows", type=int, default=None, help="cap on training windows (sampling)"
        )
        p.add_argument("--target-fpr", type=float, default=0.01)
        p.add_argument("--n-iter", type=int, default=100)

    args = parser.parse_args()

    if args.command == "compare":
        # ablation 매트릭스 전체를 같은 데이터로 학습·평가해 한 표로 비교한다.
        _print_compare(
            compare_models(
                args.normal_csv,
                args.eval_csv,
                ABLATION_MATRIX,
                validation_ratio=args.validation_ratio,
                limit=args.limit,
            )
        )
        return

    if args.command in ("train", "run"):
        _print_train(
            train_model(
                args.normal_csv,
                args.model,
                validation_ratio=args.validation_ratio,
                extractor=args.extractor,
                n_states=args.n_states,
                window=args.window,
                ensemble_size=args.ensemble,
                max_fit_windows=args.max_fit_windows,
                target_fpr=args.target_fpr,
                n_iter=args.n_iter,
                limit=args.limit,
            )
        )
    if args.command in ("evaluate", "run"):
        _print_eval(
            evaluate_model(
                args.eval_csv,
                args.model,
                target_fpr=args.target_fpr,
                limit=args.limit,
            ),
            args.model,
        )


if __name__ == "__main__":
    main()
