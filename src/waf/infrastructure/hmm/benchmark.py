"""ModelBenchmark — 여러 ModelVariant를 동일 데이터로 학습·평가해 공정 비교한다.

POC의 목표는 "HMMPayl 3 아이디어를 넣은 버전과 넣지 않은 버전 중 무엇이 더 나은가"를
같은 조건에서 재는 것이다. 그러려면 모든 변형이 '동일한 학습/평가 데이터'를 봐야 한다.
이 클래스는 데이터를 한 번만 받아 보관하고, 변형마다 새 탐지기를 학습→채점한다.

CSIC 같은 특정 데이터셋에 묶이지 않도록, 도메인 Flow 시퀀스 + 라벨(공격 여부)만 받는다.
→ 합성 데이터로도, 실제 CSIC 어댑터로도 동일하게 재사용된다.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from waf.domain.model.flow import Flow
from waf.infrastructure.hmm.evaluation import (
    ClassificationMetrics,
    EvaluationReport,
    classification_metrics,
    evaluate,
)
from waf.infrastructure.hmm.hmm_detector import HmmDetector
from waf.infrastructure.hmm.model_variant import ModelVariant


@dataclass(frozen=True, slots=True)
class ScoredMetrics:
    """학습된 탐지기를 라벨 데이터로 채점한 결과(학습 규모와 무관한 평가 지표만)."""

    report: EvaluationReport | None
    classification: ClassificationMetrics | None
    covered: int
    uncovered: int
    eval_normal: int
    eval_attack: int


def score_labeled_flows(
    detector: HmmDetector,
    labeled_eval: Sequence[tuple[Flow, bool]],
    target_fpr: float,
) -> ScoredMetrics:
    """이미 학습된 탐지기로 라벨된 Flow들을 채점해 지표를 낸다(학습/평가 공용 핵심).

    분류 지표는 모델의 '실제 파티션별 결정(blocked)'에서 뽑는다(운영점 그대로). 두 클래스가
    모두 커버될 때만 지표를 산출하고, 아니면 None(비교 불가)으로 둔다.
    """
    normal_scores: list[float] = []
    attack_scores: list[float] = []
    tp = fp = fn = tn = 0  # 양성 클래스 = 공격
    covered = 0
    uncovered = 0
    for flow, is_attack in labeled_eval:
        assessment = detector.assess_flow(flow)
        if not assessment.covered:
            uncovered += 1
            continue
        covered += 1
        if is_attack:
            attack_scores.append(assessment.score)
            tp += assessment.blocked
            fn += not assessment.blocked
        else:
            normal_scores.append(assessment.score)
            fp += assessment.blocked
            tn += not assessment.blocked

    report: EvaluationReport | None = None
    classification: ClassificationMetrics | None = None
    if normal_scores and attack_scores:
        report = evaluate(normal_scores, attack_scores, target_fpr=target_fpr)
        classification = classification_metrics(tp=tp, fp=fp, fn=fn, tn=tn)
    return ScoredMetrics(
        report=report,
        classification=classification,
        covered=covered,
        uncovered=uncovered,
        eval_normal=len(normal_scores),
        eval_attack=len(attack_scores),
    )


@dataclass(frozen=True, slots=True)
class VariantResult:
    """한 변형의 비교 결과: 학습 규모 + 평가 지표를 한데 묶는다.

    report/classification은 두 클래스(정상·공격)가 모두 커버될 때만 채워지고,
    그렇지 않으면 None(파티션 미커버로 지표 계산 불가)이다.
    """

    variant: ModelVariant
    report: EvaluationReport | None  # ROC-AUC·PR-AUC·AUCp·TPR@FPR (랭킹 지표)
    classification: ClassificationMetrics | None  # Accuracy·Precision·Recall·F1 (운영점 지표)
    train_count: int  # 학습에 쓰인 정상 flow 수
    validation_count: int  # 임계치 보정에 쓰인 검증 flow 수
    covered: int  # 평가 중 파티션 모델이 있어 채점된 flow 수
    uncovered: int  # 모델 없어 abstain된 flow 수
    eval_normal: int  # 채점된 정상 flow 수
    eval_attack: int  # 채점된 공격 flow 수


class ModelBenchmark:
    def __init__(
        self,
        normal_flows: Sequence[Flow],
        validation_flows: Sequence[Flow],
        labeled_eval_flows: Sequence[tuple[Flow, bool]],
    ) -> None:
        # 변형마다 재사용해야 하므로 리스트로 한 번 고정해 둔다(제너레이터 소진 방지).
        self._normal = list(normal_flows)
        self._validation = list(validation_flows)
        self._labeled_eval = list(labeled_eval_flows)  # (flow, is_attack)

    def evaluate_variant(self, variant: ModelVariant) -> VariantResult:
        """변형 하나를 학습→평가하고 결과를 모은다."""
        # ① 변형 설정대로 새 탐지기를 만들어 보관 데이터로 학습+보정.
        detector = variant.build_detector()
        detector.train(self._normal, self._validation)

        # ② 학습·평가 공용 채점 핵심을 재사용(파이프 평가와 같은 코드 경로).
        scored = score_labeled_flows(detector, self._labeled_eval, variant.target_fpr)

        return VariantResult(
            variant=variant,
            report=scored.report,
            classification=scored.classification,
            train_count=detector.train_count or 0,
            validation_count=detector.validation_count or 0,
            covered=scored.covered,
            uncovered=scored.uncovered,
            eval_normal=scored.eval_normal,
            eval_attack=scored.eval_attack,
        )

    def run(self, variants: Sequence[ModelVariant]) -> list[VariantResult]:
        """여러 변형을 순서대로 학습·평가해 결과 리스트를 돌려준다."""
        return [self.evaluate_variant(v) for v in variants]


def render_comparison(results: Sequence[VariantResult]) -> str:
    """변형들을 한 표로 나란히 출력한다. HMMPayl 권장 지표인 AUCp 내림차순 정렬.

    지표가 없는(파티션 미커버) 변형은 'n/a'로 표시하고 맨 아래로 보낸다.
    """

    def aucp_key(r: VariantResult) -> float:
        # 지표 없는 변형은 정렬상 가장 낮게(맨 아래) 취급.
        return r.report.partial_auc if r.report is not None else -1.0

    ranked = sorted(results, key=aucp_key, reverse=True)

    header = (
        f"{'variant':<16} {'HMMPayl':<8} {'ROC-AUC':>8} {'PR-AUC':>8} "
        f"{'AUCp':>8} {'TPR@FPR':>8} {'F1':>7} {'covered':>9}"
    )
    rule = "─" * len(header)
    lines = [rule, header, rule]
    for r in ranked:
        payl = "Y" if r.variant.uses_hmmpayl else "·"
        if r.report is None or r.classification is None:
            lines.append(
                f"{r.variant.name:<16} {payl:<8} {'n/a':>8} {'n/a':>8} "
                f"{'n/a':>8} {'n/a':>8} {'n/a':>7} {r.covered:>9}"
            )
            continue
        lines.append(
            f"{r.variant.name:<16} {payl:<8} {r.report.roc_auc:>8.3f} {r.report.pr_auc:>8.3f} "
            f"{r.report.partial_auc:>8.3f} {r.report.tpr_at_fpr:>8.3f} "
            f"{r.classification.f1:>7.3f} {r.covered:>9}"
        )
    lines.append(rule)
    return "\n".join(lines)
