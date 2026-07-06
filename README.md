# LF-WAAP — POC WAF (DDD core)

룰셋 기반 + HMM 기반 차단을 결합하는 테스트용 WAF. 지금은 **프록시 없이 도메인 코어**만
구현되어 있고, 나중에 mitmproxy/FastAPI 어댑터를 `infrastructure/proxy`에 끼우면 된다.

## 설계 (DDD)

```
src/waf/
  domain/                  # 순수 비즈니스 규칙 (I/O·프레임워크 의존 없음)
    model/                 # 값 객체: HttpRequest, Flow, Verdict, DetectionSignal
    detector/              # 포트: Detector (Protocol)
    service/               # BlockDecisionService — 차단 결정 도메인 서비스
  application/             # InspectRequest — 유스케이스 (얇은 진입점)
  infrastructure/          # 어댑터 (구현체)
    ruleset/               # RuleSetDetector (+ rules.yaml)
    hmm/                   # 바이트 단위 one-class HMM IDS (아래 참고)
    proxy/                 # (예정) mitmproxy / FastAPI 어댑터
tests/
demo.py                    # 코어 동작 + IDS 지표 확인용
```

핵심: 두 차단 방식이 모두 `Detector` 포트를 구현하고, `BlockDecisionService`가
정책(`ANY`/`ALL`)으로 조합한다. → 새 탐지 전략 추가가 도메인 결정 로직을 안 건드림.

## 동작 방식

- **RuleSetDetector**: `rules.yaml`의 정규식을 요청 페이로드(path+query+body)에 매칭. SQLi/XSS/LFI/RCE.
- **HmmDetector**: *정상* 트래픽만으로 학습한 **one-class 바이트 단위 HMM**으로 이상 탐지.
  룰이 못 잡는 변형/난독화/바이너리 공격 커버.

### HMM 서브시스템 설계 (PAYL / Anagram / HMMPayl 계열)

```
hmm/
  features.py          # 관측 심볼화: RawByte(256) · ByteClass(8) · WindowToken(n-gram) · FlowFeature(암호화)
  partition.py         # PartitionKey: (protocol, port, direction, length-bucket)
  normal_hmm.py        # 단일 파티션 one-class HMM (latent-phase 상태, FPR 보정 임계치)
  partitioned_hmm.py   # 파티션별 모델 학습/라우팅
  evaluation.py        # ROC-AUC · PR-AUC · TPR@FPR · FP/period
  hmm_detector.py      # WAF Detector 어댑터 (HttpRequest→Flow)
  http_flow.py         # HttpRequest → Flow 매퍼
```

확정된 모델링 원칙:

1. **관측 = raw byte 0~255** 가 가장 직접적. n-gram은 256ⁿ 차원폭발 →
   raw byte / byte class / windowed token / clustered token부터 시작 (`features.py`).
2. **숨은 상태는 normal/attack 라벨이 아니라 payload/flow의 잠재 국면.**
   one-class 학습(정상만) + 저우도 차단. `n_states`는 검증 지표로 고르는 용량 하이퍼파라미터.
3. **모델 분리**: (protocol, port, direction, length bucket)별 별도 HMM →
   각 모델의 "정상" 분포가 좁아져 낮은 목표 FPR 달성 가능.
4. **score = log P(seq | normal_HMM) / len(seq)** (길이 정규화 로그우도).
5. **임계치는 검증셋에서 목표 FPR로 보정** (`calibrate(val, target_fpr)`),
   anomaly ⇔ score < threshold.
6. **암호화 트래픽**이면 payload byte 대신 `FlowFeatureExtractor`
   (direction + length bin + inter-arrival bin + TCP flag 조합).
7. **2차 HMM은 기본 아님** — 1차가 검증에서 명확히 부족할 때만 비교 후보.
   (실용 경로로 `WindowTokenExtractor`로 국소 고차 구조를 먼저 포착)
8. **평가는 AIC/BIC가 아니라 IDS 지표** — ROC-AUC, PR-AUC, TPR@fixed FPR,
   FP per period (`evaluation.py`). KDD/NSL-KDD/DARPA 데이터셋 caveat 유의.

## 모델 학습 / 구동 플로우 (코드 위치 포함)

세 플로우가 동일 컴포넌트(`FeatureExtractor`·`PartitionedHmm`·`NormalHMM`)를 공유하고
진입점만 다르다. 각 단계 옆 `파일:줄`은 실제 호출되는 함수/클래스 위치다.

### ① 학습 (`train`) — 정상 트래픽 적합 + 임계치 보정

```
train_model()                                   learning_data/calibrate_cli.py:41
 ├─ iter_packets(normal_csv)                     learning_data/csic_to_packets.py:156  → PacketRecord(:78)
 ├─ CsicTrafficSource(validation_ratio)          learning_data/csic_traffic_source.py:42
 │    ├─ .normal()      (:69)  ┐ record_to_flow()(:32) → Flow
 │    └─ .validation()  (:74)  ┘
 ├─ HmmDetector(extractor, n_states, window…)    src/waf/infrastructure/hmm/hmm_detector.py:22
 └─ CalibrateModel.__call__(detector)            src/waf/application/calibrate_model.py:28
      ├─ detector.train(normal, validation)      hmm_detector.py:91
      │   └─ PartitionedHmm.train()              src/waf/infrastructure/hmm/partitioned_hmm.py:109
      │        ├─ partition_key(flow)            src/waf/infrastructure/hmm/partition.py:38  (proto,port,dir,len-bucket)
      │        ├─ FeatureExtractor.extract()     src/waf/infrastructure/hmm/features.py  (ByteClass:91 / RawByte:59 / WindowToken:113)
      │        ├─ _new_model()                   partitioned_hmm.py:49  → NormalHMM(:59) | EnsembleNormalHMM(:22)
      │        ├─ model.fit(seqs)                normal_hmm.py:96  (Baum–Welch + Laplace smoothing :121)
      │        └─ model.calibrate(val,fpr)       normal_hmm.py:170  (임계치 = 정상점수 target_fpr 분위수 :183)
      └─ PickleModelRepository.save(detector)    src/waf/infrastructure/adapter/persistence/pickle_model_repository.py:20
           → CalibrationReport(normal_count, validation_count)   calibrate_model.py:18
```

- 파티션별 학습 표본 `< min_train_sequences`이거나 검증 표본이 없으면 건너뜀(런타임 fail-open). `partitioned_hmm.py:127`
- one-class: **정상 데이터만** 적합. 공격 데이터는 학습에 쓰지 않음.
- 윈도우 채점이면 `sliding_windows()`(`normal_hmm.py:49`)로 분할, 앙상블이면 멤버별 랜덤 재시작(`ensemble_hmm.py:40`).

### ② 평가 (`evaluate`) — 저장 모델 채점 → IDS 지표

```
evaluate_model()                                 learning_data/calibrate_cli.py:103
 ├─ PickleModelRepository.load()                 pickle_model_repository.py:24   → HmmDetector 복원
 └─ for record in iter_packets(eval_csv):        csic_to_packets.py:156
      ├─ record_to_flow(record)                  csic_traffic_source.py:32
      ├─ HmmDetector.assess_flow(flow)           hmm_detector.py:94
      │    └─ PartitionedHmm.assess()            partitioned_hmm.py:134
      │         ├─ partition_key → 파티션 모델 라우팅          partition.py:38
      │         ├─ 보정 모델 없음 → Assessment(covered=False)  partitioned_hmm.py:17  (fail-open)
      │         └─ model.is_anomaly(extract(flow)) → (blocked, score)   normal_hmm.py:186
      ├─ 정상/공격 점수 누적 → evaluate()         src/waf/infrastructure/hmm/evaluation.py:142
      │    (roc_auc·pr_auc / partial_auc:99 / tpr_at_fpr:86 → EvaluationReport:23)
      ├─ 파티션별 blocked 누적 → classification_metrics()      evaluation.py:63  → ClassificationMetrics:43
      └─ _score_stats() per-class likelihood     calibrate_cli.py:80  → ScoreStats:70
 → EvaluationOutcome(:90)  →  _print_eval() 리포트 카드        calibrate_cli.py:167
```

### ③ 런타임 차단 — `Detector` 포트로 WAF에 결합

```
HttpRequest                                      src/waf/domain/model/http_request.py:15
 └─ InspectRequest.__call__(request)             src/waf/application/inspect_request.py:20
      └─ BlockDecisionService.decide(request)     src/waf/domain/service/block_decision_service.py:36
           ├─ HmmDetector.inspect(request)        hmm_detector.py:97
           │    ├─ http_request_to_flow(request)  src/waf/infrastructure/hmm/http_flow.py:26
           │    ├─ 미학습 → fail-open abstain      hmm_detector.py:99
           │    └─ PartitionedHmm.assess → DetectionSignal(blocked,reason,score)   src/waf/domain/model/detection.py:9
           ├─ RuleSetDetector.inspect(request)    src/waf/infrastructure/ruleset/ruleset_detector.py:46
           └─ 정책(ANY/ALL)으로 신호 조합 → Verdict   block_decision_service.py:18 / src/waf/domain/model/verdict.py:17
```

핵심: 세 플로우 모두 `Detector` 포트(`src/waf/domain/detector/detector.py:17`) 뒤의 동일 모델을
쓴다. 학습·평가는 CSIC 어댑터로 오프라인, 차단은 같은 `.pkl`로 온라인 → "검증한 모델 = 배포하는 모델".

## 실행

```bash
pip install -e ".[dev]"     # 또는: uv pip install -e ".[dev]"
pytest                      # 테스트
python demo.py             # 데모
```

### HMM 학습 / 평가 (CSIC 2010)

엔드투엔드 파이프라인은 `learning_data/calibrate_cli.py`로 실행한다.
세 서브커맨드: `train`(정상 트래픽으로 학습 + 임계치 보정) · `evaluate`(저장된 모델
채점 → IDS 지표) · `run`(train 후 evaluate). CSV 경로는 기본값이 채워져 있다.

```bash
cd learning_data

# 1) 학습 — 정상 트래픽만으로 one-class HMM 학습 + target-fpr로 임계치 보정
python calibrate_cli.py train \
    --normal-csv csic_database_normal.csv \
    --model hmm_bc.pkl \
    --extractor byte_class --n-states 8 --target-fpr 0.05

# 2) 평가 — 저장된 모델로 혼합셋 채점, 리포트 카드 출력
python calibrate_cli.py evaluate \
    --eval-csv csic_database.csv \
    --model hmm_bc.pkl --target-fpr 0.05

# 3) run — 학습+평가를 한 번에
python calibrate_cli.py run --model hmm_bc.pkl --target-fpr 0.05
```

주요 옵션:

| 옵션 | 의미 | 기본값 |
|------|------|--------|
| `--extractor` | 관측 심볼화: `byte_class`(8) · `raw_byte`(256) · `window_token` | `byte_class` |
| `--n-states` | HMM 숨은 상태(잠재 국면) 수 | `8` |
| `--window N` | HMMPayl 슬라이딩 윈도우 폭 (`0`=off) | `0` |
| `--ensemble K` | 랜덤 재시작 K-앙상블 | `1` |
| `--max-fit-windows` | 학습 윈도우 상한(샘플링, 비용 제어) | 없음 |
| `--target-fpr` | 임계치 보정/지표의 목표 오탐률 | `0.01` |
| `--n-iter` | Baum–Welch 반복 횟수 | `100` |
| `--limit N` | 읽을 CSV 행 수 제한(빠른 스모크 테스트) | 없음 |

`evaluate`/`run` 출력은 `reports/content.md`가 요구하는 수치(모델 구조·학습/테스트
데이터 수·Accuracy/Precision/Recall/F1·ROC-AUC·클래스별 likelihood)를 섹션별 리포트
카드로 보여준다.

### 옵션별 예시 명령어

각 옵션이 무엇을 바꾸는지 한 줄 설명과 함께. 모두 `learning_data/`에서 실행한다.

```bash
# ── --extractor : 관측 심볼화 방식 교체 ──────────────────────────────
# 256-심볼 raw byte. 표현력은 크지만 표본이 희소해 학습이 어렵다(아래 검증 결과 참고).
python calibrate_cli.py run --model hmm_raw.pkl --extractor raw_byte --target-fpr 0.05
# 8-심볼 byte class(기본·권장). 적은 데이터로도 "정상" 분포를 촘촘히 학습.
python calibrate_cli.py run --model hmm_bc.pkl  --extractor byte_class --target-fpr 0.05

# ── --n-states : 숨은 상태(잠재 국면) 수 = 모델 용량 ─────────────────
# 상태를 늘리면 payload 내부 국면을 더 세분화하지만 과적합·학습비용↑. 검증 지표로 고른다.
python calibrate_cli.py run --model hmm_s16.pkl --n-states 16 --target-fpr 0.05

# ── --window : HMMPayl 슬라이딩 윈도우 채점 ──────────────────────────
# payload를 길이-6 부분수열로 쪼개 각각 채점 후 평균 fusion. 적은 상태수로 국소 구조 포착.
python calibrate_cli.py run --model hmm_w6.pkl --window 6 --n-states 6 --target-fpr 0.05

# ── --ensemble : 랜덤 재시작 K-앙상블 ────────────────────────────────
# 서로 다른 초기화로 학습한 3개 HMM 점수를 결합 → 국소최적 완화, 회피 저항↑.
python calibrate_cli.py run --model hmm_e3.pkl --ensemble 3 --target-fpr 0.05

# ── --max-fit-windows : 학습 윈도우 상한(샘플링) ─────────────────────
# 윈도우 채점은 payload당 ~L개 윈도우를 만든다. 상한을 두어 대용량 학습 비용을 제어.
python calibrate_cli.py run --model hmm_w6.pkl --window 6 --max-fit-windows 50000 --target-fpr 0.05

# ── --target-fpr : 운영점(목표 오탐률) ──────────────────────────────
# 임계치 보정과 지표 산출 기준. 낮출수록 오탐↓·미탐↑(저FPR 구간에서 탐지율 평가).
python calibrate_cli.py evaluate --model hmm_bc.pkl --target-fpr 0.01   # 엄격(오탐 1%)
python calibrate_cli.py evaluate --model hmm_bc.pkl --target-fpr 0.05   # 완화(오탐 5%)

# ── --n-iter : Baum–Welch 반복 횟수(학습 수렴) ──────────────────────
# 늘리면 수렴↑·학습시간↑. 빠른 실험은 30 정도로 줄인다.
python calibrate_cli.py train --model hmm_bc.pkl --n-iter 200 --target-fpr 0.05

# ── --limit : 읽을 CSV 행 수 제한(빠른 스모크 테스트) ───────────────
# 앞쪽 2000행만 사용 + 반복 30회 → 파이프라인 동작을 수 초 안에 확인.
python calibrate_cli.py run --model /tmp/smoke.pkl --limit 2000 --n-iter 30 --target-fpr 0.1

# ── 종합: HMMPayl 실험 설정(raw_byte + 윈도우6 + 3-앙상블) ──────────
python calibrate_cli.py run --model hmm_payl.pkl \
    --extractor raw_byte --window 6 --ensemble 3 --n-states 6 --target-fpr 0.05
```

> ⚠️ 검증 결과(2026-06-29): `byte_class` 단일 모델(ROC 0.653 / TPR@5%FPR 0.361)이
> `raw_byte + window=6 + ensemble=3`(ROC 0.619 / TPR@5%FPR 0.106)보다 모든 지표에서
> 낫다. 256-심볼 raw_byte는 표본이 너무 희소해 윈도우·앙상블로도 보완되지 않음.

### 모델 버전 비교 (`compare`) — ablation 매트릭스

여러 모델 버전을 **같은 데이터로 한 번에** 학습·평가해 나란히 비교한다. HMMPayl 3 아이디어
(①슬라이딩 윈도우 ②K-앙상블 ③AUCp 지표)를 넣은 버전과 넣지 않은 버전의 성능 차이를
공정하게 잰다. AUCp(저FPR 운영점 지표) 내림차순으로 정렬해 출력한다.

```bash
cd learning_data
python calibrate_cli.py compare --eval-csv csic_database.csv   # 전체 ablation 매트릭스
python calibrate_cli.py compare --limit 3000                   # 빠른 스모크
```

비교 대상 프리셋(`ABLATION_MATRIX`, `src/waf/infrastructure/hmm/model_variant.py`):

| 변형 | 추출기 | window | ensemble | HMMPayl? |
|------|--------|:------:|:--------:|:--------:|
| `bc_baseline` | byte_class | off | 1 | · |
| `bc_window` | byte_class | 6 | 1 | ✓ |
| `bc_ensemble` | byte_class | off | 3 | ✓ |
| `bc_hmmpayl` | byte_class | 6 | 3 | ✓ |
| `raw_baseline` | raw_byte | off | 1 | · |
| `raw_hmmpayl` | raw_byte | 6 | 3 | ✓ |

앞 4개(byte_class 2×2)는 두 아이디어의 **순수 기여**를 분리하고, 뒤 2개(raw_byte 양 끝점)는
**추출기 효과**까지 같은 표에서 비교한다. 출력 예:

```
variant          HMMPayl   ROC-AUC   PR-AUC     AUCp  TPR@FPR      F1   covered
bc_baseline      ·           0.653    0.647    0.235    0.361   0.…       …
bc_ensemble      ✓           0.6..    …        …        …       …         …
...
```

**재사용 구조 (OOP):**
- `ModelVariant`(`model_variant.py`) — 이름 붙은 하이퍼파라미터 묶음. `build_detector()`로 동일 설정 탐지기 생성, `uses_hmmpayl`로 아이디어 적용 여부 표시. frozen dataclass라 안전 공유.
- `ModelBenchmark`(`benchmark.py`) — 도메인 `Flow`+라벨만 받아 변형들을 같은 데이터로 학습·평가(`run`). CSIC 비종속 → 합성/실데이터 모두 재사용. `render_comparison`이 비교 표 생성.
- CLI `compare_models`(`calibrate_cli.py`)는 CSIC 어댑터를 벤치마크에 주입하는 얇은 배선층.

## HMMPayl 기법 (구현 완료)

CLI/`HmmDetector` 파라미터로 제어:

- **① 슬라이딩 윈도우 채점** (`--window n`): payload를 길이-n 부분수열로 쪼개 각각
  HMM 채점 후 확률 산술평균 fusion (eq.6–8). 적은 상태수로 동작. `NormalHMM(window=n)`.
- **② K-앙상블** (`--ensemble K`): 랜덤 재시작 K개 HMM을 학습 후 점수 결합(mean/min/max).
  `EnsembleNormalHMM`.
- **③ Partial AUC** (`evaluation.partial_auc`): 저FPR 구간 [0, max_fpr] AUC를 max_fpr로
  정규화. 리포트에 `AUCp@` 로 출력. (랜덤≈max_fpr/2, 완벽=1.0)

⚠️ 성능: 슬라이딩 윈도우 채점은 payload당 윈도우 수만큼 HMM.score를 호출 → 대용량
평가에서 느림. 벡터화가 다음 최적화 후보.

## Kafka 학습/평가/추론 파이프

**멘탈 모델: 데이터를 파이프(토픽)에 넣으면 → 학습/평가/추론이 돈다.**
학습/평가/추론 로직은 모두 그대로 두고, 데이터 출처만 메시지 파이프로 바꾼 것이다.
포트/어댑터 덕분에 학습 코드(`CalibrateModel`)는 **한 줄도 안 바뀐다** — `TrafficSource`
어댑터만 CSIC CSV → Kafka 토픽으로 교체된다.

### 도메인별 토픽 분리

| 토픽 | 도메인 | 넣는 것 | 쓰는 곳 |
|------|--------|---------|---------|
| `waf.train.normal` | 학습 | 정상 Flow | 모델 적합 |
| `waf.train.validation` | 학습 | 검증 Flow | 임계치 보정 |
| `waf.eval.labeled` | 모델 테스트 | (Flow, 공격여부) | 지표 산출 |
| `waf.inspect.requests` | 실시간 추론 | HttpRequest | 차단 판정 입력 |
| `waf.inspect.verdicts` | 실시간 추론 | 판정 결과 | 차단 판정 출력 |

### 구성 (각 도메인 = 각 모듈, 모두 `src/waf/infrastructure/streaming/`)

```
channel.py         # MessageChannel 포트(publish/poll/stream) — Kafka/인메모리 좌석
memory.py          # InMemoryChannel — 브로커 없는 페이크(테스트·demo)
kafka_channel.py   # KafkaChannel — 실제 Kafka 어댑터(kafka-python, 동일 계약)
codecs.py          # Flow/HttpRequest/Verdict ↔ JSON 바이트 직렬화
traffic_source.py  # StreamingTrafficSource — 학습 토픽 → TrafficSource 포트 구현
evaluation.py      # StreamingLabeledSource + evaluate_stream — 평가 토픽 채점
inference.py       # StreamingInspector — 요청 토픽 소비 → BlockDecisionService → 판정 발행
```

핵심: 코어/테스트는 Kafka 없이도 `InMemoryChannel`로 전부 돈다(브로커는 실행 시에만 필요).

### 실행

```bash
# 0) 의존성 + 브로커
pip install -e ".[kafka]"
docker compose up -d                       # KRaft Kafka (Zookeeper 없음), localhost:9092

cd learning_data
# 1) 데이터를 파이프에 넣기 (producer)
python kafka_pipeline.py produce-train --limit 4000   # CSIC 정상 → 학습/검증 토픽
python kafka_pipeline.py produce-eval  --limit 4000   # CSIC 혼합 → 평가 토픽(라벨 포함)

# 2) 파이프에서 학습 → 모델 저장
python kafka_pipeline.py train --model hmm_kafka.pkl --extractor byte_class --target-fpr 0.05

# 3) 파이프에서 평가 → 지표 출력
python kafka_pipeline.py evaluate --model hmm_kafka.pkl --target-fpr 0.05

docker compose down                        # 브로커 정리
```

### 브로커 없이 전체 흐름 보기 (가장 빠른 이해 경로)

```bash
cd learning_data
python kafka_pipeline.py demo
```

`demo`는 한 프로세스에서 `InMemoryChannel`로 **① 발행 → ② 학습 → ③ 평가 → ④ 실시간 추론**을
순서대로 보여준다(도커·Kafka 불필요). "파이프에 넣으면 학습·평가·추론이 된다"는 흐름을
코드로 따라가며 이해하기에 가장 좋다.

## 다음 단계

1. 슬라이딩 윈도우 채점 벡터화 (성능)
2. `infrastructure/proxy/` 에 mitmproxy addon 추가 → 실제 트래픽 가로채기
3. extractor/window/ensemble/n_states 를 AUCp·TPR@FPR로 튜닝
4. 룰셋 확장 (OWASP CRS 참고), 정책/임계치 설정 외부화
5. CSIC → `waf.inspect.requests` producer 추가(실데이터 실시간 추론)
