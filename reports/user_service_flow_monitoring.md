상태 i에서 j로 간 전이 count를 c_ij라고 하면:

  A_i ~ Dirichlet(alpha_i1, alpha_i2, ..., alpha_iK)

  posterior:
  A_i | data ~ Dirichlet(alpha_i1 + c_i1, ..., alpha_iK + c_iK)

  posterior predictive:
  P(s_next=j | s=i, data)
  = (c_ij + alpha_ij) / (sum_j c_ij + sum_j alpha_ij)

  이게 실무적으로 가장 쓸만한 전이확률입니다.

  왜 좋은가

  1. zero probability 문제를 줄임
      - 순수 MLE는 한 번도 안 나온 전이를 0으로 둡니다.
      - 그러면 정상인데 학습 기간에 없었던 rare flow가 무조건 이상으로 터집니다.
      - Dirichlet prior는 pseudo-count 역할을 해서 희귀 전이에 완충을 줍니다.

  2. 데이터 적은 workflow에 유리
      - /checkout, /password-reset, /kyc 같은 플로우는 케이스 수가 적거나 seasonal합니다.
      - Dirichlet prior를 쓰면 count가 적은 row에서 과적합을 줄일 수 있습니다.

  3. 비즈니스 지식을 prior로 넣기 좋음
      - 단순 uniform prior가 아니라, 설계된 workflow graph를 prior에 반영할 수 있습니다.
      - 예:

        cart -> shipping      alpha=20
        shipping -> payment   alpha=20
        payment -> confirm    alpha=20
        cart -> confirm       forbidden
        payment -> cart       alpha=2
        payment -> payment    alpha=5  # retry

      - 즉, “정상 설계상 가능한 전이”와 “운영상 자주 있는 전이”를 분리해서 줄 수 있습니다.

  주의할 점

  가장 중요한 건 forbidden transition에 Dirichlet smoothing을 주면 안 됩니다.

  예를 들어:

  cart -> confirm
  anonymous -> withdraw
  password_reset_start -> password_change_without_token

  이런 건 business rule상 금지라면 alpha를 작게 주는 게 아니라 transition mask로 완전히 막아야 합니다.

  추천 구조:

  if transition in forbidden_edges:
      hard_violation = true
      risk = critical
  else:
      score = -log((c_ij + alpha_ij) / (C_i + alpha_i0))


  1. 금지 전이는 mask로 hard block
  2. 허용 전이에만 Dirichlet prior 부여
  3. posterior predictive probability로 anomaly score 계산
  4. alpha는 uniform이 아니라 workflow 지식 기반 asymmetric prior 권장
  5. HMM보다 먼저 observable workflow transition model에 적용




  --------------

  목표

  고객 웹서비스를 대상으로 사용자가 정상 업무 플로우를 따르지 않는 행위를 탐지한다.
  핵심은 payload anomaly가 아니라 비즈니스 플로우 이상 탐지다.

  예:

  장바구니 없이 결제 확정
  배송지 입력 없이 결제 요청
  본인확인 전 출금 요청
  reset token 검증 없이 비밀번호 변경
  쿠폰 중복 적용
  정상 UI 순서를 건너뛴 API 직접 호출

  전체 구조

  1. 고객/서비스별 workflow graph 정의
  2. 세션 또는 workflow_instance 단위로 이벤트 trace 구성
  3. 명시적 forbidden transition mask 적용
  4. 허용 전이에 대해 Dirichlet-smoothed transition probability 계산
  5. 구조화된 request event를 observation으로 사용
  6. Markov/HMM/LSTM 기반 anomaly score 산출
  7. rule violation + anomaly score + context risk를 결합

  중요한 설계 판단

  byte sequence 기반 모델은 주 모델로 쓰지 않는다.

  이유:

  byte sequence는 payload 형식, 인코딩, exploit-like content 탐지에 적합
  비즈니스 플로우 탐지에는 세션 상태, 업무 단계, API 순서, 인증 상태가 더 중요
  정상 payload로 순서만 우회하는 공격은 byte 모델이 잘 못 잡을 수 있음

  따라서 byte HMM은 별도 보조 계층으로 둘 수는 있지만, 핵심 모델은 아니다.

  핵심 입력 단위

  raw URL, raw parameter, raw cookie, raw session 값을 그대로 상태로 쓰지 않는다.
  대신 정규화된 이벤트로 변환한다.

  method
  route_template
  business_action
  workflow_type
  workflow_instance_id
  auth_context
  param_schema_id
  cookie_state
  session_state
  response_result
  delta_time_bucket
  object_type
  transition_result

  예:

  {
    "workflow_type": "checkout",
    "workflow_instance_id": "flow_hash_123",
    "action_id": "POST /orders/{id}/confirm",
    "business_action": "order_confirm",
    "auth_context": "authenticated_user",
    "param_schema_id": "orderId,paymentToken",
    "cookie_state": "session_valid,csrf_valid",
    "response_result": "2xx",
    "delta_time_bucket": "10s_30s",
    "transition_result": "invalid_order"
  }

  각 필드 처리 방침

   항목             사용 방식
  ━━━━━━━━━━━━━━━  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
   URL/path         route_template로 정규화
  ───────────────  ───────────────────────────────────────────────────────────────────────
   method           관측값에 포함
  ───────────────  ───────────────────────────────────────────────────────────────────────
   parameter        raw value 제외, param names/type/schema/validation result 사용
  ───────────────  ───────────────────────────────────────────────────────────────────────
   cookie           raw value 제외, session_valid, csrf_valid, cookie_changed 등으로 사용
  ───────────────  ───────────────────────────────────────────────────────────────────────
   session          상태가 아니라 trace grouping key
  ───────────────  ───────────────────────────────────────────────────────────────────────
   user_id          인증 후 grouping/context hash
  ───────────────  ───────────────────────────────────────────────────────────────────────
   IP               상태가 아니라 context/risk feature
  ───────────────  ───────────────────────────────────────────────────────────────────────
   byte sequence    payload anomaly 보조 계층

  전이 행렬

  전이 행렬은 Dirichlet prior를 사용해 추정한다.

  P(next_state=j | state=i)
  = (count_ij + alpha_ij) / (sum count_i + sum alpha_i)

  단, 모든 전이에 smoothing을 주면 안 된다.

  forbidden transition: mask 처리, probability 0
  allowed transition: Dirichlet smoothing 적용
  rare but valid transition: 작은 alpha
  expected transition: 큰 alpha
  retry/self transition: 별도 alpha

  즉:

  금지 전이 = 규칙 기반 high-confidence violation
  허용 전이 = 확률 기반 anomaly score

  HMM 사용 위치

  HMM은 사용할 수 있지만, hidden state와 observation을 명확히 나눈다.

  hidden state:
    anonymous
    authenticated
    browsing
    cart
    shipping
    payment_ready
    order_submitted
    password_reset_pending
    admin_sensitive_action

  observation:
    method + route_template + business_action + auth_context
    + param_schema_id + cookie_state + response_result

  단순한 서비스 플로우라면 HMM보다 먼저 Markov/state machine으로 시작하는 것이 좋다.
  HMM은 optional step, retry, background call, 같은 endpoint의 다중 의미처럼 노이즈가 많은 경우에 유리하다.

  권장 모델 계층

  Layer 1: Explicit Business State Machine
  - 필수 단계 누락
  - 순서 위반
  - 불가능한 전이
  - 가장 신뢰도 높은 탐지

  Layer 2: Markov / HMM Workflow Model
  - 정상 전이 확률 기반 이상 점수
  - 고객별/서비스별/워크플로우별 학습

  Layer 3: LSTM Sequence Model
  - 긴 세션, 복잡한 optional flow, 개인화 행동 탐지

  Layer 4: Context Risk
  - IP/ASN/Geo 변화
  - device fingerprint 변화
  - 요청 속도
  - 세션 생성 폭증

  Layer 5: Payload Anomaly
  - byte HMM, n-gram, signature, SQLi/XSS 탐지
  - 핵심이 아니라 보조

  최종 방향

  이 WAF의 차별점은 payload를 더 잘 보는 것이 아니라, 고객 서비스별로 정해진 업무 플로우를 이해하고 다음을 탐지하는 것이
  다.

  “이 요청이 문법적으로 정상인가?”
  보다
  “이 사용자가 지금 이 상태에서 이 요청을 하는 것이 정상인가?”

  따라서 우선 설계해야 할 것은 byte sequence 모델이 아니라:

  고객별 workflow graph
  workflow_instance 식별 방식
  정규화된 business event schema
  forbidden transition mask
  Dirichlet 기반 전이확률
  session/user 단위 trace reconstruction

  ----
  • 불확실한 부분에서만 HMM을 쓰려면, HMM을 “2차 탐지 모델”이 아니라 상태 복원기(state resolver) 로 두는 게 좋습니다.
  
    즉 전체 판단 흐름은 이렇게 잡습니다.
  
    명확한 이벤트:
      Markov Chain + rule mask로 처리
  
    불확실한 이벤트:
      HMM으로 현재 workflow state를 추정
      추정 결과를 다시 Markov/rule 판단에 연결
  
    핵심 구조
  
    Raw network log
    -> Canonical Web Event
    -> Business Action Mapper
    -> confidence 계산
    -> if confident: Markov 처리
    -> if uncertain: HMM state inference
    -> 최종 transition scoring
  
    1. Markov 기본 모델
  
    Markov Chain의 상태는 관측 가능한 business action입니다.
  
    S_t = checkout:cart_add
    S_t = checkout:shipping_submit
    S_t = checkout:payment_submit
    S_t = checkout:order_confirm
  
    전이:
  
    P(S_t+1 | S_t)
  
    전이확률은 Dirichlet smoothing을 적용합니다.
  
    P(j | i) = (count_ij + alpha_ij) / (sum count_i + sum alpha_i)
  
    단, 금지 전이는 확률 모델에 넣지 않습니다.
  
    cart_add -> order_confirm = forbidden mask
  
    이 부분은 OWASP/CWE-841 관점에서 “workflow enforcement”에 해당하므로 확률이 아니라 규칙입니다.
    근거: CWE-841 (https://cwe.mitre.org/data/definitions/841.html), OWASP Business Logic Security
    (https://cheatsheetseries.owasp.org/cheatsheets/Business_Logic_Security_Cheat_Sheet.html)
  
    2. HMM을 호출하는 조건
  
    HMM은 모든 요청에 쓰지 말고 아래 조건에서만 호출합니다.
  
    route_template confidence가 낮음
    business_action 후보가 2개 이상
    session/workflow_instance 복원이 불확실함
    필드가 UNKNOWN_NOT_OBSERVED로 많이 비어 있음
    같은 endpoint가 여러 workflow에서 재사용됨
    Markov 전이는 매우 낮은 확률인데 hard violation은 아님
    중간 이벤트가 누락된 것처럼 보임
  
    예:
  
    POST /submit
  
    이게 payment_submit인지 kyc_submit인지 profile_update인지 애매하면 HMM을 호출합니다.
  
    3. HMM 설계
  
    HMM의 hidden state는 business action이 아니라 업무 상태입니다.
  
    Z_t =
      anonymous
      authenticated
      browsing
      cart_open
      shipping_ready
      payment_ready
      submitted
      reset_pending
      admin_sensitive
  
    observation은 정규화된 이벤트 필드입니다.
  
    O_t =
      method
      route_template
      action_candidate
      param_schema_id
      auth_state
      session_state
      csrf_state
      status_class
      delta_time_bucket
  
    중요한 점은 observation을 하나의 거대한 문자열로 합치지 않는 것입니다.
    필드별 emission으로 나눕니다.
  
    P(O_t | Z_t)
    = P(route_template | Z_t)
    * P(method | Z_t)
    * P(param_schema_id | Z_t)
    * P(auth_state | Z_t)
    * P(session_state | Z_t)
    * P(status_class | Z_t)
  
    그래야 null 처리와 sparse data 문제가 줄어듭니다.
  
    4. HMM 전이행렬
  
    HMM 전이행렬도 workflow graph에서 시작합니다.
  
    cart_open -> shipping_ready
    shipping_ready -> payment_ready
    payment_ready -> submitted
  
    Dirichlet prior를 둡니다.
  
    A_i ~ Dirichlet(alpha_i)
  
    학습 후:
  
    A_ij = (expected_count_ij + alpha_ij) /
           (sum expected_count_i + sum alpha_i)
  
    하지만 Markov와 동일하게 forbidden edge는 mask 처리합니다.
  
    A_ij = 0 if forbidden(i, j)
  
    HMM이 금지 전이를 “확률적으로 조금 허용”하게 만들면 안 됩니다.
  
    5. null 처리
  
    null은 단순 NULL 토큰 하나로 넣지 않습니다.
  
    NOT_APPLICABLE
    UNKNOWN_NOT_OBSERVED
    MISSING_BUT_REQUIRED
    EMPTY_VALUE
    PARSE_ERROR
    REDACTED
    UNMAPPED
  
    처리 방식:
  
    NOT_APPLICABLE:
      emission 계산에서 제외
  
    UNKNOWN_NOT_OBSERVED:
      emission 계산에서 제외하고 confidence 감소
  
    MISSING_BUT_REQUIRED:
      rule violation
  
    EMPTY_VALUE:
      별도 관측 토큰
  
    PARSE_ERROR:
      anomaly feature
  
    UNMAPPED:
      HMM fallback 대상
  
    예:
  
    P(O_t | Z_t)
    = P(route | Z_t)
    * P(method | Z_t)
    * P(auth_state | Z_t)
  
    param_schema_id가 UNKNOWN_NOT_OBSERVED이면 그 항은 곱하지 않습니다.
  
    confidence *= 0.8
  
    하지만 order_confirm에 paymentToken이 없으면 unknown이 아니라 MISSING_BUT_REQUIRED입니다.
  
    risk += high
  
    6. inference 방식
  
    HMM은 불확실한 구간만 봅니다.
  
    예:
  
    확실: cart_add
    불확실: POST /submit
    불확실: GET /status
    확실: order_confirm
  
    이때 전체 세션을 다시 HMM에 넣지 말고, 주변 anchor를 포함한 짧은 window만 봅니다.
  
    window = previous_confirmed_state + uncertain_events + next_confirmed_state
  
    Viterbi로 가장 가능성 높은 hidden state path를 구합니다.
  
    cart_open -> payment_ready -> payment_pending -> submitted
  
    그리고 결과를 Markov state로 다시 변환합니다.
  
    hidden state payment_ready
    + observation POST /submit
    => business_action payment_submit
  
    7. 최종 점수
  
    최종 risk는 이렇게 합칩니다.
  
    risk =
      rule_violation_score
    + markov_transition_score
    + hmm_uncertainty_score
    + mapping_confidence_penalty
    + context_risk
  
    구체적으로:
  
    markov_transition_score = -log P(S_t | S_t-1)
  
    hmm_uncertainty_score =
      -log P(O_window | model) / window_length
  
    mapping_confidence_penalty =
      1 - mapper_confidence
  
    차단 판단은 rule violation을 최우선으로 둡니다.
  
    if forbidden_transition:
        block or high alert
  
    elif risk > threshold:
        alert or challenge
  
    else:
        allow and update state
  
    8. 학습 방식
  
    고객 규칙이 대부분 정해져 있다면 완전 비지도 HMM보다 weakly-supervised HMM이 낫습니다.
  
    초기값:
  
    hidden states = workflow graph nodes
    transition prior = 고객 workflow graph
    emission prior = route/action mapping rule
    forbidden mask = 고객 rule
  
    학습 데이터:
  
    정상 운영 로그
    정상 QA 시나리오
    고객 API 명세/OpenAPI
    gateway route config
    WAF allowlist
  
    학습 절차:
  
    1. 확실한 이벤트는 Markov count로 학습
    2. 불확실한 이벤트는 HMM forward-backward로 expected count 계산
    3. 전이행렬은 Dirichlet posterior로 갱신
    4. emission도 필드별 Dirichlet prior로 갱신
    5. forbidden edge는 학습에서 제외
  
    최종 설계 요약
  
    Markov Chain:
      정상 business action 전이의 주 모델
  
    HMM:
      action/state 매핑이 불확실한 구간의 보조 state resolver
  
    Rule Mask:
      금지 전이의 최종 판단자
  
    Dirichlet:
      허용 전이와 emission의 smoothing
  
    Null-aware Mapping:
      관측 불가와 필수값 누락을 분리
  
    이렇게 하면 Markov의 설명 가능성과 단순성을 유지하면서, 네트워크 로그 기반 자동 매핑의 불확실성은 HMM으로 흡수할 수 있
    습니다.
