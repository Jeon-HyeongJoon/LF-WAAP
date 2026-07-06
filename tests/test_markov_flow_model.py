from waf.domain.model.flow import Flow
from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.behavior.markov_flow import (
    START_STATE,
    MarkovFlowModel,
    WebFlowStateExtractor,
)
from waf.infrastructure.hmm.http_flow import http_request_to_flow


def _request(method: str, path: str, query: str = "") -> HttpRequest:
    return HttpRequest(method, path, query=query, headers={"Host": "shop.example.com"})


def _flow(method: str, path: str, query: str = "") -> Flow:
    return http_request_to_flow(_request(method, path, query))


def _normal_sequence(user_id: int) -> list[Flow]:
    return [
        _flow("GET", "/"),
        _flow("GET", "/login"),
        _flow("POST", "/login"),
        _flow("GET", f"/products/{user_id}", "ref=home"),
        _flow("POST", "/cart"),
        _flow("POST", "/checkout"),
    ]


def test_state_extractor_normalizes_http_request_shape() -> None:
    extractor = WebFlowStateExtractor()

    state = extractor.state_from_request(_request("GET", "/Products/12345", "b=2&a=1"))

    assert state == "GET /products/{num}?a,b"


def test_markov_model_blocks_unseen_web_flow_transition() -> None:
    extractor = WebFlowStateExtractor()
    model = MarkovFlowModel(target_fpr=0.1)
    normal = [_normal_sequence(i) for i in range(20)]
    validation = [_normal_sequence(i) for i in range(20, 25)]
    model.fit_from_flows(normal, validation, extractor)

    home = extractor.state_from_flow(_flow("GET", "/"))
    login = extractor.state_from_flow(_flow("GET", "/login"))
    admin_delete = extractor.state_from_flow(_flow("POST", "/admin/delete"))

    assert not model.assess_transition(START_STATE, home).blocked
    assert not model.assess_transition(home, login).blocked
    assert model.assess_transition(home, admin_delete).blocked


def test_markov_model_keeps_observed_mid_flow_branches_normal() -> None:
    home = "GET /"
    login = "GET /login"
    login_submit = "POST /login"
    product = "GET /products/{num}?page,ref"
    search = "GET /search?page,q"
    account = "GET /account"
    cart = "POST /cart"
    checkout = "POST /checkout"
    admin_delete = "POST /admin/delete"

    train = (
        [[home, login, login_submit, product, cart, checkout] for _ in range(12)]
        + [[home, login, login_submit, search, cart, checkout] for _ in range(3)]
        + [[home, login, login_submit, account, cart, checkout] for _ in range(2)]
        + [[home, login, login_submit, checkout, cart, checkout]]
    )
    validation = [
        [home, login, login_submit, product, cart, checkout],
        [home, login, login_submit, search, cart, checkout],
        [home, login, login_submit, account, cart, checkout],
        [home, login, login_submit, checkout, cart, checkout],
    ]
    model = MarkovFlowModel(target_fpr=0.1)

    model.fit(train, validation)

    assert not model.assess_transition(login_submit, account).blocked
    assert not model.assess_transition(login_submit, checkout).blocked
    assert model.assess_transition(login_submit, admin_delete).blocked
