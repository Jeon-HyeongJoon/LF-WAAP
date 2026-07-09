from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.behavior.action_catalog import ActionCatalog
from waf.infrastructure.behavior.endpoint_signature import EndpointSignatureBuilder
from waf.infrastructure.behavior.route_template import RouteTemplateResolver


def _signature(method: str, path: str, body: str):
    request = HttpRequest(
        method,
        path,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        body=body,
    )
    route = RouteTemplateResolver().resolve(request)
    return EndpointSignatureBuilder().build(request, route)


def test_action_catalog_reuses_stable_action_id_for_same_signature() -> None:
    catalog = ActionCatalog(service_id="shop")
    signature = _signature("POST", "/submit", "orderId=123&paymentToken=tok")

    first = catalog.recognize(signature)
    second = catalog.recognize(signature)
    from_fresh_catalog = ActionCatalog(service_id="shop").recognize(signature)

    assert first.action_id.startswith("auto:shop:act_")
    assert second.action_id == first.action_id
    assert from_fresh_catalog.action_id == first.action_id
    assert second.confidence == 1.0


def test_action_catalog_splits_same_route_with_different_parameter_schema() -> None:
    catalog = ActionCatalog(service_id="shop")

    payment = catalog.recognize(_signature("POST", "/submit", "orderId=123&paymentToken=tok"))
    profile = catalog.recognize(_signature("POST", "/submit", "displayName=hope"))

    assert payment.action_id != profile.action_id
