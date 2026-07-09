from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.behavior.endpoint_signature import EndpointSignatureBuilder
from waf.infrastructure.behavior.route_template import RouteTemplateResolver


def test_endpoint_signature_uses_schema_not_raw_values() -> None:
    route_resolver = RouteTemplateResolver()
    builder = EndpointSignatureBuilder()
    request = HttpRequest(
        "post",
        "/submit",
        query="ref=email&campaign=summer",
        headers={"Content-Type": "application/x-www-form-urlencoded", "Authorization": "Bearer x"},
        body="orderId=123&paymentToken=secret",
    )

    signature = builder.build(request, route_resolver.resolve(request))

    assert signature.method == "POST"
    assert signature.route_template == "/submit"
    assert signature.query_keys == ("campaign", "ref")
    assert signature.body_keys == ("orderId", "paymentToken")
    assert signature.param_schema_id == "body:orderId,paymentToken|query:campaign,ref"
    assert signature.auth_state == "authenticated"
