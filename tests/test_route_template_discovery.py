from waf.domain.model.http_request import HttpRequest
from waf.infrastructure.behavior.route_template import RouteTemplateResolver


def test_route_template_resolver_normalizes_variable_path_segments() -> None:
    resolver = RouteTemplateResolver()

    route = resolver.resolve(HttpRequest("get", "/api/v1/orders/12345/items/a8f91c2e"))

    assert route.method == "GET"
    assert route.template == "/api/v1/orders/{num}/items/{id}"
    assert route.key == "GET /api/v1/orders/{num}/items/{id}"
