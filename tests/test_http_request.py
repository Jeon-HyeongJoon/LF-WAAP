import pytest

from waf.domain.model.http_request import HttpRequest


def test_http_request_normalizes_method_and_header_names() -> None:
    request = HttpRequest(
        method="get",
        path="/products",
        headers={"Host": "shop.example.com", "X-Request-Id": "req-1"},
    )

    assert request.method == "GET"
    assert request.headers["host"] == "shop.example.com"
    assert request.headers["x-request-id"] == "req-1"


@pytest.mark.parametrize(
    ("field", "method", "path"),
    [
        ("method", "", "/products"),
        ("method", " GET", "/products"),
        ("method", "GET ", "/products"),
        ("path", "GET", ""),
        ("path", "GET", " /products"),
        ("path", "GET", "/products "),
    ],
)
def test_http_request_rejects_invalid_required_fields(
    field: str, method: str, path: str
) -> None:
    with pytest.raises(ValueError, match=field):
        HttpRequest(method=method, path=path)


def test_http_request_rejects_non_string_header_values() -> None:
    with pytest.raises(ValueError, match="headers"):
        HttpRequest(method="GET", path="/products", headers={"X-Retry": 1})
