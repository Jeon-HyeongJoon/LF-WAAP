"""Tests for CSIC->packet reconstruction using an in-memory CSV.

Uses the example row provided for this dataset — never reads the large real CSV.
"""

import io

import csic_to_packets as c2p

HEADER = (
    "classification,Method,User-Agent,Pragma,Cache-Control,Accept,Accept-encoding,"
    "Accept-charset,language,host,cookie,content-type,connection,lenght,content,index,URL"
)
ROW = (
    'Normal,POST,Mozilla/5.0 (compatible; Konqueror/3.5; Linux) KHTML/3.5.8 (like Gecko),'
    "no-cache,no-cache,"
    '"text/xml,application/xml,application/xhtml+xml,text/html;q=0.9,text/plain;q=0.8,'
    'image/png,*/*;q=0.5","x-gzip, x-deflate, gzip, deflate","utf-8, utf-8;q=0.5, *;q=0.5",'
    "en,localhost:8080,JSESSIONID=933185092E0B668B90676E0A2B0767AF,"
    "application/x-www-form-urlencoded,Connection: close,Content-Length: 68,"
    "id=3&nombre=Vino+Rioja&precio=100&cantidad=55&B1=A%F1adir+al+carrito,0,"
    "http://localhost:8080/tienda1/publico/anadir.jsp HTTP/1.1"
)


def _records(text: str):
    import csv

    rows = list(csv.reader(io.StringIO(text)))
    header, data = rows[0], rows[1:]
    fmap = c2p._build_field_map(header)
    out = []
    for row in data:
        fields = {f: (row[i] if i < len(row) else "") for f, i in fmap.items()}
        out.append(
            c2p.PacketRecord(
                label=fields.get("label", "").strip(),
                method=fields.get("method", "").strip().upper(),
                host=fields.get("host", "").strip(),
                raw=c2p._reconstruct(fields),
            )
        )
    return out


def test_reconstructs_request_line_and_path() -> None:
    rec = _records(f"{HEADER}\n{ROW}")[0]
    text = rec.raw.decode("latin-1")
    assert text.startswith("POST /tienda1/publico/anadir.jsp HTTP/1.1\r\n")
    assert rec.label == "Normal"
    assert rec.method == "POST"


def test_headers_reconstructed_with_clean_values() -> None:
    text = _records(f"{HEADER}\n{ROW}")[0].raw.decode("latin-1")
    # inline "Header: value" prefixes are stripped, host:port preserved
    assert "Host: localhost:8080\r\n" in text
    assert "Connection: close\r\n" in text
    assert "Content-Length: 68\r\n" in text
    assert "Content-Type: application/x-www-form-urlencoded\r\n" in text
    assert "Cookie: JSESSIONID=933185092E0B668B90676E0A2B0767AF\r\n" in text


def test_body_after_blank_line() -> None:
    text = _records(f"{HEADER}\n{ROW}")[0].raw.decode("latin-1")
    head, _, body = text.partition("\r\n\r\n")
    assert body == "id=3&nombre=Vino+Rioja&precio=100&cantidad=55&B1=A%F1adir+al+carrito"


def test_byte_sequence_is_byte_values() -> None:
    seq = _records(f"{HEADER}\n{ROW}")[0].byte_sequence
    assert all(0 <= b <= 255 for b in seq)
    assert len(seq) == len(_records(f"{HEADER}\n{ROW}")[0].raw)


def test_get_request_uses_url_query_no_body() -> None:
    header = "classification,Method,host,content,URL"
    row = "Normal,GET,localhost:8080,,http://localhost:8080/tienda1/index.jsp?id=2 HTTP/1.1"
    text = _records(f"{header}\n{row}")[0].raw.decode("latin-1")
    assert text.startswith("GET /tienda1/index.jsp?id=2 HTTP/1.1\r\n")
    assert text.endswith("\r\n\r\n")  # empty body


def test_label_normalization_string_and_numeric() -> None:
    # string variant (the pasted example)
    rec = _records(f"{HEADER}\n{ROW}")[0]
    assert rec.is_normal and not rec.is_anomalous

    # numeric variant (the real csic_database.csv)
    header = "classification,Method,host,content,URL"
    normal = "0,GET,h,,http://h/a HTTP/1.1"
    anomalous = "1,GET,h,,http://h/a HTTP/1.1"
    assert _records(f"{header}\n{normal}")[0].is_normal
    assert _records(f"{header}\n{anomalous}")[0].is_anomalous


def test_stream_stops_at_eof(tmp_path) -> None:
    path = tmp_path / "mini.csv"
    path.write_text(f"{HEADER}\n{ROW}\n{ROW}\n", encoding="latin-1")
    records = list(c2p.iter_packets(path))
    assert len(records) == 2
