from pathlib import Path


def test_observatory_image_accepts_additional_ca_without_weakening_tls() -> None:
    dockerfile = Path(__file__).parents[1] / "Dockerfile"
    content = dockerfile.read_text(encoding="utf-8")

    assert "--mount=type=secret,id=posttrain_ca_bundle,required=false" in content
    assert "cat /run/secrets/posttrain_ca_bundle >>" in content
    assert "REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt" in content
    assert "SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt" in content
    assert "trusted-host" not in content
    assert "allow-insecure" not in content
