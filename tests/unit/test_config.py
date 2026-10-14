from common.config import direct_database_url, normalize_database_url


def test_direct_url_derived_from_neon_pooler_host():
    pooled = "postgresql://user:secret@ep-cool-name-123456-pooler.us-east-2.aws.neon.tech/db?sslmode=require&channel_binding=require"
    direct = direct_database_url(pooled)
    assert direct == "postgresql://user:secret@ep-cool-name-123456.us-east-2.aws.neon.tech/db?sslmode=require&channel_binding=require"


def test_non_pooled_urls_are_unchanged():
    for url in ("postgresql://quant_user:quant_password@localhost:5432/quant_db",
                "postgresql://u:p@ep-x-1.eu-central-1.aws.neon.tech/db?sslmode=require"):
        assert direct_database_url(url) == url


def test_postgres_scheme_normalised():
    assert normalize_database_url("postgres://u:p@h/db") == "postgresql://u:p@h/db"
