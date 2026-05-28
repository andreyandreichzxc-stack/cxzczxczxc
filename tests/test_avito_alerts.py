from src.core.avito.alerts import check_price_alert


def test_price_drop_alert_without_threshold() -> None:
    alert = check_price_alert(
        {
            "title": "iPhone 15",
            "price": 90_000,
            "url": "https://example.test/listing",
        },
        previous_price=100_000,
    )

    assert alert is not None
    assert alert["alert_type"] == "price_drop"
    assert alert["price"] == 90_000
    assert alert["previous_price"] == 100_000


def test_price_drop_below_ten_percent_is_not_alerted() -> None:
    alert = check_price_alert(
        {
            "title": "iPhone 15",
            "price": 91_000,
            "url": "https://example.test/listing",
        },
        previous_price=100_000,
    )

    assert alert is None
