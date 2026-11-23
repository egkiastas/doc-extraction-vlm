import pytest

from docextract.schema import (
    EXTRACTION_SCHEMA,
    Receipt,
    cord_to_raw,
    from_cord,
    parse_amount,
    parse_date,
    parse_output,
    to_cord,
)


@pytest.mark.parametrize("raw, expected", [
    ("60.000", 60000),          # Indonesian thousands separator
    ("1,213,000", 1213000),
    ("12,50", 12.5),            # decimal comma
    ("193.00", 193.0),
    ("Rp 1.250.000,00", 1250000),
    ("-5.455", -5455),
    ("(3.50)", -3.5),
    ("@28,000", 28000),
    ("1X", 1),
    ("2.00", 2),
    ("RM 0.5", 0.5),
    (46000, 46000),
    ("", None),
    ("-", None),
    (None, None),
])
def test_parse_amount(raw, expected):
    assert parse_amount(raw) == expected


@pytest.mark.parametrize("raw, expected", [
    ("15/01/2019", "2019-01-15"),
    ("2018-02-09", "2018-02-09"),
    ("09 FEB 2018", "2018-02-09"),
    ("Mar 5, 2018", "2018-03-05"),
    ("02-01-19 14:32", "2019-01-02"),
    ("4 - 14 -", "4 - 14 -"),     # unparseable: kept as is
    (None, None),
])
def test_parse_date(raw, expected):
    assert parse_date(raw) == expected


CORD_GT = {
    "menu": [
        {"nm": "JASMINE MT (L)", "cnt": "1", "price": "24,000",
         "sub": [{"nm": "COCONUT JELLY", "price": "4,000"}, {"nm": "Less Sugar", "price": "0"}]},
        {"nm": ["-", "Gyro Platter Regular"], "cnt": "2", "unitprice": "@10.000",
         "price": "20.000"},
    ],
    "sub_total": {"subtotal_price": ["48,000", "48,000"], "tax_price": "4,800",
                  "discount_price": "-4,800"},
    "total": {"total_price": "48,000", "cashprice": "50,000", "changeprice": "2,000"},
}


def test_cord_mapping():
    raw = cord_to_raw(CORD_GT)
    names = [it["name"] for it in raw["line_items"]]
    # priced sub-item becomes an item, zero-priced modifier is dropped, list names joined
    assert names == ["JASMINE MT (L)", "COCONUT JELLY", "Gyro Platter Regular"]
    assert raw["subtotal"] == "48,000"  # repeated label collapsed to its first value

    r = from_cord(CORD_GT)
    assert r.line_items[2].unit_price == 10000 and r.line_items[2].price == 20000
    assert r.subtotal == 48000 and r.tax == 4800 and r.total == 48000
    assert r.discount == 4800  # sign dropped
    assert r.merchant is None  # payment lines and missing labels don't leak in


def test_cord_single_item_menu_is_a_dict():
    r = from_cord({"menu": {"nm": "TICKET CP", "cnt": "2", "price": "60.000"},
                   "total": {"total_price": "60.000"}})
    assert len(r.line_items) == 1 and r.line_items[0].price == 60000


def test_cord_mapping_survives_malformed_predicted_trees():
    # Donut can emit any nesting; the mapping must not crash on it.
    raw = cord_to_raw({"menu": [{"nm": ["A", {"x": "1"}], "price": {"p": "2"}}, "junk"],
                       "total": ["7.000"], "sub_total": "x"})
    assert raw["line_items"] == [{"name": "A", "quantity": None, "unit_price": None,
                                  "price": None}]
    assert raw["total"] is None and raw["subtotal"] is None


def test_to_cord_round_trip_keeps_values():
    tree = to_cord(from_cord(CORD_GT))
    assert tree["total"] == {"total_price": "48000"}
    assert tree["menu"][2] == {"nm": "Gyro Platter Regular", "cnt": "2", "unitprice": "10000",
                               "price": "20000"}


def test_parse_output_accepts_fenced_json_and_rejects_garbage():
    r, err = parse_output('```json\n{"total": "12,50", "line_items": null}\n```')
    assert err is None and r.total == 12.5 and r.line_items == []
    assert parse_output("I could not read this receipt.")[0] is None
    assert parse_output('{"total": "1.000",}')[1].startswith("invalid json")
    assert parse_output('{"line_items": "none"}')[1].startswith("schema")


def test_model_schema_covers_receipt_fields():
    # The JSON schema given to Ollama is written by hand; it must not drift from the model.
    assert set(EXTRACTION_SCHEMA["properties"]) == set(Receipt.model_fields)
    assert set(EXTRACTION_SCHEMA["required"]) == set(Receipt.model_fields)
