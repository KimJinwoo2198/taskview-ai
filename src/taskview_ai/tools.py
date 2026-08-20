import json

from agents import function_tool

CATALOG = {
    "product": {
        "owner": "Federal Communications Commission",
        "fields": [
            "ticket_created",
            "state",
            "issue_type",
            "issue",
            "method",
            "caller_id_number",
            "event_time",
            "os_family",
            "os_version",
            "dropoff_step",
            "error_log",
        ],
    },
    "operations": {
        "owner": "NYC Open Data",
        "fields": [
            "created_date",
            "borough",
            "agency",
            "complaint_type",
            "status",
            "resolution_hours",
            "incident_address",
            "latitude",
            "longitude",
            "exact_address",
            "birth_date",
        ],
    },
    "voc": {
        "owner": "National Highway Traffic Safety Administration",
        "fields": [
            "date_complaint_filed",
            "manufacturer",
            "make",
            "model",
            "model_year",
            "component",
            "crash",
            "fire",
            "injuries",
            "deaths",
            "vin",
            "summary",
            "created_at",
            "customer_name",
            "phone",
            "email",
            "address",
            "age",
            "message",
            "ticket_text",
            "ticket_id",
        ],
    },
}


@function_tool
def search_data_catalog(query: str) -> str:
    """Search Needex's approved data catalog. Input is a short natural-language query."""
    terms = query.lower().split()
    matches = {
        name: value
        for name, value in CATALOG.items()
        if name in query.lower() or any(term in " ".join(value["fields"]).lower() for term in terms)
    }
    return json.dumps(matches or CATALOG, ensure_ascii=False)


@function_tool
def get_privacy_transform(field_name: str) -> str:
    """Return the mandatory privacy-safe transform for a potentially sensitive field."""
    transforms = {
        "user_id": "drop",
        "account_id": "mask",
        "customer_name": "drop",
        "name": "drop",
        "phone": "drop",
        "email": "drop",
        "ticket_id": "drop",
        "address": "region_group",
        "exact_address": "region_group",
        "age": "age_band",
        "birth_date": "age_band",
        "message": "classify",
        "ticket_text": "classify",
        "error_log": "classify",
        "caller_id_number": "drop",
        "incident_address": "drop",
        "latitude": "drop",
        "longitude": "drop",
        "vin": "drop",
        "summary": "drop",
    }
    return transforms.get(field_name.lower(), "select")
