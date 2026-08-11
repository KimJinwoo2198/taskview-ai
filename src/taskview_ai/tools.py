import json

from agents import function_tool

CATALOG = {
    "product": {
        "owner": "product-data@taskview.local",
        "fields": ["event_date", "feature", "account_id", "user_id", "usage_count"],
    },
    "operations": {
        "owner": "ops-data@taskview.local",
        "fields": ["ticket_id", "created_at", "status", "assignee", "region", "resolution_hours"],
    },
    "voc": {
        "owner": "cx-data@taskview.local",
        "fields": ["ticket_id", "created_at", "customer_name", "address", "age", "message", "issue_type"],
    },
}


@function_tool
def search_data_catalog(query: str) -> str:
    """Search TaskView's approved data catalog. Input is a short natural-language query."""
    terms = query.lower().split()
    matches = {
        name: value
        for name, value in CATALOG.items()
        if name in query.lower()
        or any(term in " ".join(value["fields"]).lower() for term in terms)
    }
    return json.dumps(matches or CATALOG, ensure_ascii=False)


@function_tool
def get_privacy_transform(field_name: str) -> str:
    """Return the mandatory privacy-safe transform for a potentially sensitive field."""
    transforms = {
        "user_id": "drop",
        "account_id": "mask",
        "customer_name": "drop",
        "ticket_id": "drop",
        "address": "region_group",
        "age": "age_band",
        "message": "classify",
    }
    return transforms.get(field_name.lower(), "select")

