from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


from utils.schema import (
    ConnectorOutput,
    CustomerMetrics,
    DomainAgentOutput,
    PriorityAction,
    ProductMetrics,
    SellerMetrics,
)


def test_typed_dict_examples_can_be_instantiated() -> None:
    customer_metrics: CustomerMetrics = {
        "repeat_rate": 0.0,
        "avg_review_score": 4.1,
        "avg_delivery_days": 12.5,
        "pct_late_delivery": 0.18,
        "top_payment_type": "credit_card",
    }
    product_metrics: ProductMetrics = {
        "top_category": "health_beauty",
        "top_category_revenue": 1260000.0,
        "total_categories": 73,
        "avg_order_value": 161.0,
    }
    seller_metrics: SellerMetrics = {
        "total_sellers": 3095,
        "pct_sao_paulo": 0.60,
        "top_seller_revenue_concentration": 0.62,
    }
    domain_output: DomainAgentOutput = {
        "agent": "customer",
        "timestamp": "2026-04-15T00:00:00+00:00",
        "insights": ["Repeat rate is zero."],
        "metrics": customer_metrics,
        "top_opportunity": "Improve delivery performance.",
        "risk_flags": ["late_delivery_high"],
    }
    priority_action: PriorityAction = {
        "action": "Reduce delivery SLA to <7 days",
        "agent": "cross-domain",
        "urgency": "HIGH",
    }
    connector_output: ConnectorOutput = {
        "timestamp": "2026-04-15T00:00:00+00:00",
        "cross_domain_insights": ["Delivery delays depress review scores and retention."],
        "strategic_recommendation": "Fix logistics first to improve repeat purchases.",
        "priority_actions": [priority_action],
        "briefing": "Customer, product, and seller metrics point to logistics as the top lever.",
    }

    assert customer_metrics["top_payment_type"] == "credit_card"
    assert product_metrics["total_categories"] == 73
    assert seller_metrics["pct_sao_paulo"] == 0.60
    assert domain_output["agent"] == "customer"
    assert connector_output["priority_actions"][0]["urgency"] == "HIGH"


if __name__ == "__main__":
    test_typed_dict_examples_can_be_instantiated()
    print("test_schema.py passed")
