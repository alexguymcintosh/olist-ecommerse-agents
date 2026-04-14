import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


def load_all() -> dict:
    """Load all 8 Olist CSVs into a dict of DataFrames."""
    files = {
        "orders": "olist_orders_dataset.csv",
        "customers": "olist_customers_dataset.csv",
        "order_items": "olist_order_items_dataset.csv",
        "products": "olist_products_dataset.csv",
        "sellers": "olist_sellers_dataset.csv",
        "payments": "olist_order_payments_dataset.csv",
        "reviews": "olist_order_reviews_dataset.csv",
        "categories": "product_category_name_translation.csv"
    }
    return {name: pd.read_csv(DATA_DIR / fname) for name, fname in files.items()}


def sample(df: pd.DataFrame, n: int = 100, seed: int = 42) -> pd.DataFrame:
    """Sample rows for LLM context — never send full dataset to LLM."""
    return df.sample(min(n, len(df)), random_state=seed)


def to_llm_string(df: pd.DataFrame, max_rows: int = 50) -> str:
    """Convert DataFrame to compact string for LLM context."""
    return df.head(max_rows).to_csv(index=False)
