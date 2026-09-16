"""PostgreSQL utilities for healthcare analytics outputs."""

import os

import pandas as pd
from dotenv import load_dotenv
from sqlalchemy import create_engine
from sqlalchemy.engine import Engine


load_dotenv()


def get_engine() -> Engine:
    """Create a SQLAlchemy PostgreSQL engine from environment variables."""

    host = os.getenv("DB_HOST", "localhost")
    port = os.getenv("DB_PORT", "5432")
    database = os.getenv(
        "DB_NAME",
        "healthcare_intelligence",
    )
    user = os.getenv("DB_USER", "postgres")
    password = os.getenv("DB_PASSWORD")

    if not password:
        raise ValueError(
            "DB_PASSWORD environment variable is not set."
        )

    return create_engine(
        f"postgresql+psycopg2://{user}:{password}"
        f"@{host}:{port}/{database}"
    )


def load_dataframe_to_table(
    df: pd.DataFrame,
    table_name: str,
    engine: Engine,
) -> None:
    """Replace a PostgreSQL table with a pandas DataFrame."""

    if df.empty:
        raise ValueError("Cannot load an empty DataFrame.")

    df.to_sql(
        table_name,
        engine,
        schema="public",
        if_exists="replace",
        index=False,
        chunksize=5000,
        method="multi",
    )