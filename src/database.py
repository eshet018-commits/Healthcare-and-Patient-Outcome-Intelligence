"""PostgreSQL utilities for healthcare analytics outputs."""

import os

import pandas as pd
import psycopg2


def get_connection():
    """Create a PostgreSQL connection from environment variables."""

    return psycopg2.connect(
        host=os.getenv("DB_HOST", "localhost"),
        port=os.getenv("DB_PORT", "5432"),
        database=os.getenv("DB_NAME", "healthcare"),
        user=os.getenv("DB_USER", "postgres"),
        password=os.getenv("DB_PASSWORD"),
    )


def load_dataframe_to_table(
    df: pd.DataFrame,
    table_name: str,
    connection,
) -> None:
    """Replace a PostgreSQL table with a pandas DataFrame."""

    if df.empty:
        raise ValueError("Cannot load an empty DataFrame.")

    columns = list(df.columns)

    column_definitions = ", ".join(
        f'"{column}" TEXT'
        for column in columns
    )

    column_names = ", ".join(
        f'"{column}"'
        for column in columns
    )

    placeholders = ", ".join(
        ["%s"] * len(columns)
    )

    create_sql = f"""
        DROP TABLE IF EXISTS "{table_name}";

        CREATE TABLE "{table_name}" (
            {column_definitions}
        );
    """

    insert_sql = f"""
        INSERT INTO "{table_name}" ({column_names})
        VALUES ({placeholders});
    """

    with connection.cursor() as cursor:
        cursor.execute(create_sql)

        rows = [
            tuple(
                None if pd.isna(value) else str(value)
                for value in row
            )
            for row in df.itertuples(index=False, name=None)
        ]

        cursor.executemany(insert_sql, rows)

    connection.commit()