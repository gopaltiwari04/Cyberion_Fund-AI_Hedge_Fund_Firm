import great_expectations as gx
import polars as pl


def validate_market_data(df: pl.DataFrame, ticker: str):
    """
    Validate cleaned market data using Great Expectations 1.x.
    """

    print(f"Running Great Expectations validation for {ticker}...")

    # Convert Polars -> Pandas for Great Expectations
    pandas_df = df.to_pandas()

    # Create GX context
    context = gx.get_context()

    # Create Pandas data source
    data_source = context.data_sources.add_pandas(
        name=f"{ticker}_source"
    )

    # Create dataframe asset
    data_asset = data_source.add_dataframe_asset(
        name=f"{ticker}_asset"
    )

    # Create batch definition
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        f"{ticker}_batch"
    )

    # Create batch
    batch = batch_definition.get_batch(
        batch_parameters={
            "dataframe": pandas_df
        }
    )

    # ----------------------------------------------------
    # Expectations
    # ----------------------------------------------------

    expectations = [
        (
            "date column exists",
            gx.expectations.ExpectColumnToExist(
                column="date"
            )
        ),

        (
            "open column exists",
            gx.expectations.ExpectColumnToExist(
                column="open"
            )
        ),

        (
            "high column exists",
            gx.expectations.ExpectColumnToExist(
                column="high"
            )
        ),

        (
            "low column exists",
            gx.expectations.ExpectColumnToExist(
                column="low"
            )
        ),

        (
            "close column exists",
            gx.expectations.ExpectColumnToExist(
                column="close"
            )
        ),

        (
            "volume column exists",
            gx.expectations.ExpectColumnToExist(
                column="volume"
            )
        ),

        (
            "date contains no nulls",
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column="date"
            )
        ),

        (
            "close contains no nulls",
            gx.expectations.ExpectColumnValuesToNotBeNull(
                column="close"
            )
        ),

        (
            "close is greater than zero",
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="close",
                min_value=0,
                strict_min=True
            )
        ),

        (
            "volume is non-negative",
            gx.expectations.ExpectColumnValuesToBeBetween(
                column="volume",
                min_value=0,
                strict_min=False
            )
        ),
    ]

    # ----------------------------------------------------
    # Run validation
    # ----------------------------------------------------

    failed = False

    for description, expectation in expectations:

        result = batch.validate(expectation)

        if result.success:
            print(f"  PASS: {description}")
        else:
            print(f"  FAIL: {description}")
            failed = True

    # ----------------------------------------------------
    # Final result
    # ----------------------------------------------------

    if failed:
        raise ValueError(
            f"Great Expectations validation FAILED for {ticker}"
        )

    print(
        f"Great Expectations validation PASSED for {ticker}"
    )