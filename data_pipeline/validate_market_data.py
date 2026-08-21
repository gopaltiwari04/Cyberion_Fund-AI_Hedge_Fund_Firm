import great_expectations as gx
import polars as pl


def validate_market_data(df: pl.DataFrame, ticker: str):
    """
    Validate cleaned market data using Great Expectations.
    """

    print(f"Running Great Expectations validation for {ticker}...")

    # Convert Polars -> Pandas because GX works easily with Pandas DataFrames
    pandas_df = df.to_pandas()

    # Create an ephemeral GX context
    context = gx.get_context()

    # Create a pandas data source
    data_source = context.data_sources.add_pandas(
        name=f"{ticker}_source"
    )

    # Create a data asset
    data_asset = data_source.add_dataframe_asset(
        name=f"{ticker}_asset"
    )

    # Create a batch definition
    batch_definition = data_asset.add_batch_definition_whole_dataframe(
        f"{ticker}_batch"
    )

    batch = batch_definition.get_batch(
        batch_parameters={
            "dataframe": pandas_df
        }
    )

    # ----------------------------------------------------
    # Expectations
    # ----------------------------------------------------

    expectations = [
        gx.expectations.ExpectColumnToExist(
            column="date"
        ),

        gx.expectations.ExpectColumnToExist(
            column="open"
        ),

        gx.expectations.ExpectColumnToExist(
            column="high"
        ),

        gx.expectations.ExpectColumnToExist(
            column="low"
        ),

        gx.expectations.ExpectColumnToExist(
            column="close"
        ),

        gx.expectations.ExpectColumnToExist(
            column="volume"
        ),

        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="date"
        ),

        gx.expectations.ExpectColumnValuesToNotBeNull(
            column="close"
        ),

        gx.expectations.ExpectColumnValuesToBeGreaterThan(
            column="close",
            value=0
        ),

        gx.expectations.ExpectColumnValuesToBeGreaterThanOrEqualTo(
            column="volume",
            value=0
        ),
    ]

    # ----------------------------------------------------
    # Run expectations
    # ----------------------------------------------------

    failed = False

    for expectation in expectations:

        result = batch.validate(
            expectation
        )

        if result.success:
            print(f"  PASS: {expectation.type}")

        else:
            print(f"  FAIL: {expectation.type}")
            failed = True

    if failed:
        raise ValueError(
            f"Great Expectations validation FAILED for {ticker}"
        )

    print(
        f"Great Expectations validation PASSED for {ticker}"
    )