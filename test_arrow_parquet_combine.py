#!/usr/bin/env python3
"""
Test script to verify how PyArrow handles combining Parquet files with schema mismatches.

This reproduces the scenario from DuckDB issue #17126 where:
- test1.parquet has memberDn with all NULL values (null type)
- test2.parquet has memberDn with string values (string type)

Related: https://github.com/duckdb/duckdb/issues/17126
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pyarrow.dataset as ds
import tempfile
import os


def create_test_files(tmpdir):
    """Create test parquet files with schema mismatch."""
    # File 1: memberDn has all nulls (null type)
    table1 = pa.table({
        "groupDn": pa.array(["TEST 1", "TEST 2", "TEST 3"], type=pa.string()),
        "memberDn": pa.array([None, None, None], type=pa.null())
    })

    # File 2: memberDn has string values
    table2 = pa.table({
        "groupDn": pa.array(["TEST 1", "TEST 2", "TEST 3"], type=pa.string()),
        "memberDn": pa.array(["a", "b", "c"], type=pa.string())
    })

    file1 = os.path.join(tmpdir, "test1.parquet")
    file2 = os.path.join(tmpdir, "test2.parquet")

    pq.write_table(table1, file1)
    pq.write_table(table2, file2)

    return file1, file2


def test_dataset_api_null_first(tmpdir):
    """Test: Dataset API with null-type file first - EXPECTED TO FAIL."""
    file1, file2 = create_test_files(tmpdir)

    try:
        dataset = ds.dataset([file1, file2], format='parquet')
        table = dataset.to_table()
        print("UNEXPECTED SUCCESS")
        return True
    except Exception as e:
        print(f"EXPECTED FAILURE: {type(e).__name__}: {e}")
        return False


def test_dataset_api_string_first(tmpdir):
    """Test: Dataset API with string-type file first - EXPECTED TO SUCCEED."""
    file1, file2 = create_test_files(tmpdir)

    try:
        # Read string-typed file first
        dataset = ds.dataset([file2, file1], format='parquet')
        table = dataset.to_table()
        print(f"SUCCESS - Schema: {dataset.schema}")
        print(f"Rows: {len(table)}")
        return True
    except Exception as e:
        print(f"FAILURE: {type(e).__name__}: {e}")
        return False


def test_concat_tables_promote(tmpdir):
    """Test: concat_tables with promote_options - EXPECTED TO SUCCEED."""
    file1, file2 = create_test_files(tmpdir)

    try:
        t1 = pq.read_table(file1)
        t2 = pq.read_table(file2)
        combined = pa.concat_tables([t1, t2], promote_options='default')
        print(f"SUCCESS - Schema: {combined.schema}")
        print(f"Rows: {len(combined)}")
        return True
    except Exception as e:
        print(f"FAILURE: {type(e).__name__}: {e}")
        return False


def test_manual_cast(tmpdir):
    """Test: Manual cast null->string before concat - EXPECTED TO SUCCEED."""
    file1, file2 = create_test_files(tmpdir)

    try:
        t1 = pq.read_table(file1)
        t2 = pq.read_table(file2)

        # Cast null type to string
        t1_casted = t1.cast(pa.schema([
            ('groupDn', pa.string()),
            ('memberDn', pa.string())
        ]))

        combined = pa.concat_tables([t1_casted, t2])
        print(f"SUCCESS - Schema: {combined.schema}")
        print(f"Rows: {len(combined)}")
        return True
    except Exception as e:
        print(f"FAILURE: {type(e).__name__}: {e}")
        return False


def main():
    print(f"PyArrow version: {pa.__version__}")
    print("=" * 70)

    with tempfile.TemporaryDirectory() as tmpdir:
        tests = [
            ("Dataset API (null-type first)", test_dataset_api_null_first),
            ("Dataset API (string-type first)", test_dataset_api_string_first),
            ("concat_tables with promote_options", test_concat_tables_promote),
            ("Manual cast null->string", test_manual_cast),
        ]

        results = []
        for name, test_func in tests:
            print(f"\nTest: {name}")
            print("-" * 50)
            results.append((name, test_func(tmpdir)))

        print("\n" + "=" * 70)
        print("SUMMARY")
        print("=" * 70)
        for name, passed in results:
            status = "✓ PASS" if passed else "✗ FAIL"
            print(f"{status}: {name}")

        print("\n" + "=" * 70)
        print("CONCLUSION")
        print("=" * 70)
        print("""
Arrow CAN combine parquet files with null vs string type mismatch using:
1. concat_tables() with promote_options='default' or 'permissive'
2. Reading files in reverse order (string-typed file first)
3. Manual casting before concatenation

The Dataset API has similar limitations to DuckDB's read_parquet when
the null-typed file is read first (schema is taken from first file).
""")


if __name__ == "__main__":
    main()
