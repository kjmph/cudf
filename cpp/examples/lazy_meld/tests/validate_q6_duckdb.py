# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Optional independent SQL oracle; requires duckdb and its tpch extension, outside CTest."""

import argparse
import json
import math
import subprocess
import tempfile
from pathlib import Path

import duckdb


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("binary", type=Path, help="built q6_demo executable")
    parser.add_argument("--scale", type=float, default=0.01)
    parser.add_argument(
        "--work-dir",
        type=Path,
        default=Path(tempfile.gettempdir()) / "jit-q6-validation",
    )
    args = parser.parse_args()
    if not 0 < args.scale <= 1:
        parser.error("scale must be in (0, 1] for this POC validation")
    binary = args.binary.resolve(strict=True)
    args.work_dir.mkdir(parents=True, exist_ok=True)
    datafile = (args.work_dir / "lineitem.tbl").resolve()
    con = duckdb.connect()
    con.execute("INSTALL tpch")
    con.execute("LOAD tpch")
    con.execute(f"CALL dbgen(sf={args.scale})")
    rows = con.execute("SELECT count(*) FROM lineitem").fetchone()[0]
    sql = """SELECT sum(l_extendedprice*l_discount) FROM lineitem
WHERE l_shipdate >= DATE '1994-01-01' AND l_shipdate < DATE '1995-01-01'
AND l_discount BETWEEN 0.05 AND 0.07 AND l_quantity < 24"""
    expected = con.execute(sql).fetchone()[0]
    # SQL string quoting, not shell interpolation. DBGen's pipe-free fields match .tbl input.
    path_sql = str(datafile).replace("'", "''")
    con.execute(f"COPY lineitem TO '{path_sql}' (DELIMITER '|', HEADER false)")
    execution = subprocess.run(
        [str(binary), "--lineitem", str(datafile)],
        text=True,
        capture_output=True,
        check=True,
    )
    fields = dict(
        line.split("=", 1)
        for line in execution.stdout.splitlines()
        if "=" in line
    )
    if fields.get("correctness") != "PASS" or int(fields["rows"]) != rows:
        raise RuntimeError(
            "q6_demo failed its reference or input-row validation"
        )
    actual = None if fields["result"] == "null" else float(fields["result"])
    if (expected is None) != (actual is None) or (
        expected is not None
        and not math.isclose(
            actual, float(expected), rel_tol=1e-10, abs_tol=1e-8
        )
    ):
        raise RuntimeError(
            f"Meld revenue {actual} disagrees with DuckDB {expected}"
        )
    report = {
        "duckdb_version": duckdb.__version__,
        "generator": "DuckDB tpch/dbgen",
        "scale_factor": args.scale,
        "rows": rows,
        "duckdb_decimal_revenue": str(expected),
        "meld_float64_revenue": fields["result"],
        "comparison": "rel_tol=1e-10, abs_tol=1e-8",
        "correctness": "PASS",
        "query": sql,
        "command": [str(binary), "--lineitem", str(datafile)],
    }
    (args.work_dir / "reference.json").write_text(
        json.dumps(report, indent=2) + "\n"
    )
    print(execution.stdout, end="")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
