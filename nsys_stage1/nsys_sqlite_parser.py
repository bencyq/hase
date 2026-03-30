# -*- coding: utf-8 -*-
"""
封装 nsys profile / export sqlite / sqlite 解析。
"""
import argparse
import os
import sqlite3
import subprocess


DEFAULT_NSYS_BIN = os.environ.get("NSYS_BIN", "nsys")


def _run_command(cmd, cwd=None):
    subprocess.run(cmd, cwd=cwd, check=True)


def profile_with_nsys(
    python_bin,
    runner_path,
    model_path,
    output_prefix,
    range_name,
    warmup,
    loops,
    nsys_bin=DEFAULT_NSYS_BIN,
):
    cmd = [
        nsys_bin,
        "profile",
        "--trace",
        "cuda,nvtx",
        "--capture-range",
        "nvtx",
        "--nvtx-capture",
        range_name,
        "--output",
        output_prefix,
        python_bin,
        runner_path,
        "--model",
        model_path,
        "--warmup",
        str(warmup),
        "--loops",
        str(loops),
        "--range-name",
        range_name,
    ]
    report_path = output_prefix + ".nsys-rep"
    result = subprocess.run(cmd)
    if result.returncode != 0 and not (result.returncode == 143 and os.path.isfile(report_path)):
        raise subprocess.CalledProcessError(result.returncode, cmd)
    return report_path


def export_sqlite(report_path, output_prefix, nsys_bin=DEFAULT_NSYS_BIN):
    sqlite_path = output_prefix + ".sqlite"
    cmd = [
        nsys_bin,
        "export",
        "--type",
        "sqlite",
        "--output",
        sqlite_path,
        report_path,
    ]
    _run_command(cmd)
    if not os.path.isfile(sqlite_path):
        raise FileNotFoundError("nsys sqlite 导出失败: {}".format(sqlite_path))
    return sqlite_path


def _table_names(conn):
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
    return [row[0] for row in rows]


def _pick_table(table_names, candidates):
    for name in candidates:
        if name in table_names:
            return name
    raise RuntimeError("未找到目标表，当前表: {}".format(sorted(table_names)))


def _pick_column(columns, candidates):
    for name in candidates:
        if name in columns:
            return name
    raise RuntimeError("未找到目标字段，当前字段: {}".format(columns))


def _table_columns(conn, table_name):
    rows = conn.execute("PRAGMA table_info('{}')".format(table_name)).fetchall()
    return [row[1] for row in rows]


def _fetch_nvtx_range(conn, range_name):
    table_names = _table_names(conn)
    table_name = _pick_table(table_names, ["NVTX_EVENTS", "StringIds"])
    if table_name != "NVTX_EVENTS":
        raise RuntimeError("当前 nsys sqlite 不包含 NVTX_EVENTS 表，无法继续解析")

    columns = _table_columns(conn, table_name)
    start_col = _pick_column(columns, ["start", "startNs"])
    end_col = _pick_column(columns, ["end", "endNs"])
    event_type_col = "eventType" if "eventType" in columns else None
    text_col = "text" if "text" in columns else None
    text_id_col = "textId" if "textId" in columns else None

    if text_col is not None and text_id_col is not None:
        sql = """
            SELECT e.{start}, e.{end}
            FROM {table} e
            LEFT JOIN StringIds s ON e.{text_id} = s.id
            WHERE COALESCE(e.{text}, s.value) = ?
        """.format(
            start=start_col,
            end=end_col,
            table=table_name,
            text=text_col,
            text_id=text_id_col,
        )
    elif text_col is not None:
        sql = "SELECT {start}, {end} FROM {table} WHERE {text} = ?".format(
            start=start_col,
            end=end_col,
            table=table_name,
            text=text_col,
        )
    else:
        raise RuntimeError("NVTX_EVENTS 缺少可用文本字段: {}".format(columns))

    params = [range_name]
    if event_type_col is not None:
        sql += " AND e.{} = 59".format(event_type_col) if " e." in sql else " AND {} = 59".format(event_type_col)
    sql += " ORDER BY {} ASC LIMIT 1".format(start_col if " e." not in sql else "e." + start_col)

    row = conn.execute(sql, params).fetchone()
    if row is None:
        raise RuntimeError("未找到 NVTX range: {}".format(range_name))
    return int(row[0]), int(row[1])


def _sum_kernel_time_ns(conn, start_ns, end_ns):
    table_names = _table_names(conn)
    table_name = _pick_table(
        table_names,
        ["CUPTI_ACTIVITY_KIND_KERNEL", "CUDA_GPU_KERNEL_EVENTS"],
    )
    columns = _table_columns(conn, table_name)
    start_col = _pick_column(columns, ["start", "startNs"])
    end_col = _pick_column(columns, ["end", "endNs"])

    sql = """
        SELECT COALESCE(SUM({end_col} - {start_col}), 0)
        FROM {table_name}
        WHERE {start_col} >= ? AND {end_col} <= ?
    """.format(
        start_col=start_col,
        end_col=end_col,
        table_name=table_name,
    )
    row = conn.execute(sql, (start_ns, end_ns)).fetchone()
    return int(row[0] or 0)


def _read_gpu_name(conn):
    table_names = _table_names(conn)
    if "TARGET_INFO_GPU" not in table_names:
        return None
    columns = _table_columns(conn, "TARGET_INFO_GPU")
    name_col = _pick_column(columns, ["name", "deviceName"])
    row = conn.execute(
        "SELECT {} FROM TARGET_INFO_GPU ORDER BY rowid ASC LIMIT 1".format(name_col)
    ).fetchone()
    return row[0] if row else None


def _sum_all_kernel_time_ns(conn):
    table_names = _table_names(conn)
    table_name = _pick_table(
        table_names,
        ["CUPTI_ACTIVITY_KIND_KERNEL", "CUDA_GPU_KERNEL_EVENTS"],
    )
    columns = _table_columns(conn, table_name)
    start_col = _pick_column(columns, ["start", "startNs"])
    end_col = _pick_column(columns, ["end", "endNs"])
    row = conn.execute(
        "SELECT COALESCE(SUM({} - {}), 0) FROM {}".format(end_col, start_col, table_name)
    ).fetchone()
    return int(row[0] or 0)


def parse_pure_gpu_time(sqlite_path, range_name, loops):
    conn = sqlite3.connect(sqlite_path)
    try:
        try:
            start_ns, end_ns = _fetch_nvtx_range(conn, range_name)
            kernel_sum_ns = _sum_kernel_time_ns(conn, start_ns, end_ns)
            capture_range_name = range_name
        except RuntimeError as exc:
            if "NVTX_EVENTS" not in str(exc):
                raise
            kernel_sum_ns = _sum_all_kernel_time_ns(conn)
            capture_range_name = range_name + " (capture-range fallback)"
        gpu_name = _read_gpu_name(conn)
    finally:
        conn.close()

    pure_gpu_ms = kernel_sum_ns / max(loops, 1) / 1e6
    return {
        "gpu_kernel_time_sum_ns": kernel_sum_ns,
        "pure_gpu_ms": pure_gpu_ms,
        "capture_range_name": capture_range_name,
        "gpu_name": gpu_name,
    }


def main():
    parser = argparse.ArgumentParser(description="Parse nsys sqlite GPU time.")
    parser.add_argument("--sqlite", required=True, help="nsys 导出的 sqlite 路径")
    parser.add_argument("--range-name", default="measure", help="NVTX range 名称")
    parser.add_argument("--loops", type=int, default=50, help="正式循环次数")
    args = parser.parse_args()

    result = parse_pure_gpu_time(args.sqlite, args.range_name, args.loops)
    print(result)


if __name__ == "__main__":
    main()
