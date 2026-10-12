# -*- coding: utf-8 -*-
"""多线程分段下载 wheel 文件。

背景：本机网络环境下 download.pytorch.org 对单连接限速约 160 KB/s，
但支持 Range 分段请求，多连接并行可显著提速（实测 6 并发达 2 MB/s）。

用法：
    python fetch_wheel.py <url> <输出文件名> [线程数]
"""
import os
import sys
import threading
import time
import urllib.request

# 统一的浏览器 UA，避免被中间件按 UA 拦截
UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36"
)


def head_size(url: str) -> int:
    """用 HEAD 请求取文件总大小。"""
    req = urllib.request.Request(url, headers={"User-Agent": UA}, method="HEAD")
    with urllib.request.urlopen(req, timeout=30) as r:
        return int(r.headers["Content-Length"])


def download_part(url: str, out: str, start: int, end: int, retries: int = 5) -> None:
    """下载 [start, end] 区间，带重试；大小不符则重试。"""
    expect = end - start + 1
    for attempt in range(retries):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": UA, "Range": f"bytes={start}-{end}"},
            )
            written = 0
            # 流式写入，避免一次性把整段读进内存
            with urllib.request.urlopen(req, timeout=180) as r, open(out, "wb") as f:
                while True:
                    buf = r.read(1 << 20)
                    if not buf:
                        break
                    f.write(buf)
                    written += len(buf)
            if written == expect:
                return
            print(f"    段 {out} 不完整 {written}/{expect}，重试")
        except Exception as e:  # noqa: BLE001
            print(f"    段 {out} 异常 {e}，重试 {attempt + 1}/{retries}")
            time.sleep(2)
    raise RuntimeError(f"分段下载失败：{out}")


def main() -> None:
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(2)

    url, out = sys.argv[1], sys.argv[2]
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 12

    total = head_size(url)
    print(f"[信息] 总大小 {total / 1048576:.1f} MB，分 {n} 段并行下载")

    chunk = total // n
    plan = []
    for i in range(n):
        s = i * chunk
        e = total - 1 if i == n - 1 else (i + 1) * chunk - 1
        plan.append((i, s, e, f"{out}.part{i}"))

    errors = []
    start_time = time.time()

    def run(item):
        idx, s, e, part_path = item
        try:
            download_part(url, part_path, s, e)
            print(f"  段 {idx + 1}/{n} 完成")
        except Exception as ex:  # noqa: BLE001
            errors.append(str(ex))

    threads = [threading.Thread(target=run, args=(it,)) for it in plan]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    if errors:
        print("[失败] " + "; ".join(errors))
        sys.exit(1)

    # 按顺序合并分段
    with open(out, "wb") as fo:
        for _, _, _, part_path in plan:
            with open(part_path, "rb") as fi:
                while True:
                    buf = fi.read(1 << 22)
                    if not buf:
                        break
                    fo.write(buf)
            os.remove(part_path)

    size = os.path.getsize(out)
    elapsed = time.time() - start_time
    ok = "OK" if size == total else "大小不符！"
    print(
        f"[完成] {out}  {size / 1048576:.1f} MB  {ok}  "
        f"用时 {elapsed:.0f}s  平均 {size / 1048576 / max(elapsed, 1):.2f} MB/s"
    )
    if size != total:
        sys.exit(1)


if __name__ == "__main__":
    main()
