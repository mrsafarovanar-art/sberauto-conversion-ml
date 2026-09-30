"""Измерение полной задержки HTTP на работающем локальном сервере."""
import argparse
import concurrent.futures
import json
import time
from pathlib import Path
import httpx
import numpy as np
ROOT = Path(__file__).resolve().parent


def benchmark(url='http://127.0.0.1:8035', count=100):
    sample = json.loads((ROOT/'examples/session.json').read_text())
    with httpx.Client(base_url=url, timeout=10, trust_env=False) as client:
        assert client.get('/health').json()['status'] == 'ok'
        for _ in range(5):
            client.post('/predict', json=sample).raise_for_status()
        def request(_):
            start = time.perf_counter()
            result = client.post('/predict', json=sample)
            result.raise_for_status()
            assert type(result.json()) is int and result.json() in (0, 1)
            return time.perf_counter() - start
        sequential = [request(i) for i in range(count)]
        start = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            parallel_latencies = list(pool.map(request, range(count)))
        duration = time.perf_counter() - start
    def summarize(values):
        return {'requests': len(values), 'mean_ms': float(np.mean(values)*1000),
                'p50_ms': float(np.quantile(values, .5)*1000), 'p95_ms': float(np.quantile(values,.95)*1000),
                'max_ms': float(max(values)*1000)}
    report = {'url': url, 'warmup_requests': 5, 'sequential': summarize(sequential),
              'concurrent_8_clients': summarize(parallel_latencies), 'concurrent_requests_per_second': count/duration,
              'scope': 'local HTTP, loaded model, one server worker; excludes cold start and external network'}
    report['mean_under_3_seconds'] = report['sequential']['mean_ms'] < 3000
    assert report['mean_under_3_seconds']
    (ROOT/'reports/api_benchmark.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return report

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--url', default='http://127.0.0.1:8035')
    benchmark(parser.parse_args().url)
