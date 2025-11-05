"""Generate synthetic PPG time-series and optionally POST windows to a backend endpoint.

Usage:
  python generate_synthetic_ppg.py --duration 30 --rate 30 --hr 70 --out synthetic.csv --upload http://localhost:5000/api/ppg/upload-frame

Requirements: requests (pip install requests)
"""
import argparse
import csv
import math
import random
import time
import json

try:
    import requests
except Exception:
    requests = None


def synth_ppg(duration_s=30, rate=30, hr=70, noise=0.02):
    n = int(duration_s * rate)
    samples = []
    for i in range(n):
        t = i / rate
        # simple synthetic waveform: base sin envelope + sharp peaks
        pulse = abs(math.sin(2 * math.pi * (hr / 60.0) * t))
        # emphasize peaks
        value = 0.7 * pulse + 0.3 * (pulse ** 8)
        value += (random.random() - 0.5) * noise
        value = max(0.0, min(1.0, value))
        samples.append(value)
    return samples


def save_csv(path, samples, rate):
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['time', 'value'])
        for i, v in enumerate(samples):
            w.writerow([i / rate, v])


def upload_windows(url, samples, rate, window_size=100, step=50):
    if requests is None:
        print('requests not installed; cannot upload')
        return
    for start in range(0, len(samples) - window_size + 1, step):
        window = samples[start:start+window_size]
        payload = {'synthetic': True, 'hr': None, 'samples': window}
        try:
            r = requests.post(url, json=payload, timeout=5)
            print('uploaded window', start, 'status', r.status_code)
        except Exception as e:
            print('upload failed:', e)
            return


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--duration', type=float, default=30.0)
    p.add_argument('--rate', type=float, default=30.0)
    p.add_argument('--hr', type=float, default=70.0)
    p.add_argument('--out', type=str, default='synthetic_ppg.csv')
    p.add_argument('--upload', type=str, default=None)
    p.add_argument('--window', type=int, default=100)
    p.add_argument('--step', type=int, default=50)
    args = p.parse_args()

    samples = synth_ppg(args.duration, args.rate, args.hr)
    print(f'Generated {len(samples)} samples ({args.duration}s @ {args.rate}Hz)')
    save_csv(args.out, samples, args.rate)
    print('Saved to', args.out)
    if args.upload:
        upload_windows(args.upload, samples, args.rate, window_size=args.window, step=args.step)
    print('Done')
