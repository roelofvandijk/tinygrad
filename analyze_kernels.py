#!/usr/bin/env python3
"""
Kernel performance analyzer for tinygrad LLM inference.
Runs a model, captures kernel details, and identifies optimization opportunities.
"""

import subprocess
import re
import sys
from collections import defaultdict
from pathlib import Path

def run_benchmark(model="youtu-llm:2b-Q4_0", tokens=6):
    """Run benchmark and capture DEBUG=5 output."""
    print(f"Running benchmark: {model} with {tokens} tokens...")
    cmd = [
        "DEBUG=5",
        ".venv2/bin/python",
        "tinygrad/apps/llm.py",
        "--model", model,
        "--benchmark", str(tokens)
    ]

    result = subprocess.run(
        " ".join(cmd),
        shell=True,
        capture_output=True,
        text=True,
        cwd="/Users/rvd/src/rvd/tinygrad"
    )

    return result.stdout + result.stderr

def strip_ansi(s):
    return re.sub(r'\x1b\[[0-9;]*m', '', s)

def parse_kernel_stats(output):
    """Extract kernel performance statistics."""
    output = strip_ansi(output)
    kernels = []

    # Pattern: *** METAL    199 E_128_16n21   arg  2 mem   1.31 GB tm     12.50us/   115.39ms (... GB/s) ['op1', 'op2']
    for line in output.split('\n'):
        if '*** METAL' not in line: continue
        m = re.match(r'\*\*\* METAL\s+(\d+)\s+(\S+)\s+arg\s+\d+\s+mem\s+([\d.]+)\s+GB\s+tm\s+([\d.]+)([um]s)/\s+([\d.]+)ms\s+\(\s*(\d+)\s+GFLOPS\s+([\d|]+)\s+GB/s\)\s*(?:\[(.*?)\])?', line)
        if not m: continue
        kernel_id = int(m.group(1))
        name = m.group(2)
        mem_gb = float(m.group(3))
        time_val = float(m.group(4))
        time_unit = m.group(5)
        time_us = time_val if time_unit == 'us' else time_val * 1000  # ms -> us
        total_ms = float(m.group(6))
        bw_str = m.group(8)
        bandwidth_gbs = float(bw_str.split('|')[0])
        ops = [x.strip().strip("'") for x in m.group(9).split(",")] if m.group(9) else []
        call_count = int(total_ms * 1000 / time_us) if time_us > 0 else 0

        kernels.append({
            'id': kernel_id, 'name': name, 'mem_gb': mem_gb,
            'time_us': time_us, 'total_ms': total_ms,
            'bandwidth_gbs': bandwidth_gbs, 'call_count': call_count, 'ops': ops
        })

    return kernels

def extract_kernel_code(output, kernel_name):
    """Extract Metal source code for a specific kernel."""
    output = strip_ansi(output)
    pattern = rf'kernel void {re.escape(kernel_name)}\(.*?\n\}}'
    match = re.search(pattern, output, re.DOTALL)
    return match.group(0) if match else None

def extract_kernel_opts(output, kernel_id):
    """Extract applied optimizations for a kernel."""
    output = strip_ansi(output)
    lines = output.split('\n')
    for i, line in enumerate(lines):
        if f'*** METAL' in line and re.match(rf'\*\*\* METAL\s+{kernel_id}\s', line):
            # Check previous few lines for opts (they appear before the *** line)
            for j in range(max(0, i-5), i):
                if 'Opt(op=OptOps' in lines[j]:
                    return lines[j].strip()
            # Also check lines after
            for j in range(i, min(i+5, len(lines))):
                if 'Opt(op=OptOps' in lines[j]:
                    return lines[j].strip()
    return None

def analyze_kernels(kernels):
    """Identify optimization opportunities."""
    issues = []

    for k in kernels:
        # Check for low bandwidth
        if k['bandwidth_gbs'] < 50 and k['mem_gb'] > 0.1:
            issues.append({
                'kernel': k['name'],
                'issue': 'Low bandwidth',
                'value': f"{k['bandwidth_gbs']:.1f} GB/s (should be >100 GB/s)",
                'severity': 'high' if k['total_ms'] > 10 else 'medium',
                'kernel_data': k
            })

        # Check for hot kernels (high total time)
        if k['total_ms'] > 20:
            issues.append({
                'kernel': k['name'],
                'issue': 'Hot path kernel',
                'value': f"{k['total_ms']:.1f}ms total ({k['call_count']} calls)",
                'severity': 'high',
                'kernel_data': k
            })

        # Check for dequant operations (bitcast + cast)
        if 'bitcast' in k['ops'] and 'cast' in k['ops']:
            if k['bandwidth_gbs'] < 200:
                issues.append({
                    'kernel': k['name'],
                    'issue': 'Dequant kernel underperforming',
                    'value': f"{k['bandwidth_gbs']:.1f} GB/s (target: 300+ GB/s)",
                    'severity': 'high' if k['total_ms'] > 5 else 'medium',
                    'kernel_data': k
                })

    return sorted(issues, key=lambda x: (x['severity'] != 'high', -x['kernel_data']['total_ms']))

def print_report(kernels, issues, output, outfile=None):
    """Print analysis report."""
    report = []

    def p(line=""):
        report.append(line)
        if outfile is None:
            print(line)

    p("=" * 80)
    p("KERNEL PERFORMANCE ANALYSIS")
    p("=" * 80)
    p()

    # Summary stats
    total_time = sum(k['total_ms'] for k in kernels)
    p(f"Total kernels: {len(kernels)}")
    p(f"Total GPU time: {total_time:.1f}ms")
    p()

    # Top 10 slowest kernels
    p("=" * 80)
    p("TOP 10 SLOWEST KERNELS (by total time)")
    p("=" * 80)
    top_kernels = sorted(kernels, key=lambda x: x['total_ms'], reverse=True)[:10]
    for i, k in enumerate(top_kernels, 1):
        p(f"\n{i}. {k['name']}")
        p(f"   Total time: {k['total_ms']:.2f}ms ({k['total_ms']/total_time*100:.1f}% of total)")
        p(f"   Per-call: {k['time_us']:.2f}us × {k['call_count']} calls")
        p(f"   Bandwidth: {k['bandwidth_gbs']:.1f} GB/s ({k['mem_gb']:.2f} GB)")
        p(f"   Operations: {', '.join(k['ops'][:5])}")

        # Extract opts
        opts = extract_kernel_opts(output, k['id'])
        if opts:
            p(f"   Applied opts: {opts}")

    # Issues
    p("\n" + "=" * 80)
    p(f"OPTIMIZATION OPPORTUNITIES ({len(issues)} found)")
    p("=" * 80)

    for i, issue in enumerate(issues[:15], 1):  # Top 15 issues
        k = issue['kernel_data']
        p(f"\n{i}. [{issue['severity'].upper()}] {issue['issue']}")
        p(f"   Kernel: {issue['kernel']}")
        p(f"   {issue['value']}")
        p(f"   Impact: {k['total_ms']:.1f}ms ({k['total_ms']/total_time*100:.1f}% of total time)")

        # Show kernel code snippet for top 3
        if i <= 3:
            code = extract_kernel_code(output, issue['kernel'])
            if code:
                lines = code.split('\n')[:15]  # First 15 lines
                p(f"   Code preview:")
                for line in lines:
                    p(f"     {line}")
                if len(code.split('\n')) > 15:
                    p(f"     ... ({len(code.split('\n'))-15} more lines)")

    p("\n" + "=" * 80)

    if outfile:
        Path(outfile).write_text('\n'.join(report))
        print(f"\nFull report saved to: {outfile}")

    return '\n'.join(report)

def main():
    model = sys.argv[1] if len(sys.argv) > 1 else "youtu-llm:2b-Q4_0"
    tokens = int(sys.argv[2]) if len(sys.argv) > 2 else 6

    raw_file = "kernel_analysis_raw.log"

    # Reparse existing log if --reparse flag or raw log exists and no model arg changed
    if "--reparse" in sys.argv or (len(sys.argv) == 1 and Path(raw_file).exists()):
        print(f"Reparsing existing {raw_file}...")
        output = Path(raw_file).read_text()
    else:
        output = run_benchmark(model, tokens)
        Path(raw_file).write_text(output)
        print(f"Raw output saved to: {raw_file}")

    # Parse and analyze
    kernels = parse_kernel_stats(output)
    print(f"Parsed {len(kernels)} kernels")
    issues = analyze_kernels(kernels)

    # Generate report
    report_file = "kernel_analysis_report.txt"
    print_report(kernels, issues, output, report_file)

    return kernels, issues

if __name__ == "__main__":
    main()
