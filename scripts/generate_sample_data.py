"""
더 많은 샘플 데이터 생성 스크립트

메모리 패턴과 성능 메트릭 간의 관계를 모사한 합성 데이터 생성
"""

import json
import random
import numpy as np
from pathlib import Path


def generate_memory_pattern(num_entries):
    """랜덤 메모리 패턴 생성"""
    pattern = {}
    for i in range(num_entries):
        addr = f"0x{random.randint(0x55ea1e2e9000, 0x55ea1e2f9000):x}+0"
        value = round(random.uniform(0.0, 6.0), 1)
        pattern[addr] = value
    return pattern


def generate_task():
    """랜덤 task 생성"""
    num_memory_entries = random.randint(1, 5)
    memory_pattern = generate_memory_pattern(num_memory_entries)
    ca = round(random.uniform(0.2, 0.9), 5)

    return {
        "memory_pattern": memory_pattern,
        "ca": ca
    }


def compute_performance_metrics(tasks):
    """task 정보를 기반으로 성능 메트릭 계산 (합성 데이터용 휴리스틱)"""

    # 전체 메모리 패턴 값의 합과 평균 계산
    total_memory_value = 0
    memory_count = 0
    total_ca = 0

    for task in tasks:
        for value in task['memory_pattern'].values():
            total_memory_value += value
            memory_count += 1
        total_ca += task['ca']

    avg_memory_value = total_memory_value / memory_count if memory_count > 0 else 1.0
    avg_ca = total_ca / len(tasks)
    task_count = len(tasks)

    # 기본 실행 시간 (메모리 접근 패턴에 영향받음)
    base_execution = 8.0 + avg_memory_value * 1.2 + task_count * 0.8

    # Global (g): 모든 코어에서 실행, cache affinity 영향 큼
    g_execution = base_execution * (1.0 + (1.0 - avg_ca) * 0.3)
    g_turnaround = g_execution * 1.8 + task_count * 0.5

    # Clustered (c): 중간 성능
    c_execution = base_execution * (1.0 + (1.0 - avg_ca) * 0.15)
    c_turnaround = c_execution * 1.5 + task_count * 0.4

    # Partitioned (p): 각 코어에 고정, 성능이 안정적이지만 느림
    p_execution = base_execution * (1.0 + (1.0 - avg_ca) * 0.05)
    p_turnaround = p_execution * 1.4 + task_count * 0.6

    # 약간의 랜덤 노이즈 추가
    noise_factor = 0.1

    return {
        "g": {
            "execution_time": round(g_execution * (1 + random.uniform(-noise_factor, noise_factor)), 1),
            "turnaround_time": round(g_turnaround * (1 + random.uniform(-noise_factor, noise_factor)), 1)
        },
        "c": {
            "execution_time": round(c_execution * (1 + random.uniform(-noise_factor, noise_factor)), 1),
            "turnaround_time": round(c_turnaround * (1 + random.uniform(-noise_factor, noise_factor)), 1)
        },
        "p": {
            "execution_time": round(p_execution * (1 + random.uniform(-noise_factor, noise_factor)), 1),
            "turnaround_time": round(p_turnaround * (1 + random.uniform(-noise_factor, noise_factor)), 1)
        }
    }


def generate_sample():
    """하나의 샘플 생성"""
    num_tasks = random.randint(1, 4)  # 1~4개의 task
    tasks = [generate_task() for _ in range(num_tasks)]
    performance_metrics = compute_performance_metrics(tasks)

    return {
        "tasks": tasks,
        "performance_metrics": performance_metrics
    }


def main():
    """샘플 데이터 생성 및 저장"""
    random.seed(42)
    np.random.seed(42)

    # 100개의 샘플 생성
    num_samples = 100
    samples = [generate_sample() for _ in range(num_samples)]

    data = {"samples": samples}

    # 파일 저장
    output_path = Path(__file__).parent.parent / "data" / "memory_pattern_large.json"
    with open(output_path, 'w') as f:
        json.dump(data, f, indent=2)

    print(f"Generated {num_samples} samples")
    print(f"Saved to: {output_path}")

    # 통계 출력
    task_counts = [len(sample['tasks']) for sample in samples]
    print(f"\nTask count statistics:")
    print(f"  Min: {min(task_counts)}")
    print(f"  Max: {max(task_counts)}")
    print(f"  Mean: {np.mean(task_counts):.2f}")

    memory_entry_counts = []
    for sample in samples:
        for task in sample['tasks']:
            memory_entry_counts.append(len(task['memory_pattern']))

    print(f"\nMemory entry count statistics:")
    print(f"  Min: {min(memory_entry_counts)}")
    print(f"  Max: {max(memory_entry_counts)}")
    print(f"  Mean: {np.mean(memory_entry_counts):.2f}")


if __name__ == "__main__":
    main()