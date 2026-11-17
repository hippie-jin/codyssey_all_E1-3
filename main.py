"""Mini NPU 시뮬레이터의 기초 함수 (Python 3.8 이상, 표준 라이브러리만 사용)."""

import json
import math
from pathlib import Path
import re
from time import perf_counter


EPSILON = 1e-9
BENCHMARK_REPEATS = 10


def validate_size(size):
    """행렬 크기는 양의 정수여야 한다."""
    if type(size) is not int or size <= 0:
        raise ValueError("크기는 양의 정수여야 합니다.")


def validate_number(value):
    """bool, NaN, 무한대를 제외한 유한한 숫자인지 검사한다."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("행렬 값은 숫자여야 합니다.")
    try:
        finite = math.isfinite(value)
    except OverflowError:
        finite = False
    if not finite:
        raise ValueError("행렬 값은 유한한 숫자여야 합니다.")


def create_grid(size, value=0.0):
    """서로 독립적인 행으로 이루어진 size × size 리스트를 만든다."""
    validate_size(size)
    validate_number(value)
    return [[value for _ in range(size)] for _ in range(size)]


def validate_grid(grid, size=None):
    """숫자로 이루어진 정방행렬을 검증하고 크기를 반환한다."""
    if not isinstance(grid, list) or not grid:
        raise ValueError("행렬은 비어 있지 않은 2차원 리스트여야 합니다.")
    if size is None:
        size = len(grid)
    validate_size(size)
    if len(grid) != size:
        raise ValueError("행 수 오류: {}개가 필요합니다.".format(size))
    for row in grid:
        if not isinstance(row, list) or len(row) != size:
            raise ValueError("열 수 오류: 각 행에 {}개가 필요합니다.".format(size))
        for value in row:
            validate_number(value)
    return size


def validate_position(size, row, col):
    """파이썬의 음수 인덱스를 허용하지 않고 위치를 검사한다."""
    if type(row) is not int or type(col) is not int:
        raise ValueError("행과 열 인덱스는 정수여야 합니다.")
    if not (0 <= row < size and 0 <= col < size):
        raise ValueError("행과 열 인덱스는 0 이상 {} 미만이어야 합니다.".format(size))


def get_value(grid, row, col):
    """0부터 시작하는 행/열 위치의 값을 읽는다."""
    size = validate_grid(grid)
    validate_position(size, row, col)
    return grid[row][col]


def set_value(grid, row, col, value):
    """0부터 시작하는 행/열 위치에 값을 저장한다 (원본 변경)."""
    size = validate_grid(grid)
    validate_position(size, row, col)
    validate_number(value)
    grid[row][col] = value


def mac(pattern, filter_grid):
    """같은 위치의 값을 곱해 누적한다. 호출 전에 크기/값을 검증한다.

    검증을 분리하여 이후 성능 측정에서 연산 구간만 측정할 수 있게 한다.
    """
    score = 0.0
    for row in range(len(pattern)):
        for col in range(len(pattern)):
            score += pattern[row][col] * filter_grid[row][col]
    return score


def judge(score_a, score_b, label_a="A", label_b="B", epsilon=EPSILON):
    """차이가 epsilon보다 작으면 UNDECIDED, 그 외에는 큰 점수의 라벨."""
    validate_number(score_a)
    validate_number(score_b)
    validate_number(epsilon)
    if epsilon <= 0:
        raise ValueError("epsilon은 0보다 커야 합니다.")
    if abs(score_a - score_b) < epsilon:
        return "UNDECIDED"
    return label_a if score_a > score_b else label_b


def normalize_label(label):
    """필터 키와 expected 표기를 Cross 또는 X로 통일한다."""
    if not isinstance(label, str):
        raise ValueError("라벨은 문자열이어야 합니다.")
    labels = {"+": "Cross", "cross": "Cross", "x": "X"}
    normalized = labels.get(label.strip().lower())
    if normalized is None:
        raise ValueError("알 수 없는 라벨: {!r}".format(label))
    return normalized


def input_grid(name, size=3):
    """한 행씩 입력받고, 잘못된 행만 다시 입력받아 정방행렬을 반환한다."""
    validate_size(size)
    print("{} ({}줄 입력, 각 줄에 숫자 {}개를 공백으로 구분)".format(name, size, size))
    grid = []
    while len(grid) < size:
        tokens = input("{}행: ".format(len(grid) + 1)).split()
        try:
            if len(tokens) != size:
                raise ValueError("열 수 불일치")
            row = [float(token) for token in tokens]
            for value in row:
                validate_number(value)
        except ValueError:
            print("입력 형식 오류: 각 줄에 {}개의 유한한 숫자를 공백으로 구분해 입력하세요.".format(size))
            continue
        grid.append(row)
    return grid


def benchmark_mac(pattern, filter_grid, repeats=BENCHMARK_REPEATS):
    """검증 후 MAC을 한 번씩 측정하고 평균 밀리초를 반환한다."""
    if type(repeats) is not int or repeats < 10:
        raise ValueError("측정 횟수는 10 이상의 정수여야 합니다.")
    size = validate_grid(pattern)
    validate_grid(filter_grid, size)
    total_seconds = 0.0
    for _ in range(repeats):
        start = perf_counter()
        score = mac(pattern, filter_grid)
        elapsed = perf_counter() - start
        # 결과 검증도 측정 구간에 포함하지 않는다.
        validate_number(score)
        total_seconds += elapsed
    return total_seconds / repeats * 1000


def print_performance(samples, repeats=BENCHMARK_REPEATS):
    """크기별 (패턴, 필터) 쌍의 MAC 1회 평균 시간과 N²을 출력한다."""
    print("\n[성능 분석: 크기별 {}회 평균, MAC 함수 1회 기준]".format(repeats))
    print("크기(N×N)    평균 시간(ms)    연산 횟수(N²)")
    for size in sorted(samples):
        pattern, filter_grid = samples[size]
        try:
            average_ms = benchmark_mac(pattern, filter_grid, repeats)
            print("{}×{}          {:.6f}          {}".format(
                size, size, average_ms, size * size))
        except (ValueError, OverflowError) as error:
            print("{}×{} 측정 불가: {}".format(size, size, error))


def run_user_mode():
    """필터 두 개와 패턴을 입력받아 MAC 점수와 판정을 출력한다."""
    filter_a = input_grid("필터 A")
    filter_b = input_grid("필터 B")
    print("필터 A, B 저장 완료 (각 3×3)")
    pattern = input_grid("패턴")
    size = validate_grid(pattern)
    validate_grid(filter_a, size)
    validate_grid(filter_b, size)
    score_a = mac(pattern, filter_a)
    score_b = mac(pattern, filter_b)
    # 유한한 입력끼리 계산해도 매우 큰 값은 연산 중 무한대가 될 수 있다.
    try:
        result = judge(score_a, score_b)
    except ValueError:
        print("연산 오류: 점수가 유한한 숫자 범위를 벗어났습니다. 더 작은 값으로 다시 실행하세요.")
        return
    print("\n[MAC 결과]")
    print("A 점수:", score_a)
    print("B 점수:", score_b)
    if result == "UNDECIDED":
        print("판정: 판정 불가 (UNDECIDED, |A-B| < {})".format(EPSILON))
    else:
        print("판정:", result)
    print("성능 측정 대상: 입력 패턴과 필터 A")
    print_performance({size: (pattern, filter_a)})


def load_json_data(path):
    """파일을 읽고 최상위 스키마를 검사한다. 개별 항목은 분석 시 검사한다."""
    with open(path, encoding="utf-8-sig") as stream:
        data = json.load(stream)
    if not isinstance(data, dict):
        raise ValueError("최상위 데이터는 객체여야 합니다.")
    for key in ("filters", "patterns"):
        if not isinstance(data.get(key), dict):
            raise ValueError("{}는 객체여야 합니다.".format(key))
    return data


def prepare_filters(raw_filters):
    """크기별 필터를 정규화하고, 오류가 있는 그룹은 사유를 따로 보관한다."""
    filters = {}
    errors = {}
    print("\n[필터 로드]")
    for key, group in raw_filters.items():
        try:
            match = re.fullmatch(r"size_([1-9][0-9]*)", key)
            if match is None:
                raise ValueError("필터 키는 size_{N} 형식이어야 합니다.")
            size = int(match.group(1))
            if not isinstance(group, dict):
                raise ValueError("필터 그룹은 객체여야 합니다.")
            normalized = {}
            for label, grid in group.items():
                standard = normalize_label(label)
                if standard in normalized:
                    raise ValueError("정규화 후 중복된 필터: {}".format(standard))
                validate_grid(grid, size)
                normalized[standard] = grid
            if set(normalized) != {"Cross", "X"}:
                raise ValueError("Cross와 X 필터가 모두 필요합니다.")
            filters[key] = normalized
            print("{} 필터 로드 완료 (Cross, X)".format(key))
        except ValueError as error:
            errors[key] = str(error)
            print("{} 필터 오류: {}".format(key, error))
    return filters, errors


def analyze_case(key, case, filters, filter_errors):
    """한 케이스를 검증·판정한다. 예상 가능한 데이터 오류는 FAIL로 반환한다."""
    record = {"id": key, "passed": False, "reason": ""}
    try:
        match = re.fullmatch(r"size_([1-9][0-9]*)_([0-9]+)", key)
        if match is None:
            raise ValueError("패턴 키는 size_{N}_{idx} 형식이어야 합니다.")
        size = int(match.group(1))
        if not isinstance(case, dict):
            raise ValueError("패턴 항목은 객체여야 합니다.")
        if "input" not in case or "expected" not in case:
            raise ValueError("input과 expected가 모두 필요합니다.")
        pattern = case["input"]
        validate_grid(pattern, size)
        expected = normalize_label(case["expected"])
        filter_key = "size_{}".format(size)
        if filter_key in filter_errors:
            raise ValueError("{} 필터 오류: {}".format(filter_key, filter_errors[filter_key]))
        if filter_key not in filters:
            raise ValueError("{} 필터가 없습니다.".format(filter_key))
        group = filters[filter_key]
        score_cross = mac(pattern, group["Cross"])
        score_x = mac(pattern, group["X"])
        result = judge(score_cross, score_x, "Cross", "X")
        record.update(scores={"Cross": score_cross, "X": score_x},
                      result=result, expected=expected, passed=result == expected)
        if result == "UNDECIDED":
            record["reason"] = "동점 (점수 차이 < {}), expected={}".format(EPSILON, expected)
        elif not record["passed"]:
            record["reason"] = "판정 {}와 expected {}가 다릅니다.".format(result, expected)
    except (ValueError, OverflowError) as error:
        record["reason"] = "데이터/연산 오류: {}".format(error)
    return record


def print_summary(records):
    """케이스 결과에서 총합과 실패 목록을 계산하여 출력한다."""
    passed = sum(record["passed"] for record in records)
    print("\n[결과 요약]")
    print("총 테스트: {}개 | 통과: {}개 | 실패: {}개".format(
        len(records), passed, len(records) - passed))
    for record in records:
        if not record["passed"]:
            print("- {}: {}".format(record["id"], record["reason"]))


def run_json_mode(path=None):
    """스크립트 옆 data.json을 분석한다. 테스트 시 별도 파일 경로를 받을 수 있다."""
    if path is None:
        path = Path(__file__).resolve().with_name("data.json")
    try:
        data = load_json_data(path)
    except (OSError, UnicodeError, ValueError) as error:
        print("JSON 로드 오류: {}".format(error))
        return []
    filters, filter_errors = prepare_filters(data["filters"])
    records = []
    # data.json에 없는 3×3은 고정 십자가 예제로 측정한다.
    cross_3 = [[0.0, 1.0, 0.0], [1.0, 1.0, 1.0], [0.0, 1.0, 0.0]]
    samples = {3: (cross_3, cross_3)}
    print("\n[패턴 분석]")
    for key, case in data["patterns"].items():
        record = analyze_case(key, case, filters, filter_errors)
        records.append(record)
        print("\n--- {} ---".format(key))
        if "scores" in record:
            print("Cross 점수:", record["scores"]["Cross"])
            print("X 점수:", record["scores"]["X"])
            print("판정: {} | expected: {}".format(record["result"], record["expected"]))
            size = len(case["input"])
            if size in (5, 13, 25) and size not in samples:
                samples[size] = (case["input"], filters["size_{}".format(size)]["Cross"])
        print("PASS" if record["passed"] else "FAIL: " + record["reason"])
    print("\n성능 측정 대상: 3×3 고정 예제, 나머지는 크기별 첫 연산 성공 패턴과 Cross 필터")
    print_performance(samples)
    for size in (5, 13, 25):
        if size not in samples:
            print("{}×{} 측정 생략: 유효한 패턴/필터 쌍이 없습니다.".format(size, size))
    print_summary(records)
    return records


def main():
    """입력 방식을 선택한 뒤 해당 모드를 한 번 실행한다."""
    print("=== Mini NPU Simulator ===")
    print("1. 사용자 입력 (3×3)")
    print("2. data.json 분석")
    try:
        while True:
            choice = input("선택: ").strip()
            if choice == "1":
                run_user_mode()
                break
            if choice == "2":
                run_json_mode()
                break
            print("입력 오류: 1 또는 2를 선택하세요.")
    except (EOFError, KeyboardInterrupt):
        print("\n입력이 중단되어 프로그램을 종료합니다.")


if __name__ == "__main__":
    main()
