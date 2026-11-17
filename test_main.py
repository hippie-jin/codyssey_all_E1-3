"""기초 함수, 입력 처리, JSON 분석 및 성능 측정을 검증한다."""

from contextlib import redirect_stdout
from copy import deepcopy
from fractions import Fraction
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import main


class CoreFunctionsTests(unittest.TestCase):
    def test_known_cross_and_x_scores(self):
        cross = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        x = [[1, 0, 1], [0, 1, 0], [1, 0, 1]]
        self.assertEqual(main.mac(cross, cross), 5.0)
        self.assertEqual(main.mac(cross, x), 1.0)
        self.assertEqual(main.mac(x, cross), 1.0)
        self.assertEqual(main.mac(x, x), 5.0)

    def test_mac_all_required_sizes_and_signed_values(self):
        for size in (3, 5, 13, 25):
            with self.subTest(size=size):
                self.assertEqual(main.mac(main.create_grid(size, 1.0),
                                          main.create_grid(size, 1.0)), size * size)
        self.assertEqual(main.mac([[0.5, -2], [0, 3]],
                                  [[2, 0.5], [9, -1]]), -3.0)

    def test_grid_storage_and_independent_rows(self):
        for size in (3, 5, 13, 25):
            with self.subTest(size=size):
                grid = main.create_grid(size)
                main.set_value(grid, 0, 0, 2.5)
                main.set_value(grid, size - 1, size - 1, -1)
                self.assertEqual(main.get_value(grid, 0, 0), 2.5)
                self.assertEqual(main.get_value(grid, size - 1, size - 1), -1)
                self.assertEqual(main.get_value(grid, 1, 0), 0.0)
                self.assertIsNot(grid[0], grid[1])

    def test_invalid_sizes(self):
        for size in (0, -1, True, 3.0, "3", None):
            with self.subTest(size=size), self.assertRaises(ValueError):
                main.create_grid(size)

    def test_invalid_grids_and_numbers(self):
        for grid in ([], None, [[1, 2], [3]], [[1], [2]],
                     [[True]], [["1"]], [[float("nan")]], [[float("inf")]]):
            with self.subTest(grid=grid), self.assertRaises(ValueError):
                main.validate_grid(grid)
        with self.assertRaises(ValueError):
            main.validate_grid([[1]], 3)
        grid = main.create_grid(3)
        with self.assertRaises(ValueError):
            main.set_value(grid, 0, 0, float("nan"))
        self.assertEqual(main.get_value(grid, 0, 0), 0.0)

    def test_invalid_positions(self):
        grid = main.create_grid(3)
        for row, col in ((-1, 0), (0, -1), (3, 0), (0, 3),
                         (True, 0), (0, 1.0)):
            with self.subTest(row=row, col=col):
                with self.assertRaises(ValueError):
                    main.get_value(grid, row, col)
                with self.assertRaises(ValueError):
                    main.set_value(grid, row, col, 1)

    def test_label_aliases_whitespace_and_case(self):
        for label, expected in (("+", "Cross"), ("cross", "Cross"),
                                (" Cross ", "Cross"), ("CROSS", "Cross"),
                                (" + ", "Cross"), ("x", "X"), (" X ", "X")):
            with self.subTest(label=label):
                self.assertEqual(main.normalize_label(label), expected)

    def test_invalid_labels(self):
        for label in (None, 1, True, "", "unknown", "plus"):
            with self.subTest(label=label), self.assertRaises(ValueError):
                main.normalize_label(label)

    def test_judge_ties_boundaries_and_labels(self):
        self.assertEqual(main.judge(5, 1), "A")
        self.assertEqual(main.judge(1, 5), "B")
        self.assertEqual(main.judge(0.9, 0.8999999999999999), "UNDECIDED")
        self.assertEqual(main.judge(0, 0), "UNDECIDED")
        for difference in (main.EPSILON / 2, -main.EPSILON / 2):
            self.assertEqual(main.judge(difference, 0), "UNDECIDED")
        self.assertEqual(main.judge(main.EPSILON, 0), "A")
        self.assertEqual(main.judge(0, main.EPSILON), "B")
        self.assertEqual(main.judge(2, 1, "Cross", "X"), "Cross")
        self.assertEqual(main.judge(1, 2, "Cross", "X"), "X")

    def test_invalid_scores_and_epsilon(self):
        for score in (float("nan"), float("inf"), True, "1"):
            with self.subTest(score=score), self.assertRaises(ValueError):
                main.judge(score, 1)
        for epsilon in (0, -1, float("nan"), float("inf"), True):
            with self.subTest(epsilon=epsilon), self.assertRaises(ValueError):
                main.judge(1, 1, epsilon=epsilon)


class InputGridTests(unittest.TestCase):
    def test_retries_preserve_preceding_rows(self):
        # 1행 저장 후 2행에서 6종의 잘못된 입력을 순서대로 거부한다.
        lines = ["0 1 0", "", "0 1", "0 1 0 1", "x y z", "nan 0 1",
                 "inf 0 1", "1 1 1", "0 1 0"]
        prompts = []

        def read(prompt):
            prompts.append(prompt)
            return lines[len(prompts) - 1]

        output = io.StringIO()
        with patch("builtins.input", side_effect=read), redirect_stdout(output):
            grid = main.input_grid("테스트")
        self.assertEqual(grid, [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        self.assertEqual(prompts, ["1행: "] + ["2행: "] * 7 + ["3행: "])
        self.assertEqual(output.getvalue().count("입력 형식 오류"), 6)

    def test_input_accepts_signed_decimals_and_exponents(self):
        with patch("builtins.input", side_effect=["0.5 -1 1e-3", "0 0 0", "1 1 1"]), \
                redirect_stdout(io.StringIO()):
            grid = main.input_grid("테스트")
        self.assertEqual(grid[0], [0.5, -1.0, 0.001])

    def test_eof_is_propagated_instead_of_retried(self):
        with patch("builtins.input", side_effect=["0 1 0", EOFError]) as read, \
                redirect_stdout(io.StringIO()):
            with self.assertRaises(EOFError):
                main.input_grid("테스트")
        self.assertEqual(read.call_count, 2)

    def test_main_handles_eof_and_keyboard_interrupt(self):
        for error in (EOFError, KeyboardInterrupt):
            with self.subTest(error=error.__name__):
                output = io.StringIO()
                with patch("builtins.input", side_effect=["1", error]), \
                        redirect_stdout(output):
                    main.main()
                self.assertIn("입력이 중단되어 프로그램을 종료합니다.", output.getvalue())


class JsonModeTests(unittest.TestCase):
    def run_data(self, data):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "data.json"
            path.write_text(json.dumps(data), encoding="utf-8")
            output = io.StringIO()
            with redirect_stdout(output):
                records = main.run_json_mode(path)
        return records, output.getvalue()

    def fixture(self):
        return {
            "filters": {"size_1": {"cross": [[2]], "x": [[0]]}},
            "patterns": {"size_1_1": {"input": [[1]], "expected": "+"}},
        }

    def test_provided_data(self):
        output = io.StringIO()
        with redirect_stdout(output):
            records = main.run_json_mode()
        self.assertEqual(len(records), 6)
        self.assertEqual(sum(item["passed"] for item in records), 3)
        self.assertEqual(
            {item["id"] for item in records if not item["passed"]},
            {"size_5_1", "size_13_2", "size_25_1"})
        self.assertTrue(all(item["result"] == "UNDECIDED"
                            for item in records if not item["passed"]))
        for size in (3, 5, 13, 25):
            self.assertRegex(output.getvalue(), r"{}×{}\s+\d+\.\d+\s+{}".format(
                size, size, size * size))
        self.assertLess(output.getvalue().index("[성능 분석"),
                        output.getvalue().index("[결과 요약]"))

    def test_exact_ties_in_original_json(self):
        # JSON의 원래 소수를 float을 거치지 않고 분수로 읽어 독립적으로 재계산한다.
        path = Path(main.__file__).resolve().with_name("data.json")
        with path.open(encoding="utf-8-sig") as stream:
            data = json.load(stream, parse_float=Fraction)
        for key, expected in (("size_5_1", Fraction(9, 10)),
                              ("size_13_2", Fraction(15, 2)),
                              ("size_25_1", Fraction(49, 10))):
            with self.subTest(key=key):
                pattern = data["patterns"][key]["input"]
                group = data["filters"]["_".join(key.split("_")[:2])]
                for label in ("cross", "x"):
                    exact = sum((left * right
                                 for row, filter_row in zip(pattern, group[label])
                                 for left, right in zip(row, filter_row)), Fraction(0))
                    self.assertEqual(exact, expected)

    def test_bad_cases_do_not_stop_following_case(self):
        invalid = {
            "bad_key": {"input": [[1]], "expected": "+"},
            "size_1_2": None,
            "size_1_3": {},
            "size_1_4": {"input": [[1, 2]], "expected": "+"},
            "size_1_5": {"input": [["1"]], "expected": "+"},
            "size_1_6": {"input": [[1]], "expected": "unknown"},
            "size_2_7": {"input": [[1, 0], [0, 1]], "expected": "+"},
            "size_1_8": {"input": [[float("nan")]], "expected": "+"},
        }
        data = self.fixture()
        data["patterns"] = dict(invalid, **data["patterns"])
        records, output = self.run_data(data)
        self.assertEqual(len(records), 9)
        self.assertTrue(records[-1]["passed"])
        self.assertTrue(all(not item["passed"] and item["reason"]
                            for item in records[:-1]))
        self.assertIn("총 테스트: 9개 | 통과: 1개 | 실패: 8개", output)

    def test_bad_filter_groups(self):
        for group in (None, {}, {"cross": [[2]]},
                      {"cross": [[2]], "+": [[2]], "x": [[0]]},
                      {"cross": [[2, 3]], "x": [[0]]}):
            with self.subTest(group=group):
                data = self.fixture()
                data["filters"]["size_1"] = deepcopy(group)
                records, _ = self.run_data(data)
                self.assertFalse(records[0]["passed"])
                self.assertIn("필터 오류", records[0]["reason"])

    def test_top_level_schema_and_empty_patterns(self):
        for data in ([], {}, {"filters": [], "patterns": {}},
                     {"filters": {}, "patterns": []}):
            records, output = self.run_data(data)
            self.assertEqual(records, [])
            self.assertIn("JSON 로드 오류", output)
        records, output = self.run_data({"filters": {}, "patterns": {}})
        self.assertEqual(records, [])
        self.assertIn("총 테스트: 0개", output)

    def test_missing_and_malformed_files(self):
        with TemporaryDirectory() as folder:
            path = Path(folder) / "bad.json"
            for contents in (None, "{", b"\xff"):
                if contents is not None:
                    path.write_bytes(contents if isinstance(contents, bytes)
                                     else contents.encode())
                output = io.StringIO()
                with redirect_stdout(output):
                    self.assertEqual(main.run_json_mode(path), [])
                self.assertIn("JSON 로드 오류", output.getvalue())

    def test_menu_and_path_from_another_directory(self):
        script = Path(main.__file__).resolve()
        with TemporaryDirectory() as folder:
            result = subprocess.run([sys.executable, "-B", str(script)],
                                    input="bad\n2\n", text=True,
                                    capture_output=True, cwd=folder)
        self.assertEqual(result.returncode, 0)
        self.assertEqual(result.stderr, "")
        self.assertIn("1 또는 2", result.stdout)
        self.assertIn("총 테스트: 6개 | 통과: 3개 | 실패: 3개", result.stdout)

    def test_user_mode_still_works(self):
        cross = "0 1 0\n1 1 1\n0 1 0\n"
        x = "1 0 1\n0 1 0\n1 0 1\n"
        result = subprocess.run([sys.executable, "-B", main.__file__],
                                input="1\n" + cross + x + x,
                                text=True, capture_output=True)
        self.assertEqual(result.returncode, 0)
        self.assertIn("판정: B", result.stdout)
        self.assertRegex(result.stdout, r"3×3\s+\d+\.\d+\s+9")


class BenchmarkTests(unittest.TestCase):
    def test_average_units_repeats_and_validation_outside_timer(self):
        events = []
        times = iter(value for i in range(10) for value in (i, i + 0.002))

        def clock():
            events.append("clock")
            return next(times)

        def calculate(pattern, filter_grid):
            events.append("mac")
            return 1.0

        def validate(score):
            events.append("validate")

        with patch.object(main, "perf_counter", side_effect=clock), \
                patch.object(main, "mac", side_effect=calculate) as mac_mock, \
                patch.object(main, "validate_number", side_effect=validate):
            average = main.benchmark_mac([[1]], [[1]])
        self.assertAlmostEqual(average, 2.0)
        self.assertEqual(mac_mock.call_count, 10)
        # 행렬 검증의 숫자 검사 두 번 이후 각 회차는 시계→MAC→시계→검증 순서.
        self.assertEqual(events, ["validate"] * 2 +
                         ["clock", "mac", "clock", "validate"] * 10)

    def test_invalid_input_is_rejected_before_timing(self):
        for pattern, grid, repeats in (([[1]], [[1]], 9),
                                       ([[1]], [[1]], True),
                                       ([[1, 2]], [[1]], 10)):
            with self.subTest(repeats=repeats, pattern=pattern):
                with patch.object(main, "perf_counter") as clock:
                    with self.assertRaises(ValueError):
                        main.benchmark_mac(pattern, grid, repeats)
                    clock.assert_not_called()


if __name__ == "__main__":
    unittest.main()
