"""JSON 분석의 결과와 오류 격리를 검증한다. python3 -m unittest -v"""

from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import main


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
