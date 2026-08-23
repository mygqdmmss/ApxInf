import unittest

from tools.protocol.run_protocol_gates import (
    build_cases,
    evaluate_row,
    health_max_model_len,
)


class ProtocolGateTests(unittest.TestCase):
    def test_build_cases_contains_frozen_ten_rows(self):
        cases = build_cases(32)
        self.assertEqual(
            [case["id"] for case in cases],
            [
                "malformed_json",
                "empty_input_ids",
                "negative_token_id",
                "out_of_vocabulary_token_id",
                "unsupported_temperature",
                "over_budget",
                "unsupported_modality_field",
                "valid_short_nostream_request",
                "health_after_invalid_requests",
                "health_contract_identity",
            ],
        )
        for case in cases[1:7]:
            self.assertFalse(case["body"]["stream"])

    def test_evaluate_malformed_checks_status_only(self):
        case = build_cases(32)[0]
        self.assertTrue(evaluate_row(case, {"status_code": 400, "response": "not json"}))

    def test_evaluate_structured_errors_requires_json_error(self):
        case = build_cases(32)[1]
        self.assertFalse(evaluate_row(case, {"status_code": 400, "response": "bad"}))
        self.assertTrue(evaluate_row(case, {"status_code": 400, "response": {"error": {"type": "invalid_request"}}}))

    def test_evaluate_short_result_requires_usage(self):
        case = build_cases(32)[7]
        result = {"status_code": 200, "response": {"type": "result", "output_ids": [7], "usage": {"prompt_tokens": 8, "completion_tokens": 1, "total_tokens": 9}}}
        self.assertTrue(evaluate_row(case, result))
        result["response"]["usage"]["total_tokens"] = 8
        self.assertFalse(evaluate_row(case, result))

    def test_health_rows(self):
        cases = build_cases(32)
        self.assertTrue(evaluate_row(cases[8], {"status_code": 200, "response": {"status": "ok"}}))
        self.assertTrue(evaluate_row(cases[9], {"status_code": 200, "response": {"status": "ok", "evaluation_contract": "apxinf.qwen38_27b.inference_interface.v1"}}))

    def test_over_budget_uses_health_max_model_len(self):
        health = {"status_code": 200, "response": {"max_model_len": 1234}}
        max_model_len = health_max_model_len(health)
        self.assertEqual(build_cases(max_model_len)[5]["body"]["max_new_tokens"], 1234)
        with self.assertRaises(ValueError):
            health_max_model_len({"status_code": 200, "response": {"max_model_len": 0}})


if __name__ == "__main__":
    unittest.main()
