import copy

import pytest

from tools.forecast_serving_canary import error_metrics


def output():
    return {"mean": [10.0] * 16,
            "quantiles": {q: [10.0] * 16 for q in ("0.1", "0.5", "0.9")}}


def test_diagnostics_distinguish_one_bf16_step_from_repeat_noise():
    baseline = output()
    candidate = copy.deepcopy(baseline)
    candidate["mean"][0] += 0.0625
    result = error_metrics(baseline, candidate, [9.0, 11.0])
    assert result["max_bf16_steps"] == 1
    assert result["changed_values"] == 1
    assert result["mae"] == 0.0625 / 64
    assert result["max_context_std_units"] == 0.0625
    assert error_metrics(baseline, baseline, [10, 10])["max_abs"] == 0


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_outputs_fail_closed(bad):
    candidate = output()
    candidate["quantiles"]["0.9"][0] = bad
    with pytest.raises(ValueError):
        error_metrics(output(), candidate, [9, 11])


def test_wrong_output_shape_fails_closed():
    candidate = output()
    candidate["mean"].pop()
    with pytest.raises(ValueError):
        error_metrics(output(), candidate, [9, 11])


def test_compensating_wrong_shapes_do_not_pass_total_length_check():
    candidate = output()
    candidate["mean"].pop()
    candidate["quantiles"]["0.1"].append(10.0)
    with pytest.raises(ValueError):
        error_metrics(output(), candidate, [9, 11])
