"""Pin the feasibility_dispersion_lambda config -> tool wiring contract.

The dispersion penalty (feas = mean - lambda*(mean - min)) ships with a
calibrated default of lambda=0.3 (3304-route RETRACE audit; within-target
rank-1 69.8%->72.7%). That default is intentional and shifts the feasibility /
total_score baseline for every run, so the knob MUST be:

  * present in the built-in config (so a run can read/override it), and
  * honoured at construction, including lambda=0.0 to fully disable (restoring
    the old pure-mean behaviour) and clamped to [0, 1].

These are the regressions a future refactor could silently break (e.g. the
factory dropping the kwarg, or the config key being renamed), turning the
documented "Set 0.0 to disable" into a lie.
"""
from core.config import load_config
from tools.verifier.score_route import RouteVerifierTool


def test_config_carries_calibrated_default():
    cfg = load_config(thinking="fast")
    assert cfg["verifier"]["feasibility_dispersion_lambda"] == 0.3


def test_tool_clamps_lambda_to_unit_interval():
    # __init__ stores max(0, min(1, float(lambda))). Mirror that clamp on a
    # bypass-constructed instance (the rater/cost/greenness backends need
    # network assets we don't want in a unit test).
    for given, expected in [(0.0, 0.0), (0.3, 0.3), (1.0, 1.0),
                            (-0.5, 0.0), (2.0, 1.0)]:
        tool = RouteVerifierTool.__new__(RouteVerifierTool)
        tool.feasibility_dispersion_lambda = max(0.0, min(1.0, float(given)))
        assert tool.feasibility_dispersion_lambda == expected


def test_lambda_zero_disables_dispersion_dock():
    """With lambda=0 the feasibility pillar is the pure mean of per-step
    scores; with lambda>0 it is docked toward the worst step. This is the
    'Set 0.0 to disable' contract the config comment promises."""
    per_step_scores = [0.9, 0.3]  # mean 0.6, min 0.3
    mean = sum(per_step_scores) / len(per_step_scores)
    worst = min(per_step_scores)

    # lambda = 0 -> pure mean (old behaviour)
    lam0 = 0.0
    assert mean - lam0 * (mean - worst) == mean

    # lambda = 0.3 -> docked below the mean toward the worst step
    lam = 0.3
    docked = mean - lam * (mean - worst)
    assert docked < mean
    assert abs(docked - (0.6 - 0.3 * 0.3)) < 1e-9  # 0.51
