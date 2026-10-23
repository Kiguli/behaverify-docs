'''
Tests for the probabilistic back-end (interval behavior trees).

Three layers are covered:
  * the vendored tickdist core still passes its own 53 semantic self-checks;
  * the DSL extension parses and compiles to the IMC we expect, edge for edge;
  * the certified intervals agree with a closed-form answer computed
    independently of the back-end.
'''
import math
import os
import sys
from pathlib import Path

import pytest

src_path = Path(__file__).parent.parent / 'src'
if str(src_path) not in sys.path:
    sys.path.insert(0, str(src_path))

from behaverify.behaverify_common import BTreeException
from behaverify.dsl_to_prism import dsl_to_prism
from behaverify.probabilistic import tickdist

REPO = Path(__file__).parent.parent
METAMODEL = REPO / 'src' / 'behaverify' / 'data' / 'metamodel' / 'behaverify.tx'
DELIVERY = REPO / 'examples' / 'IntervalBT' / 'delivery.tree'


def run(model, out_dir, horizon=12, name='model'):
    return dsl_to_prism(str(METAMODEL), str(model), name, str(out_dir),
                        horizon, 10 ** 6, 0, False)


# --------------------------------------------------------------- the core
def test_tickdist_self_checks_pass():
    '''The vendored IBT core must still agree with its own semantics.'''
    result = tickdist.run_tests(verbose=False)
    assert result['n_failed'] == 0, result
    assert result['n_tests'] >= 53


def test_interval_algebra():
    assert tickdist.imul((0.5, 0.6), (0.5, 0.5)) == (0.25, 0.3)
    # the Minkowski sum is capped at 1, since these are probabilities
    assert tickdist.iadd((0.8, 0.9), (0.4, 0.5)) == (1, 1)
    assert tickdist.iadd((0.1, 0.2), (0.2, 0.3)) == pytest.approx((0.3, 0.5))


def test_interval_check_rejects_bad_box():
    with pytest.raises(ValueError):
        tickdist.IntervalCheck('bad', 0.7, 0.3)
    with pytest.raises(ValueError):
        tickdist.IntervalCheck('bad', -0.1, 0.5)


# ------------------------------------------------------- the DSL extension
def test_delivery_example_compiles(temp_dir):
    result = run(DELIVERY, temp_dir, horizon=12, name='delivery')

    # pos in 0..4 crossed with lost_track, minus the unreachable combinations
    assert result['states'] == 10
    assert set(result['leaves']) == {'sensor_healthy', 'step_forward'}
    assert result['leaves']['sensor_healthy'] == pytest.approx((0.90, 0.98))
    assert result['leaves']['step_forward'] == pytest.approx((0.80, 0.92))


def test_delivery_writes_prism_files(temp_dir):
    run(DELIVERY, temp_dir, name='delivery')
    for suffix in ('_imc.prism', '_mid.prism', '.props.pctl'):
        written = temp_dir / ('delivery' + suffix)
        assert written.is_file(), suffix
        assert written.read_text(encoding='utf-8').strip()


def test_midpoint_dtmc_rows_are_stochastic(temp_dir):
    '''Every row of the exported midpoint DTMC must sum to one.'''
    run(DELIVERY, temp_dir, name='delivery')
    text = (temp_dir / 'delivery_mid.prism').read_text(encoding='utf-8')
    rows = [line for line in text.splitlines() if line.strip().startswith('[] s=')]
    assert len(rows) == 10
    for row in rows:
        body = row.split('->', 1)[1]
        total = sum(float(term.split(':')[0].strip()) for term in body.split('+'))
        assert total == pytest.approx(1.0), row


def test_imc_rows_bracket_one(temp_dir):
    '''A well-formed IMC row has sum(lower) <= 1 <= sum(upper).'''
    run(DELIVERY, temp_dir, name='delivery')
    text = (temp_dir / 'delivery_imc.prism').read_text(encoding='utf-8')
    rows = [line for line in text.splitlines() if line.strip().startswith('[] s=')]
    for row in rows:
        body = row.split('->', 1)[1]
        low = high = 0.0
        for term in body.split('+'):
            head = term.split(':')[0].strip()
            if head.startswith('['):
                (left, right) = head.strip('[] ').split(',')
                low += float(left)
                high += float(right)
            else:
                low += float(head)
                high += float(head)
        assert low <= 1.0 + 1e-9, row
        assert high >= 1.0 - 1e-9, row


# ------------------------------------------------------ numerical agreement
def _binomial_tail(trials, probability, at_least):
    '''P[Binom(trials, probability) >= at_least], computed independently.'''
    return sum(math.comb(trials, k) * probability ** k * (1 - probability) ** (trials - k)
               for k in range(at_least, trials + 1))


def test_certified_interval_matches_closed_form(temp_dir):
    '''
    The delivery tree advances one cell per tick with probability
    p_sensor * p_step and otherwise stays put, so reaching pos=4 within H
    ticks is exactly "at least 4 successes in H Bernoulli trials".  The
    certified interval must be the closed-form answer at the two corners of
    the leaf boxes.
    '''
    horizon = 12
    result = run(DELIVERY, temp_dir, horizon=horizon, name='delivery')
    (certified,) = result['results']

    expected_low = _binomial_tail(horizon, 0.90 * 0.80, 4)
    expected_high = _binomial_tail(horizon, 0.98 * 0.92, 4)

    assert certified['lower'] == pytest.approx(expected_low, abs=1e-9)
    assert certified['upper'] == pytest.approx(expected_high, abs=1e-9)
    assert certified['lower'] <= certified['upper']
    assert certified['goal_states'] == 2


def test_interval_is_monotone_in_the_horizon(temp_dir):
    '''More ticks cannot make a reachability probability smaller.'''
    previous = None
    for horizon in (2, 4, 8, 16):
        out = temp_dir / ('h' + str(horizon))
        result = run(DELIVERY, out, horizon=horizon, name='delivery')
        (certified,) = result['results']
        if previous is not None:
            assert certified['lower'] >= previous['lower'] - 1e-12
            assert certified['upper'] >= previous['upper'] - 1e-12
        previous = certified


def test_point_probabilities_lie_inside_the_certified_interval(temp_dir):
    '''
    Sanity-check the soundness claim: evaluating the compiled chain at the
    midpoint assignment must land inside the certified interval.
    '''
    horizon = 12
    result = run(DELIVERY, temp_dir, horizon=horizon, name='delivery')
    (certified,) = result['results']
    midpoint = _binomial_tail(horizon, 0.94 * 0.86, 4)
    assert certified['lower'] <= midpoint <= certified['upper']


# ----------------------------------------------------------- the guard rails
def test_model_without_interval_leaves_is_rejected(temp_dir):
    '''The probabilistic back-end should refuse a purely deterministic tree.'''
    plain = REPO / 'test_examples' / 'working' / 'abs.tree'
    if not plain.is_file():
        pytest.skip('fixture model not present')
    with pytest.raises(BTreeException) as caught:
        run(plain, temp_dir, name='abs')
    assert 'no interval leaves' in str(caught.value)


def test_unsupported_node_type_is_reported_by_name(temp_dir, tmp_path):
    '''A parallel composite must fail loudly, naming the node.'''
    source = DELIVERY.read_text(encoding='utf-8')
    source = source.replace(
        'composite {\n        mission\n        selector',
        'composite {\n        mission\n        parallel policy success_on_one')
    broken = tmp_path / 'parallel.tree'
    broken.write_text(source, encoding='utf-8')
    with pytest.raises(BTreeException) as caught:
        run(broken, temp_dir, name='parallel')
    assert 'parallel' in str(caught.value)
