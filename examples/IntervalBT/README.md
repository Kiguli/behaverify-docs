# Interval Behavior Trees

An **interval behavior tree** (IBT) is a stochastic behavior tree whose
probabilistic leaves carry *closed probability intervals* rather than point
probabilities.

Probability estimates from trials, manufacturer envelopes or learned modules
rarely justify choosing one exact value. A certificate computed from a single
point probability can therefore be stronger than the calibration actually
supports. An IBT keeps the uncertainty explicit and returns a **certified
interval** `[P_min, P_max]`.

## Running the example

```bash
behaverify prism examples/IntervalBT/delivery.tree ./output --horizon 12
```

Expected output:

```
--- interval behavior tree ---
interval leaves  : 2
  sensor_healthy in [0.9, 0.98]
  step_forward in [0.8, 0.92]
reachable states : 10
--- certified intervals for P[F<=12 target] ---
  spec_0 : [0.999023, 1.000000]  (2 goal states)
```

The `delivery.tree` model advances one cell per tick with probability
`p_sensor * p_step`, so reaching the goal within `H` ticks is exactly "at
least 4 successes in `H` Bernoulli trials". The reported bounds are the
closed-form binomial tails at the two corners of the leaf boxes — which is
what `tests/test_probabilistic.py` asserts.

## The two new leaf types

`interval_check` — succeeds with some unknown *p* in the declared box,
leaving the state unchanged. The failure branch carries the exact complement,
so the two outcomes stay coupled rather than becoming independent parameters.

```
interval_checks {
    interval_check {
        sensor_healthy
        arguments {}
        read_variables {}
        probability { [0.90, 0.98] }
    }
}
```

`interval_action` — a set of branches, each with its own box, update and
returned status. In a two-branch action, one branch may declare `complement`
instead of a box, in which case it carries exactly the complementary mass of
the other.

```
interval_actions {
    interval_action {
        step_forward
        arguments {}
        local_variables {}
        read_variables { pos }
        write_variables { pos, lost_track }
        initial_values {}
        branches {
            branch {
                [0.80, 0.92]
                update { ... return_statement { result { success } } }
            }
            branch {
                complement
                update { ... return_statement { result { failure } } }
            }
        }
    }
}
```

## What the back-end does

1. Builds the IBT from the `.tree` model.
2. Compiles **only the reachable fragment** of the induced interval Markov
   chain (IMC), so the IMC is never materialised in full.
3. Discharges bounded reachability by robust value iteration run directly on
   the tree.
4. Reads each `INVARSPEC` in the model as a reachability target.
5. Writes `<name>_imc.prism` (interval-annotated), `<name>_mid.prism`
   (midpoint DTMC, checkable by stock PRISM) and `<name>.props.pctl`.

## Limitations

Rejected by name rather than silently mis-modelled: arrays, `NEURAL`
variables, `parallel` composites, the `X_is_Y` and `one_shot` decorators, and
`monitor_statement` / `read_environment` / `write_environment` inside leaf
updates. (`X_is_Y` is not in general an injective status relabelling, so it
is not an inverter under the IBT semantics.)

## Provenance

The compilation core lives in `src/behaverify/probabilistic/tickdist.py` and
implements the semantics of *Interval Behavior Trees and Their Verification*
(Wooding & Johnson). It is stdlib-only and carries 53 semantic self-checks,
runnable directly:

```bash
python src/behaverify/probabilistic/tickdist.py test
```
