Verification Strategies
=======================

.. rst-class:: lead

   BehaVerify supports four complementary strategies today: symbolic
   model checking, simulation, runtime monitoring, and probabilistic
   verification of interval behavior trees.

.. mermaid::

   flowchart LR
     A[.tree model] --> B[BehaVerify core]
     B --> C[SMV] --> M[nuXmv<br/>model checking]
     B --> P[py_trees] --> S[simulation]
     B --> R[monitor] --> O[runtime enforcement]
     B --> I[IMC] --> V[robust value iteration<br/>+ PRISM export]

1. Symbolic model checking (``nuxmv`` mode)
-------------------------------------------

SMV model + ``specifications { ... }`` block → nuXmv decides
``INVARSPEC`` / ``CTLSPEC`` / ``LTLSPEC`` queries. Fully symbolic and
sound; the fast-forwarding encoding pushes the practical limit to
thousands of BT nodes.

Strengths:

- Exhaustive --- no missed counter-examples.
- Handles integer, enum, and boolean blackboards natively.
- Supports neural-leaf encoding for ACAS-Xu-style problems.

Limitations:

- Requires a nuXmv binary (licence forbids redistribution).
- Treats stochasticity as non-determinism → worst-case bounds only.

2. Simulation (``python`` mode)
-------------------------------

A ``py_trees`` implementation runs for ``--max_iter`` ticks, printing
blackboard state and (optionally) per-node local variables. Simulation
is *not* exhaustive but is useful for:

- Rapid debugging of ``.tree`` logic before invoking nuXmv.
- Generating empirical success / failure rates over random seeds.
- Running against a real sensor/actuator stack via custom readers.

3. Runtime monitoring (``monitor`` / contingency blocks)
--------------------------------------------------------

The ``monitors { ... }`` block compiles an LTL formula into a
stateful monitor that runs alongside the BT and fires a configured
``trigger_action`` when the monitor observes a violation. Suitable for
on-robot deployment where exhaustive verification is either
infeasible or complemented by runtime safety nets.

4. Probabilistic verification (``prism`` mode)
----------------------------------------------

Answers quantitative questions such as :math:`\Pr[F^{\le H}\,\mathrm{goal}]`
on an *interval behavior tree* (IBT): a stochastic behavior tree whose
probabilistic leaves carry closed probability intervals rather than point
probabilities.

Two new leaf types extend the DSL:

- ``interval_check`` --- succeeds with some unknown :math:`p` in a declared
  box :math:`[p_{lo}, p_{hi}]`, leaving the state unchanged. The failure
  branch carries the exact complement :math:`[1-p_{hi}, 1-p_{lo}]`, so the
  two outcomes stay coupled.
- ``interval_action`` --- a set of branches, each with its own box, update
  and returned status. One branch of a two-branch action may declare
  ``complement`` instead of a box.

.. code-block:: text

   interval_checks {
       interval_check {
           sensor_healthy
           arguments {}
           read_variables {}
           probability { [0.90, 0.98] }
       }
   }

Why intervals? Probability estimates from trials, manufacturer envelopes or
learned modules rarely justify choosing one exact value, and a certificate
computed from a single point probability may be stronger than the
calibration supports. An IBT reports a **certified interval**
:math:`[P_{\min}, P_{\max}]` instead of one number.

The back-end:

- builds the IBT from the ``.tree`` model, then compiles **only the
  reachable fragment** of the induced *interval Markov chain* (IMC), so the
  IMC is never materialised in full;
- discharges bounded reachability by robust value iteration run directly on
  the tree;
- reads each ``INVARSPEC`` in the model as a reachability target;
- exports the compiled chain as PRISM (an interval-annotated model plus a
  midpoint DTMC that stock PRISM can check) for cross-validation.

.. code-block:: bash

   behaverify prism examples/IntervalBT/delivery.tree ./output --horizon 12

Known limitations: arrays, ``NEURAL`` variables, ``parallel`` composites and
the ``X_is_Y`` / ``one_shot`` decorators are not supported, and each is
rejected by name rather than silently mis-modelled. (``X_is_Y`` is not in
general an injective status relabelling, so it is not an inverter under the
IBT semantics.)

Summary
-------

.. list-table::
   :header-rows: 1

   * - Strategy
     - Soundness
     - Scales to
     - Handles stochasticity
   * - Model checking (nuXmv)
     - exact
     - 2,000+ nodes
     - as non-determinism
   * - Simulation
     - approximate
     - any size
     - yes, per seed
   * - Runtime monitors
     - per trace
     - any size
     - yes, per trace
   * - Probabilistic (``prism``)
     - certified interval
     - millions of states
     - yes, natively
