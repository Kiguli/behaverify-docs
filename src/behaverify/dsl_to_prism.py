'''
dsl_to_prism.py

The probabilistic back-end.  Where ``dsl_to_nuxmv`` answers "can this ever
happen", this module answers "with what probability, given that the leaf
probabilities are only known to lie in a box".

A ``.tree`` model whose leaves include ``interval_check`` / ``interval_action``
nodes is an *interval behavior tree* (IBT): a stochastic behavior tree whose
probabilistic leaves carry closed probability intervals rather than point
probabilities.  This module

  1. walks the model and builds the IBT node structures of
     ``behaverify.probabilistic.tickdist``,
  2. compiles the reachable fragment of the induced *interval Markov chain*
     (IMC) with ``compile_imc`` -- only reachable states are generated, so the
     IMC is never materialized in full,
  3. discharges bounded-reachability queries by robust value iteration run
     directly on the tree, returning a *certified interval*
     [P_min, P_max] rather than a single number, and
  4. exports the compiled model as PRISM, so the result can be cross-checked.

Each ``INVARSPEC`` in the model's ``specifications`` block is read as a
reachability *target*: the back-end reports the certified interval for
``P[F<=H target]``.  ``LTLSPEC`` / ``CTLSPEC`` entries are ignored here (they
belong to the nuXmv back-end) and are reported as skipped.

Deliberate limitations of this back-end, each raised as a ``BTreeException``
with the offending name rather than silently mis-modelled:

  * arrays, ``NEURAL`` variables and ``DEFINE`` variables with a node scope;
  * ``parallel`` composites, and the ``X_is_Y`` / ``one_shot`` decorators
    (``X_is_Y`` is not in general an injective status relabelling, so it is
    not an ``Inverter`` in the IBT semantics);
  * ``monitor_statement`` / ``read_environment`` / ``write_environment``
    statements inside leaf updates;
  * ``running`` as a leaf outcome is carried through faithfully, but a tree
    that never settles will simply have no reachable absorbing state.

The expression interpreter below evaluates the DSL's prefix expressions
against a *concrete* state.  This is deliberately separate from
``meta_functions``, which evaluates meta-code at generation time.
'''
import argparse
import math
import os
from fractions import Fraction

from behaverify.behaverify_common import BTreeException, is_array, is_neural
from behaverify.check_grammar import validate_model
from behaverify.probabilistic import tickdist
from behaverify.probabilistic.tickdist import (
    Action,
    Branch,
    Check,
    IntervalAction,
    IntervalCheck,
    Inverter,
    Repeat,
    Selector,
    Sequence,
    compile_imc,
    robust_vi,
)

STATUS_MAP = {'success': tickdist.SUC, 'failure': tickdist.FAIL, 'running': tickdist.RUN}


# --------------------------------------------------------------------- state
class State(tuple):
    '''
    An immutable, hashable variable valuation: a sorted tuple of (name, value).

    tickdist memoizes on the state, and ``compile_imc`` uses it as a dict key,
    so the state has to be hashable -- hence a tuple subclass rather than a
    dict.  Lookup is linear, which is fine: models have a handful of variables
    and the cost is dominated by the interval arithmetic.
    '''
    __slots__ = ()

    def __new__(cls, mapping):
        return super().__new__(cls, tuple(sorted(mapping.items())))

    def get(self, name):
        '''The value bound to name, or an error naming the unknown variable.'''
        for (key, value) in self:
            if key == name:
                return value
        raise BTreeException([], 'Probabilistic back-end :: unknown variable ' + str(name))

    def set(self, name, value):
        '''A new State with name rebound; the receiver is left alone.'''
        as_dict = dict(self)
        if name not in as_dict:
            raise BTreeException([], 'Probabilistic back-end :: assignment to unknown variable ' + str(name))
        as_dict[name] = value
        return State(as_dict)

    def as_dict(self):
        '''A mutable copy, for callers that want to build a successor.'''
        return dict(self)

    def __repr__(self):
        return '{' + ', '.join(str(k) + '=' + str(v) for (k, v) in self) + '}'


# ------------------------------------------------------------- interpreter
def _truthy(value, what):
    '''Assert a value is boolean, naming where it came from if not.'''
    if not isinstance(value, bool):
        raise BTreeException([], 'Probabilistic back-end :: ' + what + ' did not evaluate to a boolean, got ' + repr(value))
    return value


def _numeric(value, what):
    '''Assert a value is numeric, naming where it came from if not.'''
    if isinstance(value, bool) or not isinstance(value, (int, float, Fraction)):
        raise BTreeException([], 'Probabilistic back-end :: ' + what + ' did not evaluate to a number, got ' + repr(value))
    return value


def _binary_chain(args, operation):
    '''Left-fold a variadic operator over its already-evaluated arguments.'''
    result = args[0]
    for operand in args[1:]:
        result = operation(result, operand)
    return result


def _apply_function(name, args, where):
    '''Evaluate one already-evaluated prefix function application.'''
    def nums():
        return [_numeric(a, name + ' in ' + where) for a in args]

    def bools():
        return [_truthy(a, name + ' in ' + where) for a in args]

    if name == 'not':
        return not bools()[0]
    if name == 'and':
        return all(bools())
    if name == 'or':
        return any(bools())
    if name == 'implies':
        (left, right) = bools()
        return (not left) or right
    if name in ('equivalent', 'equivalant'):
        (left, right) = bools()
        return left == right
    if name == 'eq':
        return args[0] == args[1]
    if name == 'neq':
        return args[0] != args[1]
    if name == 'lt':
        return nums()[0] < nums()[1]
    if name == 'gt':
        return nums()[0] > nums()[1]
    if name == 'lte':
        return nums()[0] <= nums()[1]
    if name == 'gte':
        return nums()[0] >= nums()[1]
    if name == 'neg':
        return -nums()[0]
    if name == 'abs':
        return abs(nums()[0])
    if name == 'add':
        return _binary_chain(nums(), lambda a, b: a + b)
    if name == 'sub':
        return nums()[0] - nums()[1]
    if name == 'mult':
        return _binary_chain(nums(), lambda a, b: a * b)
    if name == 'idiv':
        (left, right) = nums()
        if right == 0:
            raise BTreeException([], 'Probabilistic back-end :: division by zero in ' + where)
        return int(left // right)
    if name == 'mod':
        (left, right) = nums()
        if right == 0:
            raise BTreeException([], 'Probabilistic back-end :: modulo by zero in ' + where)
        return left % right
    if name == 'rdiv':
        (left, right) = nums()
        if right == 0:
            raise BTreeException([], 'Probabilistic back-end :: division by zero in ' + where)
        return Fraction(left) / Fraction(right) if not isinstance(left, float) and not isinstance(right, float) else left / right
    if name == 'floor':
        return math.floor(nums()[0])
    if name == 'min':
        return min(nums())
    if name == 'max':
        return max(nums())
    if name == 'count':
        return sum(1 for value in bools() if value)
    if name == 'if':
        return args[1] if _truthy(args[0], 'if condition in ' + where) else args[2]
    raise BTreeException(
        [],
        'Probabilistic back-end :: function ' + str(name) + ' (used in ' + where
        + ') is not supported. The probabilistic back-end evaluates expressions over a '
        'concrete state and supports the arithmetic, comparison and logical functions; '
        'temporal and node-status functions belong to the nuXmv back-end.'
    )


def eval_code(code, state, ctx, where):
    '''
    Evaluate a ``code_statement`` against a concrete state.

    @code ::> the textX code_statement object
    @state ::> a State
    @ctx ::> {'constants' : ..., 'enumerations' : ..., 'args' : ...}
    @where ::> human-readable location, used in error messages
    '''
    if code is None:
        raise BTreeException([], 'Probabilistic back-end :: empty expression in ' + where)
    if getattr(code, 'function_call', None) is not None:
        return eval_function(code.function_call, state, ctx, where)
    if getattr(code, 'code_statement', None) is not None:
        return eval_code(code.code_statement, state, ctx, where)
    if getattr(code, 'indexed_var', None) is not None:
        raise BTreeException([], 'Probabilistic back-end :: array indexing (' + str(code.indexed_var) + ' in ' + where + ') is not supported')
    if getattr(code, 'agent_param_ref', None):
        raise BTreeException([], 'Probabilistic back-end :: agent parameters are not supported (' + where + ')')
    atom = getattr(code, 'atom', None)
    if atom is None:
        raise BTreeException([], 'Probabilistic back-end :: could not interpret expression in ' + where)
    if getattr(code, 'read_at', None) is not None or getattr(code, 'trace_num', None) is not None:
        raise BTreeException([], 'Probabilistic back-end :: "at"/"trace" indexing is not supported (' + where + ')')
    return eval_atom(atom, state, ctx, where)


def eval_function(function_call, state, ctx, where):
    '''Evaluate a prefix function application against a concrete state.'''
    name = function_call.function_name
    if name in ('loop', 'case_loop', 'index'):
        raise BTreeException([], 'Probabilistic back-end :: ' + name + ' (in ' + where + ') is not supported; it implies arrays or meta-iteration')
    if getattr(function_call, 'bound', None) is not None:
        raise BTreeException([], 'Probabilistic back-end :: bounded temporal function ' + str(name) + ' (in ' + where + ') is not supported')
    if getattr(function_call, 'node_name', None) is not None and not function_call.values:
        raise BTreeException([], 'Probabilistic back-end :: node-status function ' + str(name) + ' (in ' + where + ') is only meaningful in a nuXmv specification')
    args = [eval_code(value, state, ctx, where) for value in function_call.values]
    return _apply_function(name, args, where)


def eval_atom(atom, state, ctx, where):
    '''An atom is either a literal constant or a reference to a name.'''
    if getattr(atom, 'constant', None) is not None:
        return atom.constant
    reference = getattr(atom, 'reference', None)
    if reference is None:
        raise BTreeException([], 'Probabilistic back-end :: empty atom in ' + where)
    if reference in ctx['args']:
        return ctx['args'][reference]
    if reference in ctx['constants']:
        return ctx['constants'][reference]
    if reference in ctx['enumerations']:
        return reference
    if reference in ctx['variable_names']:
        return state.get(reference)
    # A bare identifier that is not a variable, constant or declared argument
    # is an enumeration value used without declaring it; the grammar checker
    # already rejects that, so reaching here means we mis-parsed something.
    raise BTreeException([], 'Probabilistic back-end :: unresolved reference ' + str(reference) + ' in ' + where)


def eval_assign(assign, state, ctx, where):
    '''Evaluate an ``assign_value`` (a case/default chain) to a single value.'''
    if assign is None:
        raise BTreeException([], 'Probabilistic back-end :: missing assignment in ' + where)
    for case_result in assign.case_results:
        if _truthy(eval_code(case_result.condition, state, ctx, where), 'case condition in ' + where):
            return _single(case_result.values, state, ctx, where)
    return _single(assign.default_result.values, state, ctx, where)


def _single(values, state, ctx, where):
    if len(values) != 1:
        raise BTreeException(
            [],
            'Probabilistic back-end :: non-deterministic assignment in ' + where
            + ' (' + str(len(values)) + ' candidate values). Use an interval_action to '
            'express a probabilistic choice.'
        )
    return eval_code(values[0], state, ctx, where)


def eval_return(return_statement, state, ctx, where):
    '''Evaluate a ``return_statement`` to one of tickdist's statuses.'''
    for case_result in return_statement.case_results:
        if _truthy(eval_code(case_result.condition, state, ctx, where), 'return case in ' + where):
            return STATUS_MAP[case_result.status]
    return STATUS_MAP[return_statement.default_result.status]


def apply_statements(statements, state, ctx, where):
    '''Apply a leaf's update statements in order, returning the new state.'''
    for statement in statements:
        variable_statement = getattr(statement, 'variable_statement', None)
        if variable_statement is None:
            offending = next(
                (kind for kind in ('read_statement', 'write_statement', 'monitor_statement')
                 if getattr(statement, kind, None) is not None),
                'unknown statement'
            )
            raise BTreeException([], 'Probabilistic back-end :: ' + offending + ' in ' + where + ' is not supported')
        if variable_statement.indexed_var_name is not None or variable_statement.assigns:
            raise BTreeException([], 'Probabilistic back-end :: array assignment in ' + where + ' is not supported')
        target = variable_statement.variable
        state = state.set(target.name, eval_assign(variable_statement.assign, state, ctx, where))
    return state


# ---------------------------------------------------------------- tree build
def _probability(code, ctx, where):
    '''Probability bounds are model constants, so they evaluate without a state.'''
    value = eval_code(code, State({}), ctx, where)
    value = _numeric(value, 'probability bound in ' + where)
    if not 0 <= value <= 1:
        raise BTreeException([], 'Probabilistic back-end :: probability bound ' + str(value) + ' in ' + where + ' is outside [0,1]')
    return value


def _leaf_args(leaf, arguments, ctx, where):
    '''Bind a leaf's declared arguments to the values it was created with.'''
    if len(arguments) != len(leaf.arguments):
        raise BTreeException([], 'Probabilistic back-end :: ' + where + ' expects ' + str(len(leaf.arguments)) + ' arguments, got ' + str(len(arguments)))
    bound = dict(ctx['args'])
    for (declared, supplied) in zip(leaf.arguments, arguments):
        bound[declared.argument_name] = eval_code(supplied, State({}), ctx, where)
    return bound


def build_tree(model_node, ctx):
    '''Translate a BehaVerify tree node into a tickdist IBT node.'''
    while hasattr(model_node, 'sub_root'):
        model_node = model_node.sub_root

    if hasattr(model_node, 'leaf'):
        return build_leaf(model_node, ctx)

    node_type = model_node.node_type
    name = model_node.name

    if node_type == 'sequence':
        return Sequence(*[build_tree(child, ctx) for child in model_node.children])
    if node_type == 'selector':
        return Selector(*[build_tree(child, ctx) for child in model_node.children])
    if node_type == 'parallel':
        raise BTreeException([], 'Probabilistic back-end :: parallel composite ' + name + ' is not supported; the IBT tick semantics covers sequence and selector')
    if node_type == 'inverter':
        return Inverter(build_tree(model_node.child, ctx))
    if node_type == 'repeat':
        return Repeat(build_tree(model_node.child, ctx), int(model_node.repeat))
    if node_type in ('X_is_Y', 'one_shot'):
        raise BTreeException(
            [],
            'Probabilistic back-end :: decorator ' + name + ' of type ' + node_type
            + ' is not supported. The IBT semantics relabels statuses by a bijection '
            '(inverter); X_is_Y and one_shot are not bijections.'
        )
    raise BTreeException([], 'Probabilistic back-end :: unsupported node type ' + str(node_type) + ' on node ' + str(name))


def _make_predicate(condition, leaf_ctx, where):
    '''A state predicate closing over one leaf's condition and bindings.'''
    def predicate(state):
        return _truthy(eval_code(condition, state, leaf_ctx, where), 'condition of ' + where)
    return predicate


def _make_update(pre_statements, post_statements, leaf_ctx, where):
    '''A state transformer closing over one leaf's (or branch's) updates.'''
    def update(state):
        state = apply_statements(pre_statements, state, leaf_ctx, where)
        return apply_statements(post_statements, state, leaf_ctx, where)
    return update


def build_leaf(model_node, ctx):
    '''Translate one BehaVerify leaf into the matching tickdist leaf.'''
    leaf = model_node.leaf
    name = model_node.name if model_node.name is not None else leaf.name
    where = 'node ' + name
    args = _leaf_args(leaf, model_node.arguments, ctx, where)
    leaf_ctx = dict(ctx)
    leaf_ctx['args'] = args
    node_type = leaf.node_type

    if node_type in ('check', 'environment_check'):
        return Check(name, _make_predicate(leaf.condition, leaf_ctx, where))

    if node_type == 'action':
        status = _constant_status(leaf.return_statement)
        if status is None:
            raise BTreeException(
                [],
                'Probabilistic back-end :: action ' + name + ' returns a state-dependent '
                'status. Split it into an interval_action whose branches each return a '
                'fixed status, or make the return_statement unconditional.'
            )
        update = _make_update(list(leaf.pre_update_statements),
                              list(leaf.post_update_statements), leaf_ctx, where)
        return Action(name, update, status)

    if node_type == 'interval_check':
        low = _probability(leaf.prob_min, leaf_ctx, where)
        high = _probability(leaf.prob_max, leaf_ctx, where)
        if low > high:
            raise BTreeException([], 'Probabilistic back-end :: ' + where + ' has an empty probability box [' + str(low) + ',' + str(high) + ']')
        return IntervalCheck(name, low, high)

    if node_type == 'interval_action':
        branches = []
        for (index, branch) in enumerate(leaf.branches):
            branch_where = where + ' branch ' + str(index)
            status = _constant_status(branch.return_statement)
            if status is None:
                raise BTreeException([], 'Probabilistic back-end :: ' + branch_where + ' must return a fixed status')
            update = _make_update(list(branch.pre_update_statements),
                                  list(branch.post_update_statements), leaf_ctx, branch_where)
            if branch.complement:
                branches.append(Branch(status, update))
            else:
                branches.append(Branch(status, update,
                                       _probability(branch.prob_min, leaf_ctx, branch_where),
                                       _probability(branch.prob_max, leaf_ctx, branch_where)))
        return IntervalAction(name, branches)

    raise BTreeException([], 'Probabilistic back-end :: unsupported leaf type ' + str(node_type) + ' on node ' + str(name))


def _constant_status(return_statement):
    '''The status if the return_statement is unconditional, else None.'''
    if return_statement.case_results:
        return None
    return STATUS_MAP[return_statement.default_result.status]


# -------------------------------------------------------------- model set-up
def build_context(model, constants, declared_enumerations):
    '''Build the evaluation context, rejecting variable kinds we cannot model.'''
    for variable in model.variables:
        if is_array(variable):
            raise BTreeException([], 'Probabilistic back-end :: array variable ' + variable.name + ' is not supported')
        if is_neural(variable):
            raise BTreeException([], 'Probabilistic back-end :: NEURAL variable ' + variable.name + ' is not supported; neural leaves belong to the nuXmv back-end')
    return {
        'constants': dict(constants),
        'enumerations': set(declared_enumerations),
        'variable_names': {variable.name for variable in model.variables},
        'args': {},
    }


def build_initial_state(model, ctx):
    '''Evaluate each variable's initial assignment, in declaration order.'''
    state = State({variable.name: None for variable in model.variables})
    for variable in model.variables:
        where = 'initial value of ' + variable.name
        state = state.set(variable.name, eval_assign(variable.assign, state, ctx, where))
    return state


def collect_targets(model, ctx):
    '''Each INVARSPEC is read as a bounded-reachability target.'''
    targets = []
    skipped = []
    for (index, specification) in enumerate(model.specifications):
        label = 'spec_' + str(index)
        if specification.spec_type != 'INVARSPEC':
            skipped.append((label, specification.spec_type))
            continue
        code = specification.code_statement
        targets.append((label, lambda state, c=code, lbl=label: _truthy(
            eval_code(c, state, ctx, 'target ' + lbl), 'target ' + lbl)))
    return (targets, skipped)


# ------------------------------------------------------------- PRISM export
def _fmt(value):
    '''PRISM wants plain decimal literals.'''
    return repr(float(value))


def prism_imc(imc, goal_indices, comment):
    '''
    The compiled IMC, as an interval-annotated PRISM module.

    Note this file documents the certified intervals; stock PRISM checks the
    midpoint DTMC written alongside it. Robust value iteration on the IMC is
    done by this back-end, not by PRISM.
    '''
    lines = ['// ' + comment, '// interval-annotated IMC: each edge carries [lower, upper]',
             'dtmc', '', 'module M', '  s : [0..' + str(imc.n() - 1) + '] init 0;']
    for index in range(imc.n()):
        terms = []
        for (position, successor) in enumerate(imc.succ[index]):
            low = imc.lo[index][position]
            high = imc.hi[index][position]
            if (low, high) == (1, 1):
                terms.append('1 : (s\'=' + str(successor) + ')')
            else:
                terms.append('[' + _fmt(low) + ',' + _fmt(high) + '] : (s\'=' + str(successor) + ')')
        lines.append('  [] s=' + str(index) + ' -> ' + ' + '.join(terms) + ';')
    lines.extend(['endmodule', '', _goal_label(goal_indices), ''])
    return '\n'.join(lines)


def prism_midpoint(imc, goal_indices, comment):
    '''The midpoint DTMC: a well-formed PRISM model for cross-checking.'''
    midpoints = {leaf: (box[0] + box[1]) / 2 for (leaf, box) in imc.box.items()}
    lines = ['// ' + comment, '// midpoint DTMC (each interval replaced by its midpoint)',
             'dtmc', '', 'module M', '  s : [0..' + str(imc.n() - 1) + '] init 0;']
    matrix = imc.point_matrix(midpoints)
    for index in range(imc.n()):
        terms = [_fmt(probability) + ' : (s\'=' + str(successor) + ')'
                 for (probability, successor) in zip(matrix[index], imc.succ[index])
                 if probability != 0]
        if not terms:
            terms = ['1 : (s\'=' + str(index) + ')']
        lines.append('  [] s=' + str(index) + ' -> ' + ' + '.join(terms) + ';')
    lines.extend(['endmodule', '', _goal_label(goal_indices), ''])
    return '\n'.join(lines)


def _goal_label(goal_indices):
    '''The PRISM label naming every goal state of the compiled chain.'''
    if not goal_indices:
        return 'label "goal" = false;'
    return 'label "goal" = ' + ' | '.join('s=' + str(index) for index in sorted(goal_indices)) + ';'


def prism_properties(horizon):
    '''The PCTL property file matching the horizon we verified at.'''
    return ('Pmin=? [ F<=' + str(horizon) + ' "goal" ]\n'
            'Pmax=? [ F<=' + str(horizon) + ' "goal" ]\n'
            'P=? [ F<=' + str(horizon) + ' "goal" ]\n')


# ---------------------------------------------------------------- entry point
def dsl_to_prism(metamodel_file, model_file, output_name, write_location,
                 horizon, max_states, recursion_limit, no_checks):
    '''
    Compile a model to an IMC, verify it, and write the PRISM export.

    @metamodel_file ::> points to the file with the metamodel
    @model_file ::> points to the file with the model
    @output_name ::> stem for the written files
    @write_location ::> where to write files
    @horizon ::> the bound H in P[F<=H target]
    @max_states ::> abort if the reachable fragment exceeds this
    @recursion_limit ::> python recursion limit for deep models
    @no_checks ::> skip grammar validation
    '''
    (model, _variables, constants, declared_enumerations) = validate_model(
        metamodel_file, model_file, recursion_limit, no_checks)

    if model.root is None:
        raise BTreeException([], 'Probabilistic back-end :: the model has no tree')

    ctx = build_context(model, constants, declared_enumerations)
    root = build_tree(model.root, ctx)
    initial_state = build_initial_state(model, ctx)

    interval_leaves = tickdist.leaf_boxes(root)
    if not interval_leaves:
        raise BTreeException(
            [],
            'Probabilistic back-end :: this model has no interval leaves, so there is '
            'nothing probabilistic to verify. Add an interval_check or interval_action, '
            'or use the nuxmv back-end.'
        )

    imc = compile_imc(root, initial_state, max_states=max_states)
    (targets, skipped) = collect_targets(model, ctx)

    results = []
    all_goal_indices = set()
    for (label, predicate) in targets:
        goal_states = [state for state in imc.states if predicate(state)]
        goal_indices = {imc.index[state] for state in goal_states}
        all_goal_indices |= goal_indices
        (lower, _) = robust_vi(imc, goal_states, horizon, lower=True)
        (upper, _) = robust_vi(imc, goal_states, horizon, lower=False)
        results.append({
            'label': label,
            'goal_states': len(goal_states),
            'lower': float(lower[0]),
            'upper': float(upper[0]),
        })

    os.makedirs(write_location, exist_ok=True)
    comment = output_name + ': IBT compiled from ' + os.path.basename(model_file) + ', |R|=' + str(imc.n())
    stem = os.path.join(write_location, output_name)
    with open(stem + '_imc.prism', 'w', encoding='utf-8') as file_object:
        file_object.write(prism_imc(imc, all_goal_indices, comment))
    with open(stem + '_mid.prism', 'w', encoding='utf-8') as file_object:
        file_object.write(prism_midpoint(imc, all_goal_indices, comment))
    with open(stem + '.props.pctl', 'w', encoding='utf-8') as file_object:
        file_object.write(prism_properties(horizon))

    print('--- interval behavior tree ---')
    print('interval leaves  : ' + str(len(interval_leaves)))
    for (leaf, box) in sorted(interval_leaves.items()):
        print('  ' + leaf + ' in [' + _fmt(box[0]) + ', ' + _fmt(box[1]) + ']')
    print('reachable states : ' + str(imc.n()))
    print('tickdist calls   : ' + str(imc.calls))
    print('compile seconds  : ' + format(imc.seconds, '.3f'))
    print('--- certified intervals for P[F<=' + str(horizon) + ' target] ---')
    if not results:
        print('  (no INVARSPEC in the model, so there is no reachability target)')
    for result in results:
        print('  ' + result['label'] + ' : [' + format(result['lower'], '.6f') + ', '
              + format(result['upper'], '.6f') + ']  (' + str(result['goal_states']) + ' goal states)')
    for (label, spec_type) in skipped:
        print('  ' + label + ' : skipped (' + spec_type + ' is handled by the nuxmv back-end)')
    print('wrote ' + stem + '_imc.prism, ' + stem + '_mid.prism, ' + stem + '.props.pctl')

    return {
        'states': imc.n(),
        'calls': imc.calls,
        'seconds': imc.seconds,
        'leaves': {leaf: (float(box[0]), float(box[1])) for (leaf, box) in interval_leaves.items()},
        'results': results,
        'skipped': skipped,
    }


def main():
    '''Stand-alone entry point, for running this back-end directly.'''
    parser = argparse.ArgumentParser(description='Compile an interval behavior tree to an IMC and verify it.')
    parser.add_argument('model_file')
    parser.add_argument('location')
    parser.add_argument('--output_name', default=None)
    parser.add_argument('--horizon', type=int, default=10)
    parser.add_argument('--max_states', type=int, default=10 ** 6)
    parser.add_argument('--recursion_limit', type=int, default=1000)
    parser.add_argument('--no_checks', action='store_true')
    args = parser.parse_args()
    metamodel_file = os.path.dirname(os.path.realpath(__file__)) + '/data/metamodel/behaverify.tx'
    output_name = args.output_name or os.path.splitext(os.path.basename(args.model_file))[0]
    dsl_to_prism(metamodel_file, args.model_file, output_name, args.location,
                 args.horizon, args.max_states, args.recursion_limit, args.no_checks)


if __name__ == '__main__':
    main()
