#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
TickDist -- symbolic compilation of an interval-annotated behavior tree (IBT)
into the per-state transition polytopes of an interval Markov chain (IMC).

Vendored into BehaVerify as the probabilistic back-end core.  The semantics
implemented here are those of "Interval Behavior Trees and Their
Verification" (Wooding & Johnson); ``behaverify.dsl_to_prism`` builds the
node structures below from a ``.tree`` model instead of by hand.

This is a real implementation of Algorithm ``alg:tickdist`` of the letter
(Section III, "Symbolic compilation", and its extended-version pseudocode,
including the Extend helper).  It exists so that a practitioner can verify a
tree without first hand-writing a model file: you declare the tree, and
``compile_imc`` walks the reachable flat state space and emits the IMC.

Semantics implemented (verbatim from the letter):

  chk(phi)                 -> {(Suc,s):[1,1]}  if phi(s) else {(Fail,s):[1,1]}
  act(u,sigma)             -> {(sigma,u(s)):[1,1]}
  ipchk([plo,phigh])       -> {(Suc,s):[plo,phigh], (Fail,s):[1-phigh,1-plo]}
  ipact(([plo_i,phigh_i],u_i,r_i)_i)
                           -> {(r_i,u_i(s)):[plo_i,phigh_i]}_i
  Sequence(n_1..n_k)       -> D<-TickDist(n_1,s); D<-Extend(D,Suc ,n_i) for i=2..k
  Selector(n_1..n_k)       -> D<-TickDist(n_1,s); D<-Extend(D,Fail,n_i) for i=2..k
  Inverter(n)              -> status relabelling Suc<->Fail, intervals untouched
  Repeat(n,k)              -> TickDist(Sequence(n,...,n) with k copies, s)

  Extend(D,sigma*,n): every sigma*-outcome (sigma*,s') of D is continued by
  ticking n AT s' (not at the original s); the two intervals are combined by
  the interval product [a,b].[c,d]=[ac,bd] and outcomes that re-converge on a
  common (status,state) are merged by the capped Minkowski sum
  [a,b]+[c,d]=[min(1,a+c),min(1,b+d)].

Two evaluation modes share one code path:

  * interval mode  -- every interval leaf contributes its calibrated box;
  * point mode     -- a leaf assignment pi in Pi(T) collapses every box to a
                      point, so TickDist returns the one-step distribution of
                      the induced DTMC M_pi.  (Theorem 1 step (i) needs both.)

Every edge is *also* carried in factorized form, matching the convention the
hand-written case studies use (``case3/case3.py`` and ``case4/case4_leg.py``):
a factor is a pair ``(leaf_id, sign)`` with ``sign=+1`` meaning "multiply by
the leaf value" and ``sign=-1`` meaning "multiply by (1 - leaf value)".
Because re-convergence *adds* paths, an edge is in general a *sum* of such
products, i.e. a multi-affine polynomial in the leaf parameters, so the
factorized field of an edge is a tuple of monomials.  This is what lets the
same compiled model be re-evaluated exactly at any leaf assignment.

Arithmetic is type-generic: pass ``float`` for speed or ``fractions.Fraction``
for exact rational answers (the unit tests use Fraction where the letter's
numbers are exact).

CLI
    python tickdist.py demo     # worked trace of the letter's Figure 1
    python tickdist.py test     # unit tests, writes results_tickdist.json
    python tickdist.py json     # just (re)write results_tickdist.json

    python tickdist_case3_check.py [full]
                                # compiles case 3's tree and checks, edge for
                                # edge, that TickDist reproduces the
                                # hand-written outcomes() of case3.py; writes
                                # results_tickdist_case3.json

No .tex file is modified by this module; all reported numbers are
machine-written to results_tickdist.json / results_tickdist_case3.json.
"""
from __future__ import annotations

import json
import os
import platform
import random
import sys
import time
from fractions import Fraction
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence as Seq, Tuple

# ---------------------------------------------------------------- statuses ---
SUC = 'Suc'
FAIL = 'Fail'
RUN = 'Run'
STATUSES = (SUC, FAIL, RUN)

Key = Tuple[str, Any]                     # (status, successor state)
Ivl = Tuple[Any, Any]                     # (lo, hi)
Factor = Tuple[str, int]                  # (leaf_id, sign in {+1,-1})
Monomial = Tuple[Factor, ...]             # a product of factors
Terms = Tuple[Monomial, ...]              # a sum of products


# ------------------------------------------------------- interval algebra ---
def imul(a: Ivl, b: Ivl) -> Ivl:
    """Interval product [a,b].[c,d] = [ac,bd] on [0,1]^2 (letter, Sec. III)."""
    return (a[0] * b[0], a[1] * b[1])


def iadd(a: Ivl, b: Ivl, cap: bool = True) -> Ivl:
    """Capped Minkowski sum [a,b]+[c,d] = [min(1,a+c), min(1,b+d)]."""
    lo, hi = a[0] + b[0], a[1] + b[1]
    if cap:
        lo = lo if lo <= 1 else 1
        hi = hi if hi <= 1 else 1
    return (lo, hi)


ZERO: Ivl = (0, 0)
ONE: Ivl = (1, 1)


# ------------------------------------------------------------- node types ---
class Node:
    """Base class.  Nodes are immutable and hashable so they can be memoized."""
    label = '?'

    def children(self) -> Tuple['Node', ...]:
        return ()

    def leaves(self) -> Iterable['Node']:
        if not self.children():
            yield self
        for c in self.children():
            for l in c.leaves():
                yield l

    def __repr__(self) -> str:                       # pragma: no cover - cosmetic
        cs = self.children()
        if not cs:
            return self.label
        return '%s(%s)' % (self.label, ', '.join(repr(c) for c in cs))


class Check(Node):
    """chk(phi): deterministic blackboard predicate, no state change."""

    def __init__(self, name: str, pred: Callable[[Any], bool]):
        self.name, self.pred = name, pred
        self.label = 'chk_%s' % name


class Action(Node):
    """act(u,sigma): deterministic blackboard update returning a fixed status."""

    def __init__(self, name: str, update: Callable[[Any], Any] = lambda s: s,
                 status: str = SUC):
        if status not in STATUSES:
            raise ValueError('bad status %r' % (status,))
        self.name, self.update, self.status = name, update, status
        self.label = 'act_%s' % name


class IntervalCheck(Node):
    """ipchk([plo,phigh]): succeeds with an unknown p in the box, no update.

    The Fail branch carries the exact complement [1-phigh, 1-plo], so the two
    outcomes are *coupled*: they are the +1 and -1 signs of one leaf id.
    """

    def __init__(self, name: str, lo, hi):
        if not (0 <= lo <= hi <= 1):
            raise ValueError('ipchk %s: bad box [%s,%s]' % (name, lo, hi))
        self.name, self.lo, self.hi = name, lo, hi
        self.label = 'ipchk_%s' % name


class Branch:
    """One branch of an ipact: ([plo,phigh], u, r).

    ``lo``/``hi`` may be omitted for *exactly one* branch of a 2-branch ipact,
    in which case that branch carries the complementary mass of the other one
    and is emitted with sign -1 of the same leaf id.
    """

    def __init__(self, status: str, update: Callable[[Any], Any] = lambda s: s,
                 lo=None, hi=None, param: Optional[str] = None):
        if status not in STATUSES:
            raise ValueError('bad status %r' % (status,))
        self.status, self.update, self.lo, self.hi, self.param = \
            status, update, lo, hi, param

    def __repr__(self) -> str:                       # pragma: no cover - cosmetic
        return 'Branch(%s,[%s,%s])' % (self.status, self.lo, self.hi)


class IntervalAction(Node):
    """ipact(([plo_i,phigh_i], u_i, r_i)_{i=1..k}).

    ``couple`` controls the factorization of a 2-branch ipact:
      'auto' (default) -- if the two boxes are exact complements the branches
                          are emitted as (name,+1) and (name,-1), i.e. p and
                          1-p of a single parameter (this is what the running
                          example of the letter's Fig. 1 means, and it is the
                          convention of case3.py / case4_leg.py);
      True             -- force the coupled form (second box must be omitted
                          or be the exact complement);
      False            -- every branch gets its own independent parameter
                          ``name#i``.
    """

    def __init__(self, name: str, branches: Seq[Branch], couple: Any = 'auto'):
        self.name = name
        self.branches = tuple(branches)
        self.label = 'ipact_%s' % name
        if len(self.branches) < 1:
            raise ValueError('ipact %s: no branches' % name)
        missing = [i for i, b in enumerate(self.branches)
                   if b.lo is None or b.hi is None]
        if len(missing) > 1:
            raise ValueError('ipact %s: at most one branch may omit its box'
                             % name)
        if missing and len(self.branches) != 2:
            raise ValueError('ipact %s: an omitted box needs exactly 2 branches'
                             % name)
        if missing:
            j = missing[0]
            other = self.branches[1 - j]
            b = self.branches[j]
            b.lo, b.hi = 1 - other.hi, 1 - other.lo
            couple = True if couple == 'auto' else couple
        if couple == 'auto':
            couple = (len(self.branches) == 2 and
                      self.branches[0].lo == 1 - self.branches[1].hi and
                      self.branches[0].hi == 1 - self.branches[1].lo)
        if couple and len(self.branches) != 2:
            raise ValueError('ipact %s: coupled form needs 2 branches' % name)
        if couple:
            b0, b1 = self.branches
            if not (b0.lo == 1 - b1.hi and b0.hi == 1 - b1.lo):
                raise ValueError('ipact %s: coupled branches are not exact '
                                 'complements' % name)
        self.couple = bool(couple)
        for i, b in enumerate(self.branches):
            if not (0 <= b.lo <= b.hi <= 1):
                raise ValueError('ipact %s branch %d: bad box [%s,%s]'
                                 % (name, i, b.lo, b.hi))

    def params(self) -> List[Tuple[str, int, Ivl]]:
        """(leaf_id, sign, declared box) for each branch, in branch order."""
        if self.couple:
            b0, b1 = self.branches
            return [(self.name, +1, (b0.lo, b0.hi)),
                    (self.name, -1, (b1.lo, b1.hi))]
        out = []
        for i, b in enumerate(self.branches):
            pid = b.param or ('%s#%d' % (self.name, i)
                              if len(self.branches) > 1 else self.name)
            out.append((pid, +1, (b.lo, b.hi)))
        return out


class Sequence(Node):
    label = 'Sequence'

    def __init__(self, *children: Node):
        if not children:
            raise ValueError('Sequence with no children is undefined '
                             '(the letter does not give a base case)')
        self._children = tuple(children)

    def children(self):
        return self._children


class Selector(Node):
    label = 'Selector'

    def __init__(self, *children: Node):
        if not children:
            raise ValueError('Selector with no children is undefined')
        self._children = tuple(children)

    def children(self):
        return self._children


class Inverter(Node):
    """Decorator: relabels statuses by a bijection, intervals untouched."""
    label = 'Inverter'

    def __init__(self, child: Node, swap: Optional[Dict[str, str]] = None):
        self._children = (child,)
        self.swap = dict(swap) if swap else {SUC: FAIL, FAIL: SUC, RUN: RUN}
        img = sorted(self.swap.values())
        if sorted(self.swap.keys()) != img or len(set(img)) != len(img):
            raise ValueError('Inverter: status relabelling is not a bijection')

    def children(self):
        return self._children


class Repeat(Node):
    """Bounded repeat.  Unrolls, within one tick, into n copies of its child.

    mode='while_success' (the letter's rule) -> Sequence(n_1,...,n_1)
    mode='until_success'                     -> Selector(n_1,...,n_1)
    """
    label = 'Repeat'

    def __init__(self, child: Node, n: int, mode: str = 'while_success'):
        if n < 1:
            raise ValueError('Repeat needs n >= 1')
        if mode not in ('while_success', 'until_success'):
            raise ValueError('bad Repeat mode %r' % (mode,))
        self._children, self.n, self.mode = (child,), int(n), mode
        self.label = 'Repeat[%d,%s]' % (n, mode)

    def children(self):
        return self._children

    def unroll(self) -> Node:
        copies = [self._children[0]] * self.n
        return (Sequence(*copies) if self.mode == 'while_success'
                else Selector(*copies))


# ------------------------------------------------- leaf boxes / assignments --
def leaf_boxes(node: Node) -> Dict[str, Ivl]:
    """leaf_id -> declared box, for every *parameter* of the tree.

    A coupled leaf contributes one entry (its +1 box); the -1 sign is the
    complement of that same entry.
    """
    box: Dict[str, Ivl] = {}

    def put(pid, ivl):
        if pid in box and box[pid] != ivl:
            raise ValueError('leaf id %r declared with two different boxes '
                             '%s and %s' % (pid, box[pid], ivl))
        box[pid] = ivl

    for lf in node.leaves():
        if isinstance(lf, IntervalCheck):
            put(lf.name, (lf.lo, lf.hi))
        elif isinstance(lf, IntervalAction):
            for pid, sign, ivl in lf.params():
                if sign > 0:
                    put(pid, ivl)
    return box


def check_wellformed(node: Node, strict: bool = True) -> Dict[str, Any]:
    """Per-leaf well-formedness: sum_i plo_i <= 1 <= sum_i phigh_i.

    Also reports, per ipact, whether the declared box is *projection tight*,
    i.e. equal to the projection of the calibration polytope
    {p in box : sum p = 1} onto each coordinate.  TickDist copies the declared
    box verbatim, so a non-tight declaration makes D(s) strictly larger than
    the set of achievable one-step distributions (still sound, just looser).
    """
    rep: Dict[str, Any] = {'leaves': [], 'ok': True, 'violations': [],
                           'non_tight': []}
    seen = set()
    for lf in node.leaves():
        if isinstance(lf, IntervalCheck):
            lo_sum = lf.lo + (1 - lf.hi)
            hi_sum = lf.hi + (1 - lf.lo)
            boxes = [(lf.lo, lf.hi), (1 - lf.hi, 1 - lf.lo)]
            kind, name = 'ipchk', lf.name
        elif isinstance(lf, IntervalAction):
            boxes = [(b.lo, b.hi) for b in lf.branches]
            lo_sum = sum(b[0] for b in boxes)
            hi_sum = sum(b[1] for b in boxes)
            kind, name = 'ipact', lf.name
        else:
            continue
        if name in seen:
            continue
        seen.add(name)
        ok = (lo_sum <= 1 <= hi_sum)
        tight = []
        for i, (lo, hi) in enumerate(boxes):
            rest_lo = lo_sum - lo
            rest_hi = hi_sum - hi
            tlo = max(lo, 1 - rest_hi)
            thi = min(hi, 1 - rest_lo)
            tight.append((tlo, thi))
        is_tight = all(t == b for t, b in zip(tight, boxes))
        rep['leaves'].append({'leaf': name, 'kind': kind,
                              'boxes': [[_num(a), _num(b)] for a, b in boxes],
                              'sum_lo': _num(lo_sum), 'sum_hi': _num(hi_sum),
                              'wellformed': bool(ok),
                              'projection_tight': bool(is_tight),
                              'tight_boxes': [[_num(a), _num(b)]
                                              for a, b in tight]})
        if not ok:
            rep['ok'] = False
            rep['violations'].append(name)
        if not is_tight:
            rep['non_tight'].append(name)
    if strict and not rep['ok']:
        raise ValueError('per-leaf well-formedness violated: %s'
                         % rep['violations'])
    return rep


def check_assignment(node: Node, leafvals: Dict[str, Any]) -> None:
    """Verify that ``leafvals`` is a leaf assignment pi in Pi(T)."""
    for lf in node.leaves():
        if isinstance(lf, IntervalCheck):
            if lf.name not in leafvals:
                raise KeyError('leaf assignment misses %r' % lf.name)
            v = leafvals[lf.name]
            if not (lf.lo <= v <= lf.hi):
                raise ValueError('pi(%s)=%s outside [%s,%s]'
                                 % (lf.name, v, lf.lo, lf.hi))
        elif isinstance(lf, IntervalAction):
            tot = 0
            for pid, sign, (lo, hi) in lf.params():
                if sign < 0:
                    tot += 1 - leafvals[pid]
                    continue
                if pid not in leafvals:
                    raise KeyError('leaf assignment misses %r' % pid)
                v = leafvals[pid]
                if not (lo <= v <= hi):
                    raise ValueError('pi(%s)=%s outside [%s,%s]'
                                     % (pid, v, lo, hi))
                tot += v
            if tot != 1:
                raise ValueError('ipact %s: branch probabilities sum to %s != 1'
                                 % (lf.name, tot))


def sample_assignment(node: Node, rng: random.Random,
                      exact: bool = False, denom: int = 1000) -> Dict[str, Any]:
    """Draw a pi in Pi(T) uniformly-ish (exact=True gives Fractions)."""
    out: Dict[str, Any] = {}

    def draw(lo, hi):
        if exact:
            a = int(Fraction(lo) * denom)
            b = int(Fraction(hi) * denom)
            return Fraction(rng.randint(a, b), denom)
        return rng.uniform(float(lo), float(hi))

    for lf in node.leaves():
        if isinstance(lf, IntervalCheck):
            out.setdefault(lf.name, draw(lf.lo, lf.hi))
        elif isinstance(lf, IntervalAction):
            if lf.couple:
                b0 = lf.branches[0]
                out.setdefault(lf.name, draw(b0.lo, b0.hi))
                continue
            pars = lf.params()
            rem = Fraction(1) if exact else 1.0
            for i, (pid, sign, (lo, hi)) in enumerate(pars):
                rest = pars[i + 1:]
                rest_lo = sum(p[2][0] for p in rest)
                rest_hi = sum(p[2][1] for p in rest)
                a = max(lo, rem - rest_hi)
                b = min(hi, rem - rest_lo)
                if i == len(pars) - 1:
                    v = rem
                else:
                    v = draw(a, b)
                out[pid] = v
                rem = rem - v
    return out


# -------------------------------------------------- factorized evaluation ---
def edge_interval(factors: Monomial, box: Dict[str, Ivl]) -> Ivl:
    """Interval of one monomial -- identical to case3.py/case4_leg.py."""
    lo = hi = 1
    for (lid, sign) in factors:
        a, b = box[lid]
        lo = lo * (a if sign > 0 else 1 - b)
        hi = hi * (b if sign > 0 else 1 - a)
    return (lo, hi)


def edge_value(factors: Monomial, leafvals: Dict[str, Any]):
    """Point probability of one monomial -- identical to case3.py."""
    p = 1
    for (lid, sign) in factors:
        v = leafvals[lid]
        p = p * (v if sign > 0 else 1 - v)
    return p


def terms_interval(terms: Terms, box: Dict[str, Ivl], cap: bool = True) -> Ivl:
    """Interval hull of a *sum* of monomials (no coupling exploited)."""
    lo = hi = 0
    for m in terms:
        a, b = edge_interval(m, box)
        lo, hi = lo + a, hi + b
    if cap:
        lo, hi = (lo if lo <= 1 else 1), (hi if hi <= 1 else 1)
    return (lo, hi)


def terms_value(terms: Terms, leafvals: Dict[str, Any]):
    p = 0
    for m in terms:
        p = p + edge_value(m, leafvals)
    return p


# ------------------------------------------------------------- the recursion -
class Edge:
    """One (status, successor) entry: the spec's interval plus its polynomial."""
    __slots__ = ('ivl', 'terms')

    def __init__(self, ivl: Ivl, terms: Terms):
        self.ivl, self.terms = ivl, terms

    def __repr__(self) -> str:                       # pragma: no cover - cosmetic
        return '[%s,%s]' % (_fmt(self.ivl[0]), _fmt(self.ivl[1]))


Dist = Dict[Key, Edge]


class _Ctx:
    def __init__(self, box, leafvals, seq_mode, cap, trace):
        self.box = box              # leaf_id -> box used by the recursion
        self.leafvals = leafvals    # None in interval mode
        self.seq_mode = seq_mode    # 'iterative' (Alg. 1) | 'binary' (Sec. III)
        self.cap = cap
        self.calls = 0
        self.trace: Optional[List[dict]] = [] if trace else None
        self.depth = 0


def _leafbox(ctx: _Ctx, pid: str) -> Ivl:
    if ctx.leafvals is not None:
        v = ctx.leafvals[pid]
        return (v, v)
    return ctx.box[pid]


def _merge(D: Dist, key: Key, ivl: Ivl, terms: Terms, cap: bool) -> None:
    """D(key) <- D(key) (+) ivl, with [0,0] as the missing-key default."""
    e = D.get(key)
    if e is None:
        D[key] = Edge(ivl, terms)
    else:
        e.ivl = iadd(e.ivl, ivl, cap)
        e.terms = e.terms + terms


def _tick(node: Node, s: Any, ctx: _Ctx) -> Dist:
    ctx.calls += 1
    if ctx.trace is not None:
        ctx.depth += 1
        rec = {'depth': ctx.depth, 'node': node.label, 'state': _st(s)}
        ctx.trace.append(rec)
    D = _tick_inner(node, s, ctx)
    if ctx.trace is not None:
        rec['result'] = dist_json(D)
        ctx.depth -= 1
    return D


def _tick_inner(node: Node, s: Any, ctx: _Ctx) -> Dist:
    # ---- leaves ----------------------------------------------------------
    if isinstance(node, Check):
        st = SUC if node.pred(s) else FAIL
        return {(st, s): Edge(ONE, ((),))}

    if isinstance(node, Action):
        return {(node.status, node.update(s)): Edge(ONE, ((),))}

    if isinstance(node, IntervalCheck):
        p = _leafbox(ctx, node.name)
        D: Dist = {}
        _merge(D, (SUC, s), p, (((node.name, +1),),), ctx.cap)
        _merge(D, (FAIL, s), (1 - p[1], 1 - p[0]),
               (((node.name, -1),),), ctx.cap)
        return D

    if isinstance(node, IntervalAction):
        D = {}
        for br, (pid, sign, _decl) in zip(node.branches, node.params()):
            a, b = _leafbox(ctx, pid)
            ivl = (a, b) if sign > 0 else (1 - b, 1 - a)
            _merge(D, (br.status, br.update(s)), ivl, (((pid, sign),),), ctx.cap)
        return D

    # ---- composites ------------------------------------------------------
    if isinstance(node, (Sequence, Selector)):
        target = SUC if isinstance(node, Sequence) else FAIL
        kids = node.children()
        D = _tick(kids[0], s, ctx)
        if ctx.seq_mode == 'binary' and len(kids) > 2:
            tail = (Sequence if isinstance(node, Sequence) else Selector)(*kids[1:])
            return _extend(D, target, tail, ctx)
        for nxt in kids[1:]:
            D = _extend(D, target, nxt, ctx)
        return D

    # ---- decorators ------------------------------------------------------
    if isinstance(node, Inverter):
        D = _tick(node.children()[0], s, ctx)
        out: Dist = {}
        for (st, s2), e in D.items():
            out[(node.swap[st], s2)] = Edge(e.ivl, e.terms)
        return out

    if isinstance(node, Repeat):
        return _tick(node.unroll(), s, ctx)

    raise TypeError('TickDist: unhandled node type %r' % type(node).__name__)


def _extend(D: Dist, target: str, n: Node, ctx: _Ctx) -> Dist:
    """Extend(D, sigma*, n) of the letter's pseudocode."""
    out: Dist = {k: Edge(e.ivl, e.terms) for k, e in D.items() if k[0] != target}
    for (st, s1), e in D.items():
        if st != target:
            continue
        Dc = _tick(n, s1, ctx)                     # ticked AT s1, not at s
        for k2, e2 in Dc.items():
            ivl = imul(e.ivl, e2.ivl)
            terms = tuple(m1 + m2 for m1 in e.terms for m2 in e2.terms)
            _merge(out, k2, ivl, terms, ctx.cap)
    return out


# ------------------------------------------------------------- public API ---
class StateDist:
    """Result of one TickDist invocation at one state."""

    def __init__(self, node: Node, s: Any, dist: Dist, box: Dict[str, Ivl],
                 leafvals, seq_mode: str, cap: bool, calls: int,
                 trace: Optional[List[dict]]):
        self.node, self.state, self.dist = node, s, dist
        self.box, self.leafvals = box, leafvals
        self.seq_mode, self.cap, self.calls, self.trace = \
            seq_mode, cap, calls, trace

    # -- status-tagged views ----------------------------------------------
    def intervals(self) -> Dict[Key, Ivl]:
        return {k: e.ivl for k, e in self.dist.items()}

    def terms(self) -> Dict[Key, Terms]:
        return {k: e.terms for k, e in self.dist.items()}

    def hull(self) -> Dict[Key, Ivl]:
        """Interval obtained by evaluating the factorized polynomial instead."""
        b = ({p: (v, v) for p, v in self.leafvals.items()}
             if self.leafvals is not None else self.box)
        return {k: terms_interval(e.terms, b, self.cap)
                for k, e in self.dist.items()}

    # -- IMC views (statuses stripped) -------------------------------------
    def imc_edges(self) -> List[Tuple[Any, Monomial]]:
        """Parallel-edge form ``[(successor, factors), ...]`` -- exactly the
        representation ``case3.py``/``case4_leg.py`` build by hand."""
        out: List[Tuple[Any, Monomial]] = []
        for (st, s2), e in self.dist.items():
            for m in e.terms:
                out.append((s2, m))
        return out

    def imc(self) -> Dict[Any, Ivl]:
        """successor -> interval, merging statuses by the capped sum."""
        out: Dict[Any, Ivl] = {}
        for (st, s2), e in self.dist.items():
            out[s2] = iadd(out.get(s2, ZERO), e.ivl, self.cap)
        return out

    def imc_terms(self) -> Dict[Any, Terms]:
        out: Dict[Any, Terms] = {}
        for (st, s2), e in self.dist.items():
            out[s2] = out.get(s2, ()) + e.terms
        return out

    # -- checks ------------------------------------------------------------
    def row_sums(self, imc: bool = False) -> Tuple[Any, Any]:
        d = self.imc() if imc else self.intervals()
        return (sum(v[0] for v in d.values()), sum(v[1] for v in d.values()))

    def check(self, strict: bool = True) -> Dict[str, Any]:
        """sum(lo) <= 1 <= sum(hi) on both the tagged map and its projection."""
        rep = {}
        ok = True
        for name, is_imc in (('tagged', False), ('imc', True)):
            lo, hi = self.row_sums(is_imc)
            good = (lo <= 1 <= hi)
            rep[name] = {'sum_lo': _num(lo), 'sum_hi': _num(hi),
                         'ok': bool(good)}
            ok = ok and good
        for k, e in self.dist.items():
            if not (0 <= e.ivl[0] <= e.ivl[1] <= 1):
                ok = False
                rep.setdefault('bad_edges', []).append(str(k))
        rep['ok'] = bool(ok)
        if strict and not ok:
            raise ValueError('distribution well-formedness violated at %s: %s'
                             % (_st(self.state), rep))
        return rep

    def __repr__(self) -> str:                       # pragma: no cover - cosmetic
        items = ', '.join('(%s,%s):[%s,%s]' % (k[0], _st(k[1]),
                                               _fmt(e.ivl[0]), _fmt(e.ivl[1]))
                          for k, e in self.dist.items())
        return '{%s}' % items


def tickdist_full(node: Node, s: Any, leafvals: Optional[Dict[str, Any]] = None,
                  seq_mode: str = 'iterative', cap: bool = True,
                  trace: bool = False, check: bool = True) -> StateDist:
    """Full result object (intervals + factorization + IMC views)."""
    if seq_mode not in ('iterative', 'binary'):
        raise ValueError('seq_mode must be "iterative" (Alg. 1) or "binary" '
                         '(Sec. III prose)')
    box = leaf_boxes(node)
    if leafvals is not None:
        check_assignment(node, leafvals)
    ctx = _Ctx(box, leafvals, seq_mode, cap, trace)
    D = _tick(node, s, ctx)
    sd = StateDist(node, s, D, box, leafvals, seq_mode, cap, ctx.calls,
                   ctx.trace)
    if check:
        sd.check(strict=True)
    return sd


def tickdist(node: Node, s: Any, leafvals: Optional[Dict[str, Any]] = None,
             seq_mode: str = 'iterative', cap: bool = True,
             check: bool = True) -> Dict[Key, Ivl]:
    """TickDist(N, s) -> {(status, successor state): (lo, hi)}.

    Interval mode (``leafvals is None``) returns the calibrated intervals;
    point mode (``leafvals`` a leaf assignment pi in Pi(T)) returns degenerate
    intervals (p, p), i.e. the one-step distribution of the DTMC M_pi.
    """
    return tickdist_full(node, s, leafvals, seq_mode, cap,
                         check=check).intervals()


def tickdist_point(node: Node, s: Any, leafvals: Dict[str, Any],
                   seq_mode: str = 'iterative') -> Dict[Key, Any]:
    """Point mode as plain numbers."""
    return {k: v[0] for k, v in
            tickdist(node, s, leafvals, seq_mode).items()}


# ------------------------------------------------------- IMC materialization -
class IMC:
    """Reachable-fragment IMC compiled from an IBT -- no model file by hand."""

    def __init__(self, states, index, edges, lo, hi, terms, node, box,
                 seconds, calls):
        self.states, self.index = states, index
        self.edges = edges          # per state: [(succ_idx, factors), ...]
        self.lo, self.hi = lo, hi   # per state: [interval per distinct succ]
        self.succ = terms           # per state: [succ_idx per distinct succ]
        self.node, self.box = node, box
        self.seconds, self.calls = seconds, calls

    def n(self):
        return len(self.states)

    def point_matrix(self, leafvals):
        """Row-stochastic point probabilities for one leaf assignment."""
        P = []
        for i in range(self.n()):
            acc: Dict[int, Any] = {}
            for (j, m) in self.edges[i]:
                acc[j] = acc.get(j, 0) + edge_value(m, leafvals)
            P.append([acc.get(j, 0) for j in self.succ[i]])
        return P


def compile_imc(node: Node, s0: Any, seq_mode: str = 'iterative',
                cap: bool = True, max_states: int = 10 ** 6) -> IMC:
    """BFS over the reachable flat states, calling TickDist once per state."""
    t0 = time.perf_counter()
    box = leaf_boxes(node)
    check_wellformed(node, strict=True)          # per-leaf, once for the tree
    index = {s0: 0}
    states = [s0]
    edges: List[List[Tuple[int, Monomial]]] = []
    lo: List[List[Any]] = []
    hi: List[List[Any]] = []
    succ: List[List[int]] = []
    calls = 0
    q = [s0]
    head = 0
    while head < len(q):
        s = q[head]
        head += 1
        sd = tickdist_full(node, s, None, seq_mode, cap, check=True)
        calls += sd.calls
        ivl = sd.imc()
        order = list(ivl.keys())
        for t in order:
            if t not in index:
                if len(states) >= max_states:
                    raise RuntimeError('state space exceeds %d' % max_states)
                index[t] = len(states)
                states.append(t)
                q.append(t)
        succ.append([index[t] for t in order])
        lo.append([ivl[t][0] for t in order])
        hi.append([ivl[t][1] for t in order])
        tm = sd.imc_terms()
        edges.append([(index[t], m) for t in order for m in tm[t]])
    return IMC(states, index, edges, lo, hi, succ, node, box,
               time.perf_counter() - t0, calls)


# ------------------------------------------------------- robust / point VI ---
def omax(vals, lo, hi, lower: bool):
    """One O-maximization backup (Givan et al.): O(d log d)."""
    idx = sorted(range(len(vals)), key=lambda i: vals[i], reverse=not lower)
    rem = 1 - sum(lo)
    if rem < 0:
        raise ValueError('sum of lower bounds exceeds 1')
    w = list(lo)
    for i in idx:
        if rem <= 0:
            break
        add = hi[i] - lo[i]
        add = add if add < rem else rem
        w[i] += add
        rem -= add
    if rem > 0:
        raise ValueError('sum of upper bounds below 1')
    return sum(wi * vals[i] for i, wi in enumerate(w))


def robust_vi(imc: IMC, target, H: int, lower: bool = True):
    """H sweeps of robust value iteration for F^{<=H} target."""
    tgt = set(target if not callable(target) else
              [s for s in imc.states if target(s)])
    tset = set(imc.index[s] for s in tgt if s in imc.index)
    V = [1 if i in tset else 0 for i in range(imc.n())]
    table = [list(V)]
    for _ in range(H):
        Vn = []
        for i in range(imc.n()):
            if i in tset:
                Vn.append(1)
                continue
            vals = [V[j] for j in imc.succ[i]]
            Vn.append(omax(vals, imc.lo[i], imc.hi[i], lower))
        V = Vn
        table.append(list(V))
    return V, table


def point_vi(imc: IMC, leafvals, target, H: int):
    tgt = set(target if not callable(target) else
              [s for s in imc.states if target(s)])
    tset = set(imc.index[s] for s in tgt if s in imc.index)
    P = imc.point_matrix(leafvals)
    V = [1 if i in tset else 0 for i in range(imc.n())]
    table = [list(V)]
    for _ in range(H):
        Vn = []
        for i in range(imc.n()):
            if i in tset:
                Vn.append(1)
                continue
            Vn.append(sum(p * V[j] for p, j in zip(P[i], imc.succ[i])))
        V = Vn
        table.append(list(V))
    return V, table


# ------------------------------------------------------------------ helpers -
def _num(x):
    if isinstance(x, Fraction):
        return float(x)
    return x


def _fmt(x):
    if isinstance(x, Fraction):
        return '%d/%d' % (x.numerator, x.denominator) if x.denominator != 1 \
            else str(x.numerator)
    return ('%g' % x)


def _st(s):
    return s if isinstance(s, (int, str, float)) else str(s)


def _exact(x):
    if isinstance(x, Fraction):
        return '%d/%d' % (x.numerator, x.denominator)
    return repr(x)


def dist_json(d) -> Dict[str, Any]:
    """{(status,state):...} -> JSON-safe dict with floats and exact strings."""
    if isinstance(d, StateDist):
        d = d.dist
    out = {}
    for k, e in d.items():
        ivl = e.ivl if isinstance(e, Edge) else e
        key = '%s,%s' % (k[0], _st(k[1])) if isinstance(k, tuple) else _st(k)
        rec = {'lo': float(ivl[0]), 'hi': float(ivl[1]),
               'lo_exact': _exact(ivl[0]), 'hi_exact': _exact(ivl[1])}
        if isinstance(e, Edge):
            rec['terms'] = [[[lid, sg] for (lid, sg) in m] for m in e.terms]
        out[key] = rec
    return out


# ============================================================== example trees =
def fig1_tree(F=Fraction):
    """The letter's Figure 1: two-stage pickup-and-deliver.

    state = stage in {0,1,2};  s0 = need to pick, s1 = carrying, s2 = delivered.
    root = Selector(deliver, pick, chk_{s2}) with
      deliver = Sequence(chk_{s1}, ipact_deliver [.85,.95])
      pick    = Sequence(chk_{s0}, ipact_pick    [.70,.90])
    """
    def box(a, b):
        return (F(a, 100), F(b, 100)) if F is Fraction else (a / 100.0, b / 100.0)

    dl_lo, dl_hi = box(85, 95)
    pk_lo, pk_hi = box(70, 90)
    deliver = Sequence(
        Check('s1', lambda s: s == 1),
        IntervalAction('deliver',
                       [Branch(SUC, lambda s: 2, dl_lo, dl_hi),
                        Branch(FAIL, lambda s: s)]))       # complement branch
    pick = Sequence(
        Check('s0', lambda s: s == 0),
        IntervalAction('pick',
                       [Branch(SUC, lambda s: 1, pk_lo, pk_hi),
                        Branch(FAIL, lambda s: s)]))
    return Selector(deliver, pick, Check('s2', lambda s: s == 2))


def reconv_tree(F=Fraction):
    """Sequence whose two paths re-converge on (Fail, s): exercises the cap.

    Sequence(ipchk a[1/4,1/2], ipchk b[1/4,1/2]) at any state s gives
      (Suc ,s) = a.b                    = [1/16, 1/4]
      (Fail,s) = (1-a) (+) a.(1-b)      = [1/2,3/4] (+) [1/8,3/8]
                                        = [5/8, min(1,9/8)] = [5/8, 1]
    """
    q = (F(1, 4), F(1, 2)) if F is Fraction else (0.25, 0.5)
    return Sequence(IntervalCheck('a', *q), IntervalCheck('b', *q))


def selector_tree(F=Fraction):
    """Mirror of reconv_tree: Selector re-converges on (Suc, s)."""
    q = (F(1, 4), F(1, 2)) if F is Fraction else (0.25, 0.5)
    return Selector(IntervalCheck('a', *q), IntervalCheck('b', *q))


def projection_tree(F=Fraction):
    """Two paths that re-converge only *after* stripping status tags."""
    half, tq = (F(1, 2), F(3, 4)) if F is Fraction else (0.5, 0.75)
    qt = F(1, 4) if F is Fraction else 0.25
    a = IntervalAction('a', [Branch(SUC, lambda s: s, half, tq),
                             Branch(FAIL, lambda s: s + 1, qt, half)],
                       couple=False)
    b = IntervalAction('b', [Branch(SUC, lambda s: s + 1, qt, half),
                             Branch(FAIL, lambda s: s, half, tq)],
                       couple=False)
    return Sequence(a, b)


def branchy_tree(k: int, F=Fraction):
    """Sequence of k ipacts that each continue on TWO Suc branches.

    Every child continues on several branches, so the number of outcome traces
    is 2^k: this is the case main.tex calls "exponential in its arity" and the
    case the extended version's cost claim ("O(|paths(T)|), the number of
    non-Fail leaves of T") gets wrong.
    """
    b = (F(2, 5), F(3, 5)) if F is Fraction else (0.4, 0.6)
    kids = [IntervalAction('b%d' % i,
                           [Branch(SUC, lambda s: s + ('L',), *b),
                            Branch(SUC, lambda s: s + ('R',), *b)],
                           couple=False)
            for i in range(k)]
    return Sequence(*kids)


def scratch_tree(F=Fraction):
    """Intra-tick scratch variables.

    Real trees (e.g. case4_leg.py) need a flag that one subtree writes and a
    later subtree reads *within the same tick* -- here ``fix``, set by the
    localize subtree and read by the motion subtree.  Sigma in the letter has
    no room for such a flag, so the tick is run on an extended state
    (s, scratch) and a final deterministic action projects the scratch away.
    TickDist needs no change for this: the state is any hashable object.
    """
    q = (F(3, 5), F(4, 5)) if F is Fraction else (0.6, 0.8)
    localize = Selector(
        Sequence(IntervalCheck('fix_ok', *q),
                 Action('set_fix', lambda s: (s[0], 1))),
        Action('no_fix', lambda s: (s[0], 0), SUC))
    motion = Selector(
        Sequence(Check('have_fix', lambda s: s[1] == 1),
                 Action('track', lambda s: (s[0] + 1, s[1]))),
        Action('drift', lambda s: (s[0] - 1, s[1]), SUC))
    project = Action('project', lambda s: (s[0], 0))
    return Sequence(localize, motion, project)


def karity_tree(F=Fraction):
    """Sequence(n1,n2,n3) where Alg. 1's iterative Extend and Sec. III's
    binary 'extend by Sequence(n2..nk)' give *different* intervals."""
    half = F(1, 2) if F is Fraction else 0.5
    f4 = (F(2, 5), F(4, 5)) if F is Fraction else (0.4, 0.8)
    n1 = IntervalAction('g', [Branch(SUC, lambda s: 'a', half, half),
                              Branch(FAIL, lambda s: 'z', half, half)],
                        couple=False)
    n2 = IntervalAction('h', [Branch(SUC, lambda s: s, *f4),
                              Branch(FAIL, lambda s: s, *f4)], couple=False)
    n3 = IntervalAction('k', [Branch(SUC, lambda s: s, *f4),
                              Branch(FAIL, lambda s: s, *f4)], couple=False)
    return Sequence(n1, n2, n3)


# ==================================================================== tests ==
class T:
    """Tiny assertion recorder so every check lands in the JSON."""

    def __init__(self):
        self.rows: List[dict] = []
        self.failed = 0

    def eq(self, name, got, want, note=''):
        ok = (got == want)
        self.rows.append({'test': name, 'ok': bool(ok), 'exact_equality': True,
                          'got': _js(got), 'want': _js(want), 'note': note})
        if not ok:
            self.failed += 1
            print('FAIL %s\n  got  %s\n  want %s' % (name, got, want))
        return ok

    def true(self, name, cond, detail=None, note=''):
        ok = bool(cond)
        self.rows.append({'test': name, 'ok': ok, 'exact_equality': False,
                          'detail': _js(detail), 'note': note})
        if not ok:
            self.failed += 1
            print('FAIL %s  detail=%s' % (name, detail))
        return ok


def _js(x):
    if isinstance(x, Fraction):
        return [float(x), '%d/%d' % (x.numerator, x.denominator)]
    if isinstance(x, dict):
        return {('%s,%s' % k if isinstance(k, tuple) else str(k)): _js(v)
                for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_js(v) for v in x]
    return x


def _ivl_map(sd_or_dict):
    d = sd_or_dict.intervals() if isinstance(sd_or_dict, StateDist) \
        else sd_or_dict
    return {(k[0], k[1]): (v[0], v[1]) for k, v in d.items()}


def run_tests(verbose=True) -> Dict[str, Any]:
    t = T()
    F = Fraction
    res: Dict[str, Any] = {}

    # ---------------------------------------------------------------- 1 ----
    # The letter's Figure 1, exact rational arithmetic.
    tree = fig1_tree()
    check_wellformed(tree)
    per_state = {}
    for s in (0, 1, 2):
        sd = tickdist_full(tree, s, trace=(s == 0))
        per_state[s] = sd
    got_imc = {s: {k: (v[0], v[1]) for k, v in per_state[s].imc().items()}
               for s in (0, 1, 2)}
    want_imc = {
        0: {1: (F(70, 100), F(90, 100)), 0: (F(10, 100), F(30, 100))},
        1: {2: (F(85, 100), F(95, 100)), 1: (F(5, 100), F(15, 100))},
        2: {2: (F(1), F(1))},
    }
    t.eq('fig1.imc_edges', got_imc, want_imc,
         'IMC edges printed in Fig. 1(b): s0->s1 [.70,.90], s0->s0 [.10,.30], '
         's1->s2 [.85,.95], s1->s1 [.05,.15], s2->s2 [1,1]')
    t.eq('fig1.tagged_at_s0', _ivl_map(per_state[0]),
         {(SUC, 1): (F(70, 100), F(90, 100)),
          (FAIL, 0): (F(10, 100), F(30, 100))}, 'status-tagged outcomes at s0')
    t.eq('fig1.tagged_at_s1', _ivl_map(per_state[1]),
         {(SUC, 2): (F(85, 100), F(95, 100)),
          (FAIL, 1): (F(5, 100), F(15, 100))}, 'status-tagged outcomes at s1')
    t.eq('fig1.tagged_at_s2', _ivl_map(per_state[2]),
         {(SUC, 2): (F(1), F(1))}, 'absorbing target')

    # factorization: exactly the (leaf_id, sign) convention of case3.py
    t.eq('fig1.factors_at_s0',
         sorted((s2, m) for (s2, m) in per_state[0].imc_edges()),
         sorted([(0, (('pick', -1),)), (1, (('pick', +1),))]),
         'each edge is a product of (leaf_id, sign) factors')

    # the worked trace of Remark "alg:tickdist trace"
    deliver, pick, chk2 = tree.children()
    t.eq('fig1.trace.deliver_at_s0', _ivl_map(tickdist(deliver, 0)),
         {(FAIL, 0): (F(1), F(1))}, 'Remark: deliver returns {(Fail,s0):[1,1]}')
    t.eq('fig1.trace.pick_at_s0', _ivl_map(tickdist(pick, 0)),
         {(SUC, 1): (F(70, 100), F(90, 100)),
          (FAIL, 0): (F(10, 100), F(30, 100))},
         'Remark: TickDist(pick,s0)')
    t.eq('fig1.trace.chk_s2_at_s0', _ivl_map(tickdist(chk2, 0)),
         {(FAIL, 0): (F(1), F(1))}, 'Remark: chk_{s2} fails at stage 0')

    # ---------------------------------------------------------------- 2 ----
    # Sequence with re-converging paths -> capped Minkowski sum.
    rt = reconv_tree()
    sd = tickdist_full(rt, 's')
    t.eq('reconv.sequence', _ivl_map(sd),
         {(SUC, 's'): (F(1, 16), F(1, 4)), (FAIL, 's'): (F(5, 8), F(1))},
         'hand-computed: (Suc)=a.b=[1/16,1/4]; '
         '(Fail)=(1-a)+a(1-b)=[1/2,3/4]+[1/8,3/8]=[5/8,min(1,9/8)]')
    t.eq('reconv.terms', sd.terms()[(FAIL, 's')],
         ((('a', -1),), (('a', +1), ('b', -1))),
         'the re-converged edge is a SUM of two monomials')
    t.eq('reconv.cap_binds', sd.hull()[(FAIL, 's')], (F(5, 8), F(1)),
         'terms_interval caps too')
    t.eq('reconv.uncapped_hull',
         terms_interval(sd.terms()[(FAIL, 's')], sd.box, cap=False),
         (F(5, 8), F(9, 8)),
         'without the cap the polynomial hull leaves the simplex (9/8)')

    # Selector: mirror image.
    st = tickdist_full(selector_tree(), 's')
    t.eq('reconv.selector', _ivl_map(st),
         {(SUC, 's'): (F(3, 8), F(7, 8)), (FAIL, 's'): (F(1, 4), F(9, 16))},
         'Selector extends Fail: (Fail)=(1-a)(1-b)=[1/4,9/16]; '
         '(Suc)=a (+) (1-a).b=[1/4,1/2](+)[1/8,3/8]=[3/8,7/8] (no cap here)')
    t.eq('reconv.selector.terms', st.terms()[(SUC, 's')],
         ((('a', +1),), (('a', -1), ('b', +1))),
         'the Selector re-converges two paths onto (Suc,s)')

    # projection-level re-convergence (two statuses, one successor)
    pt = tickdist_full(projection_tree(), 0)
    t.eq('reconv.projection.tagged', _ivl_map(pt),
         {(FAIL, 1): (F(1, 4), F(1, 2)),
          (SUC, 1): (F(1, 8), F(3, 8)),
          (FAIL, 0): (F(1, 4), F(9, 16))}, 'before stripping statuses')
    t.eq('reconv.projection.imc',
         {k: (v[0], v[1]) for k, v in pt.imc().items()},
         {1: (F(3, 8), F(7, 8)), 0: (F(1, 4), F(9, 16))},
         'stripping statuses merges (Fail,1) and (Suc,1) by the capped sum')

    # ---------------------------------------------------------------- 3 ----
    # Inverter and Repeat.
    inv = tickdist_full(Inverter(reconv_tree()), 's')
    t.eq('inverter.relabels_only', _ivl_map(inv),
         {(FAIL, 's'): (F(1, 16), F(1, 4)), (SUC, 's'): (F(5, 8), F(1))},
         'Inverter swaps status tags and leaves intervals untouched')
    rep = tickdist_full(Repeat(IntervalCheck('a', F(1, 4), F(1, 2)), 2), 's')
    t.eq('repeat.unrolls_to_sequence', _ivl_map(rep),
         {(SUC, 's'): (F(1, 16), F(1, 4)), (FAIL, 's'): (F(5, 8), F(1))},
         'Repeat(n,2) == Sequence(n,n); note the SAME leaf id appears twice, '
         'so the interval product decouples the two occurrences')
    t.eq('repeat.dependency_loss', rep.terms()[(FAIL, 's')],
         ((('a', -1),), (('a', +1), ('a', -1))),
         'monomial a.(1-a): interval product gives [1/8,3/8] whereas the true '
         'range of p(1-p) on [1/4,1/2] is [3/16,1/4]')

    # ---------------------------------------------------------------- 4 ----
    # Point mode == interval mode with degenerate boxes, and rows sum to 1.
    pi = {'a': F(1, 3), 'b': F(2, 5)}
    pm = tickdist(reconv_tree(), 's', leafvals=pi)
    want = {(SUC, 's'): (F(1, 3) * F(2, 5), F(1, 3) * F(2, 5)),
            (FAIL, 's'): (1 - F(1, 3) * F(2, 5), 1 - F(1, 3) * F(2, 5))}
    t.eq('point.exact', _ivl_map(pm), {k: (v[0], v[1]) for k, v in want.items()},
         'point mode returns the DTMC row of M_pi')
    t.eq('point.rows_sum_to_one', sum(v[0] for v in pm.values()), F(1),
         'exact stochasticity')
    # polynomial evaluation agrees with the recursion
    sdp = tickdist_full(reconv_tree(), 's', leafvals=pi)
    t.eq('point.terms_agree',
         {k: terms_value(v, pi) for k, v in sdp.terms().items()},
         {k: v[0] for k, v in _ivl_map(sdp).items()},
         'edge_value(factors) reproduces the recursion exactly')

    # ---------------------------------------------------------------- 5 ----
    # Well-formedness of the returned map.
    wf = {}
    for name, tr, s in (('fig1@s0', fig1_tree(), 0),
                        ('reconv', reconv_tree(), 's'),
                        ('selector', selector_tree(), 's'),
                        ('projection', projection_tree(), 0),
                        ('karity', karity_tree(), 'x')):
        sd = tickdist_full(tr, s)
        wf[name] = sd.check(strict=False)
        t.true('wellformed.%s' % name, wf[name]['ok'], wf[name],
               'sum(lo) <= 1 <= sum(hi) on the tagged map and its projection')
    res['wellformed'] = wf

    # ---------------------------------------------------------------- 6 ----
    # Enclosure (Theorem 1, step (i)) by randomized cross-check.
    rng = random.Random(20260809)
    trees = [('fig1', fig1_tree(), [0, 1, 2]),
             ('reconv', reconv_tree(), ['s']),
             ('selector', selector_tree(), ['s']),
             ('projection', projection_tree(), [0, 1]),
             ('karity', karity_tree(), ['x'])]
    encl = {'checked': 0, 'violations': []}
    for nm, tr, sts in trees:
        for _ in range(200):
            pi = sample_assignment(tr, rng, exact=True, denom=200)
            for s in sts:
                ivl = tickdist(tr, s, seq_mode='iterative')
                pnt = tickdist(tr, s, leafvals=pi)
                tot = 0
                for k, p in pnt.items():
                    tot += p[0]
                    a, b = ivl.get(k, (0, 0))
                    encl['checked'] += 1
                    if not (a <= p[0] <= b):
                        encl['violations'].append(
                            {'tree': nm, 'state': _st(s), 'key': str(k),
                             'p': float(p[0]), 'lo': float(a), 'hi': float(b)})
                if tot != 1:
                    encl['violations'].append({'tree': nm, 'state': _st(s),
                                               'row_sum': float(tot)})
    t.true('enclosure.random_leaf_assignments', not encl['violations'],
           {'checked': encl['checked'],
            'violations': encl['violations'][:5]},
           'every pi in Pi(T) lands inside TickDist(N,s) -- Thm 1 step (i)')
    res['enclosure'] = {'checked': encl['checked'],
                        'violations': encl['violations']}

    # ---------------------------------------------------------------- 7 ----
    # End-to-end: compile the IMC and reproduce Fig. 1(c)/(d) by robust VI.
    imc = compile_imc(fig1_tree(), 0)
    t.eq('imc.states', [_st(s) for s in imc.states], [0, 1, 2],
         'BFS reaches exactly {s0,s1,s2}')
    Vlo, tab_lo = robust_vi(imc, [2], 4, lower=True)
    Vhi, tab_hi = robust_vi(imc, [2], 4, lower=False)
    t.eq('vi.lower_H4_s0', Vlo[0], F(76517, 80000),
         'Fig. 1(c): lower V_4(s0) = .9565 (exact 76517/80000 = 0.9564625)')
    t.eq('vi.upper_H4_s0', Vhi[0], F(79857, 80000),
         'Fig. 1(c): upper V_4(s0) = .9982 (exact 79857/80000 = 0.9982125)')
    t.eq('vi.table_lower_s0', [float(r[0]) for r in tab_lo],
         [0.0, 0.0, 0.5950, 0.86275, 0.9564625], 'Fig. 1(c) column V_k(s0)')
    t.eq('vi.table_lower_s1', [float(r[1]) for r in tab_lo],
         [0.0, 0.85, 0.9775, 0.996625, 0.99949375], 'Fig. 1(c) column V_k(s1)')
    # the whole 5x4 table printed in Fig. 1(c), cell by cell
    printed = [[0.0, 0.0, 0.0, 0.0],
               [0.0, 0.0, 0.85, 0.95],
               [0.595, 0.855, 0.9775, 0.9975],
               [0.8628, 0.9833, 0.9966, 0.9999],
               [0.9565, 0.9982, 0.9995, 0.9999]]
    computed = [[float(tab_lo[k][0]), float(tab_hi[k][0]),
                 float(tab_lo[k][1]), float(tab_hi[k][1])]
                for k in range(5)]
    dev = [[abs(computed[i][j] - printed[i][j]) for j in range(4)]
           for i in range(5)]
    worst = max(max(r) for r in dev)
    t.true('vi.table_matches_letter_to_4dp', worst <= 1e-4,
           {'printed': printed, 'computed': computed, 'max_abs_dev': worst},
           'every cell of the Fig. 1(c) table is reproduced to its printed '
           'precision')
    mismatch = [{'row_k': i, 'col': ['V_lo(s0)', 'V_hi(s0)', 'V_lo(s1)',
                                     'V_hi(s1)'][j],
                 'printed': printed[i][j], 'computed': computed[i][j],
                 'computed_rounded_4dp': round(computed[i][j], 4)}
                for i in range(5) for j in range(4)
                if round(computed[i][j], 4) != printed[i][j]]
    res['fig1_table_cells_not_matching_printed_rounding'] = mismatch
    mid = {'deliver': F(90, 100), 'pick': F(80, 100)}
    Vm, _ = point_vi(imc, mid, [2], 4)
    t.true('vi.midpoint_inside', Vlo[0] <= Vm[0] <= Vhi[0],
           {'lo': float(Vlo[0]), 'mid': float(Vm[0]), 'hi': float(Vhi[0])},
           'Fig. 1(d): the midpoint DTMC number sits inside the interval')
    res['fig1_vi'] = {
        'states': [_st(s) for s in imc.states],
        'lower_table': [[float(v) for v in r] for r in tab_lo],
        'upper_table': [[float(v) for v in r] for r in tab_hi],
        'V_lower_s0': float(Vlo[0]), 'V_upper_s0': float(Vhi[0]),
        'V_lower_s0_exact': _exact(Vlo[0]), 'V_upper_s0_exact': _exact(Vhi[0]),
        'midpoint_dtmc_s0': float(Vm[0]),
        'letter_claims': {'V_lower_s0': 0.9565, 'V_upper_s0': 0.9982,
                          'midpoint': 0.9864},
        'tickdist_calls': imc.calls, 'seconds': imc.seconds,
    }
    t.true('vi.letter_rounding',
           round(float(Vlo[0]), 4) == 0.9565 and
           round(float(Vhi[0]), 4) == 0.9982,
           {'lo': float(Vlo[0]), 'hi': float(Vhi[0])},
           'matches the numbers printed in Fig. 1(c)/(d)')
    t.true('vi.letter_midpoint_0.9864',
           round(float(Vm[0]), 4) == 0.9864, {'mid': float(Vm[0])},
           'Fig. 1(d) says the midpoint DTMC reports 0.9864')

    # ---------------------------------------------------------------- 8 ----
    # SPEC DISCREPANCY: Alg. 1's iterative Extend loop vs Sec. III's
    # "extend by Sequence(n_2..n_k)".  They differ whenever the cap binds.
    kt = karity_tree()
    it = _ivl_map(tickdist_full(kt, 'x', seq_mode='iterative'))
    bi = _ivl_map(tickdist_full(kt, 'x', seq_mode='binary'))
    t.true('spec.karity_forms_differ', it != bi,
           {'iterative': _js(it), 'binary': _js(bi)},
           'Alg. 1 (iterative Extend) and Sec. III (binary Extend by the '
           'tail Sequence) are NOT the same map when the cap binds')
    t.eq('spec.karity_iterative', it[(FAIL, 'a')], (F(7, 25), F(18, 25)),
         'iterative: (Fail,a) = [.28,.72]')
    t.eq('spec.karity_binary', bi[(FAIL, 'a')], (F(7, 25), F(1, 2)),
         'binary: (Fail,a) = [.28,.50] -- tighter, both sound')
    res['spec_karity'] = {'iterative': _js(it), 'binary': _js(bi)}

    # both remain sound
    bad = []
    for _ in range(400):
        pi = sample_assignment(kt, rng, exact=True, denom=200)
        pnt = tickdist(kt, 'x', leafvals=pi)
        for k, p in pnt.items():
            for nm, mp in (('iterative', it), ('binary', bi)):
                a, b = mp.get(k, (0, 0))
                if not (a <= p[0] <= b):
                    bad.append({'mode': nm, 'key': str(k), 'p': float(p[0])})
    t.true('spec.karity_both_sound', not bad, bad[:5],
           'both readings enclose every leaf assignment')

    # ---------------------------------------------------------------- 9 ----
    # SPEC NOTE: the capped interval and the factorized polynomial disagree.
    sd = tickdist_full(reconv_tree(), 's')
    poly = terms_interval(sd.terms()[(FAIL, 's')], sd.box, cap=False)
    t.true('spec.cap_vs_polynomial',
           sd.intervals()[(FAIL, 's')] != poly,
           {'capped': _js(sd.intervals()[(FAIL, 's')]), 'polynomial': _js(poly)},
           'the cap makes D(s) NOT the interval hull of the multi-affine '
           'polynomial of Remark rem:pla; it is that hull intersected with '
           '[0,1] (sound, and strictly tighter here)')
    t.true('spec.cap_lower_never_binds',
           all(tickdist_full(tr, s).intervals()[k][0] ==
               terms_interval(tickdist_full(tr, s).terms()[k],
                              leaf_boxes(tr), cap=False)[0]
               for tr, s in ((fig1_tree(), 0), (reconv_tree(), 's'),
                             (selector_tree(), 's'), (projection_tree(), 0))
               for k in tickdist_full(tr, s).intervals()),
           None, 'lower bounds are cap-free, so lo == polynomial lo exactly')

    # --------------------------------------------------------------- 10 ----
    # SPEC NOTE: a k-branch ipact whose box is not projection tight makes
    # D(s) strictly larger than the achievable set (still sound).
    wf_k = check_wellformed(karity_tree(), strict=False)
    t.true('spec.non_tight_ipact', wf_k['non_tight'] == ['h', 'k'],
           wf_k['non_tight'],
           'ipact boxes are copied verbatim; [.4,.8]+[.4,.8] is feasible only '
           'on [.4,.6], so the emitted interval is not the tight projection')
    res['leaf_wellformedness'] = {'fig1': check_wellformed(fig1_tree(),
                                                           strict=False),
                                  'karity': wf_k}

    # --------------------------------------------------------------- 11 ----
    # Extend really ticks at the outcome's state, not at s.
    move = Action('move', lambda s: s + 1)
    gate = Check('is1', lambda s: s == 1)
    t.eq('extend.state_threading', _ivl_map(tickdist(Sequence(move, gate), 0)),
         {(SUC, 1): (1, 1)},
         'the gate is evaluated at the successor state 1, not at 0')
    t.eq('extend.running_passthrough',
         _ivl_map(tickdist(Sequence(Action('r', lambda s: s, RUN), move), 0)),
         {(RUN, 0): (1, 1)},
         'a Running outcome is neither Suc nor Fail, so Sequence stops')

    # --------------------------------------------------------------- 12 ----
    # Intra-tick scratch variables (what case4_leg.py's per-tick flags need).
    sc = tickdist_full(scratch_tree(), (0, 0))
    t.eq('scratch.tick', {(k[0], k[1]): v for k, v in sc.intervals().items()},
         {(SUC, (1, 0)): (F(3, 5), F(4, 5)),
          (SUC, (-1, 0)): (F(1, 5), F(2, 5))},
         'a flag written by one subtree and read by a later one inside the '
         'same tick; the final action projects the scratch field away')
    t.true('scratch.rowsum', sc.row_sums(imc=True)[0] <= 1 <=
           sc.row_sums(imc=True)[1], _js(sc.row_sums(imc=True)), '')

    # --------------------------------------------------------------- 13 ----
    # COMPLEXITY: cost is the number of outcome traces, not the leaf count.
    cost = []
    for k in range(1, 9):
        sd = tickdist_full(branchy_tree(k), ())
        cost.append({'arity_k': k, 'leaves': k, 'outcomes': len(sd.dist),
                     'monomials': sum(len(e.terms) for e in sd.dist.values()),
                     'tickdist_calls': sd.calls})
    res['complexity_branchy_sequence'] = cost
    t.eq('complexity.paths_are_exponential',
         [c['outcomes'] for c in cost], [2 ** c['arity_k'] for c in cost],
         'a Sequence of k children that each continue on 2 branches has 2^k '
         'outcomes, NOT O(#non-Fail leaves)=O(k): both files now carry the '
         'corrected O(|paths(T)|) sentence with this 2^k caveat')
    t.true('complexity.calls_track_paths',
           all(c['tickdist_calls'] >= 2 ** c['arity_k'] for c in cost),
           cost, '')

    # --------------------------------------------------------------- 14 ----
    # Negative tests: both well-formedness gates actually fire.
    bad_leaf = IntervalAction('bad', [Branch(SUC, lambda s: 1, F(3, 5), F(7, 10)),
                                      Branch(FAIL, lambda s: 0, F(3, 5),
                                             F(7, 10))], couple=False)
    try:
        check_wellformed(bad_leaf)
        fired = False
    except ValueError:
        fired = True
    t.true('negative.per_leaf_gate', fired, None,
           'sum plo = 1.2 > 1 is rejected (Pi(T) would be empty)')
    thin = IntervalAction('thin', [Branch(SUC, lambda s: 1, F(2, 5), F(3, 5))],
                          couple=False)
    try:
        tickdist(thin, 0)
        fired2 = False
    except ValueError:
        fired2 = True
    t.true('negative.row_sum_gate', fired2, None,
           'a returned map with sum(hi)=0.6 < 1 is rejected at the state level')

    # --------------------------------------------------------------- 15 ----
    # Float path agrees with the exact rational path on Fig. 1.
    ff = tickdist_full(fig1_tree(F=float), 0)
    t.true('float.matches_exact',
           all(abs(float(ff.imc()[k][i]) -
                   float(got_imc[0][k][i])) < 1e-12
               for k in got_imc[0] for i in (0, 1)),
           _js({k: [float(v[0]), float(v[1])] for k, v in ff.imc().items()}),
           'float and Fraction arithmetic agree to 1e-12')

    res['fig1_trace'] = per_state[0].trace
    res['spec_findings'] = [
        {'id': 'F1', 'where': 'Alg. 1 (extended) vs Sec. III prose (both files)',
         'issue': 'the k-ary Sequence/Selector is given twice and the two '
                  'readings are not the same function: Alg. 1 iterates '
                  'Extend(D,sigma*,n_i) child by child, the prose extends once '
                  'by the whole tail Sequence(n_2..n_k). The capped Minkowski '
                  'sum does not distribute over the interval product, so they '
                  'differ whenever the cap binds inside the tail.',
         'evidence': res['spec_karity'],
         'impact': 'both are sound; neither reading dominates. Implemented '
                   'Alg. 1 as the default (seq_mode="iterative") because it is '
                   'the printed algorithm and the one the Remark traces.'},
        {'id': 'F2', 'where': 'Extend, line "D\'(s\',s\'\') <- D\'(s\',s\'\') (+) I.J"',
         'issue': 'the key may be absent from D\'; the pseudocode never gives '
                  'the missing-key default. [0,0] is the only choice that '
                  'makes (+) associative with the identity.',
         'impact': 'cosmetic, but the algorithm is not executable as printed.'},
        {'id': 'F3', 'where': 'Alg. 1 if/elif chain',
         'issue': 'no case for Inverter, Repeat or stochastic leaves (they are '
                  'described only in the surrounding prose) and no final else, '
                  'so the printed function returns None on any decorator.',
         'impact': 'implemented from the prose; Repeat unrolls to a Sequence '
                   '(while-success). The justification sentence "ticking the '
                   'child until Fail at most n times" reads like a retry loop, '
                   'which would be a Selector; the Sequence unrolling is the '
                   'while-success reading. Both are provided (Repeat.mode).'},
        {'id': 'F4', 'where': 'cost claim, main_extended.tex',
         'issue': '|paths(T)| is not the leaf count; an earlier extended '
                  'draft said "the number of non-Fail leaves of T". Both '
                  'files now carry the corrected O(|paths(T)|) sentence '
                  'with the 2^k caveat.',
         'evidence': res['complexity_branchy_sequence'],
         'impact': 'measured 2^k outcomes for a Sequence of k two-Suc-branch '
                   'children (k leaves).'},
        {'id': 'F5', 'where': 'Remark rem:pla ("multi-affine polynomials '
                              'TickDist computes")',
         'issue': 'with the cap, D(s) is NOT the interval hull of those '
                  'polynomials; it is that hull intersected with [0,1]. The '
                  'cap fires on ordinary trees (any re-convergence of two '
                  'interval branches).',
         'evidence': {'reconv_(Fail,s)_capped': [0.625, 1.0],
                      'reconv_(Fail,s)_polynomial_hull': [0.625, 1.125]},
         'impact': 'sound and tighter; only the wording needs care. Lower '
                   'bounds are never capped, so lo is exactly the polynomial '
                   'lower endpoint (test spec.cap_lower_never_binds).'},
        {'id': 'F6', 'where': 'Def. ipact + Remark rem:pla tightness claim',
         'issue': 'TickDist copies [plo_i,phigh_i] verbatim, but the '
                  'calibration polytope {p in box, sum p = 1} may project to a '
                  'strictly smaller box per branch. Then D(s) is strictly '
                  'larger than the achievable set even for a single leaf '
                  'firing onto distinct successors, contradicting "the two '
                  'coincide when one interval leaf fires at s onto distinct '
                  'successors".',
         'evidence': {'declared': [[0.4, 0.8], [0.4, 0.8]],
                      'projection_tight': [[0.4, 0.6], [0.4, 0.6]]},
         'impact': 'sound. check_wellformed() reports projection_tight per '
                   'leaf; every leaf of Fig. 1 and of case 3 is tight, so no '
                   'published number is affected.'},
        {'id': 'F7', 'where': 'flat state space Sigma',
         'issue': 'a tick that needs a flag written by one subtree and read by '
                  'a later one (case4_leg.py uses several) has nowhere to put '
                  'it: Sigma is the post-tick state space.',
         'impact': 'no code change needed -- run the tick on (s, scratch) and '
                   'project with a final deterministic action (test '
                   'scratch.tick) -- but the letter should say so.'},
        {'id': 'F8', 'where': 'Fig. 1(c) table',
         'issue': 'two printed cells do not equal the computed value rounded '
                  'to 4 dp.',
         'evidence': res['fig1_table_cells_not_matching_printed_rounding'],
         'impact': 'V_hi_3(s0) = 0.98325 printed as .9833 (round-half-up vs '
                   'round-half-even) and V_hi_4(s1) = 0.99999375 printed as '
                   '.9999 (should be 1.0000). The headline numbers .9565 / '
                   '.9982 / .9864 are all exactly reproduced.'},
    ]
    res['tests'] = t.rows
    res['n_tests'] = len(t.rows)
    res['n_failed'] = t.failed
    if verbose:
        print('%d checks, %d failed' % (len(t.rows), t.failed))
    return res


# ==================================================================== demo ===
def demo() -> None:
    tree = fig1_tree()
    print('IBT of Fig. 1: %r\n' % (tree,))
    print('per-leaf well-formedness:')
    for row in check_wellformed(tree)['leaves']:
        print('  %-8s boxes=%s  sum_lo=%.2f <= 1 <= sum_hi=%.2f  tight=%s'
              % (row['leaf'], row['boxes'], row['sum_lo'], row['sum_hi'],
                 row['projection_tight']))
    print()
    for s in (0, 1, 2):
        sd = tickdist_full(tree, s)
        print('TickDist(root, s%d) = %r' % (s, sd))
        print('   IMC edges : %s' % ', '.join(
            's%d->s%s [%s,%s]' % (s, _st(t2), _fmt(v[0]), _fmt(v[1]))
            for t2, v in sd.imc().items()))
        print('   factorized: %s' % ', '.join(
            's%s x %s' % (_st(t2), m if m else '1')
            for t2, m in sd.imc_edges()))
        lo, hi = sd.row_sums()
        print('   row sums  : lo=%s <= 1 <= hi=%s' % (_fmt(lo), _fmt(hi)))
    imc = compile_imc(tree, 0)
    Vlo, tab = robust_vi(imc, [2], 4, True)
    Vhi, _ = robust_vi(imc, [2], 4, False)
    print('\ncompiled IMC: %d states, %d TickDist calls, %.4f ms'
          % (imc.n(), imc.calls, 1000 * imc.seconds))
    print('robust VI, H=4, F s2 :  [%.4f, %.4f]   (Fig. 1(d): [0.9565,0.9982])'
          % (float(Vlo[0]), float(Vhi[0])))


# ==================================================================== main ===
def _write_json(res: Dict[str, Any], path: str) -> str:
    res['meta'] = {
        'generated': time.strftime('%Y-%m-%dT%H:%M:%S'),
        'python': sys.version.split()[0],
        'platform': platform.platform(),
        'module': os.path.abspath(__file__),
        'arithmetic': 'fractions.Fraction (exact) unless stated otherwise',
    }
    with open(path, 'w') as fh:
        json.dump(res, fh, indent=1, sort_keys=False, default=str)
    return path


def main(argv: List[str]) -> int:
    cmd = argv[1] if len(argv) > 1 else 'test'
    # Writing next to an installed module is wrong; default to the cwd and let
    # a caller override with an explicit path.
    out = (argv[2] if len(argv) > 2
           else os.path.join(os.getcwd(), 'results_tickdist.json'))
    if cmd == 'demo':
        demo()
        return 0
    if cmd in ('test', 'json'):
        t0 = time.perf_counter()
        res = run_tests(verbose=(cmd == 'test'))
        res['seconds'] = time.perf_counter() - t0
        _write_json(res, out)
        print('wrote %s' % out)
        return 1 if res['n_failed'] else 0
    print(__doc__)
    return 2


if __name__ == '__main__':
    sys.exit(main(sys.argv))
