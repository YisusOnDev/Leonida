"""Share repeated instruction bodies of a translated module as SPIR-V functions (SPIRV-SIZE-3).

rdna2spirv.py emits every source instruction inline. Some expand to hundreds of SPIR-V instructions: a typed
buffer load (address, bounds, fetch, realign, swizzle) is about 160, `image_bvh_intersect_ray` about 2,700, and a
legacy multiply 20. A module with 3,281 typed loads is then 5 M words, and drivers take minutes to compile it.

This pass takes the SPIR-V lines each top-level instruction emitted (its span, recorded by Gen.insn) and turns
the span into a function body: the values it reads from %main (registers, %active, values an earlier instruction
computed) become parameters, the registers it writes and the values a later instruction reads become results. Two
spans whose bodies are the same after renaming (the same instruction on other registers, for instance) call one
function. Only spans that occur at least twice and are big enough to pay for the call are outlined.

The call is exact. Each span is one region of straight or structured code inside %main with one entry (the block
it starts in) and one exit (the block it ends in); the function runs the same instructions on the same values:

- A Function-storage pointer (a register, %active, %scc ...) can't be passed (logical addressing), so a span is
  taken only when it uses such a pointer as the pointer of OpLoad / OpStore and nothing else. Each first load
  becomes a parameter loaded at the call site; a load after a store to the same pointer reads the stored value.
  A store is taken only in the span's first or last block (code every path through the span runs), and the caller
  stores the result after the call. Only private Function memory moves; buffer, image, workgroup and atomic
  accesses stay in the body, in their order.
- A span that defines a block whose label a later OpPhi names (the instruction after it starts in that block)
  leaves that code in the caller's block, so the later name is renamed to the label the span started in.
- Not taken: spans with OpVariable, a terminator of %main (OpReturn, OpKill, OpTerminateInvocation, OpDemote...,
  OpUnreachable), a barrier, mesh or geometry output, a branch out of the span, a last line that ends a block,
  or a parameter / result that is not a scalar or vector of numbers or bools (pointers, images).

Results go back through a struct when there are several. The emitted text keeps its order and names outside the
spans, and the later passes (NoContraction, type dedupe) see the functions like any other code.
"""

from __future__ import annotations

from collections import Counter

# A span is considered when it is at least this many words; a call site costs about 4 + 1 per argument words plus
# 4-5 per result and per register load / store it moves to the caller.
MIN_SPAN_WORDS = 40
MAX_PARAMS = 64
_TERMINATORS = ("OpBranch", "OpBranchConditional", "OpSwitch", "OpReturn", "OpReturnValue", "OpKill",
                "OpUnreachable", "OpTerminateInvocation")
_REFUSE = frozenset(("OpVariable", "OpReturn", "OpReturnValue", "OpKill", "OpUnreachable", "OpTerminateInvocation",
                     "OpDemoteToHelperInvocation", "OpDemoteToHelperInvocationEXT", "OpControlBarrier",
                     "OpEmitVertex", "OpEndPrimitive", "OpEmitStreamVertex", "OpEndStreamPrimitive",
                     "OpSetMeshOutputsEXT", "OpEmitMeshTasksEXT", "OpBeginInvocationInterlockEXT",
                     "OpEndInvocationInterlockEXT", "OpFunction", "OpFunctionEnd", "OpFunctionParameter",
                     "OpLifetimeStart", "OpLifetimeStop", "OpCopyMemory", "OpCopyMemorySized"))


def _words(tokens):
    """Approximate words of one instruction: every token but the `=` (literals of more than a word are rare)."""
    return len(tokens) - (1 if len(tokens) > 1 and tokens[1] == "=" else 0)


class _Refuse(Exception):
    pass


def value_types(types_lines):
    """Type ids a parameter or result may have: ints, floats, bools and vectors of them."""
    ok = set()
    for line in types_lines:
        t = line.split()
        if len(t) >= 3 and t[1] == "=":
            if t[2] in ("OpTypeInt", "OpTypeFloat", "OpTypeBool"):
                ok.add(t[0])
            elif t[2] == "OpTypeVector" and len(t) >= 4 and t[3] in ok:
                ok.add(t[0])
    return ok


def function_pointer_types(types_lines):
    return {t[0] for t in (l.split() for l in types_lines)
            if len(t) >= 4 and t[1] == "=" and t[2] == "OpTypePointer" and t[3] == "Function"}


class Outliner:
    def __init__(self, main_lines, types_lines, prefix="%of"):
        """main_lines: every line of %main known before placement (the entry prologue and all block bodies), to
        find the type of each %main id and how often each id is named."""
        self.vtypes = value_types(types_lines)
        self.fptr = function_pointer_types(types_lines)
        self.deftype = {}
        self.labels = set()
        self.uses = Counter()
        for line in main_lines:
            t = line.split()
            if len(t) >= 3 and t[1] == "=":
                if t[2] == "OpLabel":
                    self.labels.add(t[0])
                elif len(t) >= 4:
                    self.deftype[t[0]] = t[3]
            self.uses.update(x for x in t if x.startswith("%"))
        self.prefix = prefix
        # Type identity: dedupe_types later merges equal vectors declared twice (%v4uint, %v4uint_fmt), and SPIR-V
        # forbids two equal OpTypeFunction, so signatures are compared on canonical types and an existing
        # OpTypeFunction of the module is reused.
        self._canon = {}
        for line in types_lines:
            t = line.split()
            if len(t) >= 3 and t[1] == "=":
                if t[2] == "OpTypeVector" and len(t) >= 5:
                    self._canon[t[0]] = ("vec", self._canon.get(t[3], t[3]), t[4])
        self.functions = {}      # canonical key -> (name, fn type, return type, function lines)
        self.type_lines = []
        self._struct = {}
        self._fntype = {}
        for line in types_lines:
            t = line.split()
            if len(t) >= 4 and t[1] == "=" and t[2] == "OpTypeFunction":
                self._fntype.setdefault(tuple(self._canon.get(x, x) for x in t[3:]), t[0])
        self.outlined = 0
        self.calls = Counter()

    # ---- analysis of one span ----
    def analyze(self, lines, entry):
        """The function a span would become, or None. lines: the span's lines (stripped); entry: the label of the
        block the span starts in."""
        toks = [l.split() for l in lines]
        if not toks or sum(_words(t) for t in toks) < MIN_SPAN_WORDS:
            return None
        try:
            return self._analyze(toks, entry)
        except _Refuse:
            return None

    def _analyze(self, toks, entry):
        last = toks[-1]
        op_last = last[2] if len(last) > 2 and last[1] == "=" else last[0]
        if op_last in _TERMINATORS or op_last in ("OpSelectionMerge", "OpLoopMerge") or op_last == "OpLabel":
            raise _Refuse("ends block")
        first = toks[0]
        if len(first) > 2 and first[1] == "=" and first[2] == "OpLabel":
            raise _Refuse("starts with label")
        defined = {}
        labels = []
        for k, t in enumerate(toks):
            if len(t) >= 3 and t[1] == "=":
                if t[2] == "OpLabel":
                    labels.append(t[0])
                defined[t[0]] = k
        last_label_at = max((defined[l] for l in labels), default=-1)
        first_label_at = min((defined[l] for l in labels), default=len(toks))
        in_span = Counter(x for t in toks for x in t if x.startswith("%"))
        body = []
        params = []                 # (outer id or ("load", ptr), type)
        param_of = {}               # outer value id -> canonical param token
        load_param = {}             # ptr -> current value token inside the body (param or forwarded store value)
        stored = {}                 # ptr -> value token (inside the body) of its last store
        store_order = []
        sub = {}                    # body token -> replacement (forwarded loads)
        entry_used = False

        def ref(x):
            nonlocal entry_used
            if x in sub:
                return sub[x]
            if x in defined:
                return x
            if x in self.labels:
                if x != entry:
                    raise _Refuse("external label")
                entry_used = True
                return "%@E"
            ty = self.deftype.get(x)
            if ty is None:
                return x                        # module scope: constant, type, global variable, function
            if ty in self.fptr:
                raise _Refuse("fptr use")                   # a Function pointer anywhere but OpLoad / OpStore's pointer
            if ty not in self.vtypes:
                raise _Refuse("param type")
            if x not in param_of:
                if len(params) >= MAX_PARAMS:
                    raise _Refuse("too many params")
                param_of[x] = f"%@P{len(params)}"
                params.append((x, ty))
            return param_of[x]

        for k, t in enumerate(toks):
            has_res = len(t) >= 3 and t[1] == "="
            op = t[2] if has_res else t[0]
            if op in _REFUSE:
                raise _Refuse("forbidden op")
            if op == "OpLoad" and has_res and len(t) >= 5 and self.deftype.get(t[4]) in self.fptr:
                ptr = t[4]
                if ptr in stored:
                    sub[t[0]] = stored[ptr]
                elif ptr in load_param:
                    sub[t[0]] = load_param[ptr]
                else:
                    if t[3] not in self.vtypes or len(params) >= MAX_PARAMS:
                        raise _Refuse("load type")
                    tok = f"%@P{len(params)}"
                    params.append((("load", ptr), t[3]))
                    load_param[ptr] = tok
                    sub[t[0]] = tok
                continue
            if op == "OpStore" and len(t) >= 3 and self.deftype.get(t[1]) in self.fptr:
                if first_label_at < k < last_label_at:
                    raise _Refuse("conditional store")      # only the first and the last block run on every path               # a store under a condition of the span
                ptr = t[1]
                val = ref(t[2])
                if ptr not in stored:
                    store_order.append(ptr)
                stored[ptr] = val
                continue
            if op in ("OpBranch", "OpBranchConditional", "OpSwitch", "OpSelectionMerge", "OpLoopMerge"):
                for x in t[1:]:
                    if x.startswith("%") and x in self.labels and x not in defined:
                        raise _Refuse("branch out")
            out = []
            for i, x in enumerate(t):
                if not x.startswith("%") or (has_res and i == 0):
                    out.append(x)
                elif has_res and i == 3 and op != "OpLabel":
                    out.append(x)               # the result type
                else:
                    out.append(ref(x))
            body.append(out)
        # Results: stored registers (unless the value is outer or constant), then span ids read after the span.
        results = []                # (body token, type, ("store", ptr) | ("def", outer id))
        call_stores = []            # (ptr, outer value) stores the caller makes without a result
        for ptr in store_order:
            v = stored[ptr]
            if v.startswith("%@P"):
                call_stores.append((ptr, ("arg", int(v[3:]))))     # a parameter: the caller stores its argument
            elif v in defined:
                results.append((v, self._def_type(toks, defined[v]), ("store", ptr)))
            else:
                call_stores.append((ptr, v))
        label_out = None
        for x, k in defined.items():
            if self.uses[x] > in_span[x]:
                t = toks[k]
                if t[2] == "OpLabel":
                    if k != last_label_at:
                        raise _Refuse("label escapes")
                    label_out = x           # the block the span ends in: the caller's block afterwards
                    continue
                if x in sub:                # a register load read after the span: what the load gave
                    v = sub[x]
                    if v.startswith("%@P"):
                        call_stores.append((x, ("copy", int(v[3:]), t[3])))
                    else:
                        results.append((v, t[3], ("def", x)))
                    continue
                results.append((x, self._def_type(toks, k), ("def", x)))
        for _, ty, _ in results:
            if ty not in self.vtypes:
                raise _Refuse("result type")
        if len(results) > MAX_PARAMS:
            raise _Refuse("too many results")
        # Canonical names: body ids in order of definition.
        canon = {}
        n = 0
        for t in body:
            if len(t) >= 3 and t[1] == "=":
                canon[t[0]] = f"%@{n}"
                n += 1

        def cn(x):
            return canon.get(x, x)
        cbody = [" ".join(cn(x) for x in t) for t in body]
        rets = [cn(r[0]) for r in results]
        key = (tuple(ty for _, ty in params), tuple(r[1] for r in results), tuple(rets), tuple(cbody))
        words = sum(_words(t) for t in body)
        return {"key": key, "params": params, "results": results, "call_stores": call_stores,
                "label_out": label_out, "words": words, "entry_used": entry_used}

    @staticmethod
    def _def_type(toks, k):
        t = toks[k]
        return t[3] if len(t) >= 4 else None

    # ---- emission ----
    def _type_of_results(self, results):
        if not results:
            return "%void"
        if len(results) == 1:
            return results[0][1]
        members = tuple(r[1] for r in results)
        key = tuple(self._canon.get(m, m) for m in members)
        if key not in self._struct:
            name = f"{self.prefix}_st{len(self._struct)}"
            self._struct[key] = name
            self.type_lines.append(f"{name} = OpTypeStruct " + " ".join(members))
        return self._struct[key]

    def _fn_type(self, ret, ptypes):
        sig = tuple(self._canon.get(x, x) for x in (ret,) + tuple(ptypes))
        if sig not in self._fntype:
            name = f"{self.prefix}_ft{len(self.type_lines)}"
            self._fntype[sig] = name
            self.type_lines.append(f"{name} = OpTypeFunction {ret}" + "".join(" " + p for p in ptypes))
        return self._fntype[sig]

    def function_for(self, info):
        key = info["key"]
        if key in self.functions:
            return self.functions[key]
        ptypes, _, rets, cbody = key
        k = len(self.functions)
        name = f"{self.prefix}{k}"
        ret = self._type_of_results(info["results"])
        ftype = self._fn_type(ret, ptypes)

        def inst(s):
            return (s.replace("%@E", f"{name}_e").replace("%@P", f"{name}_p").replace("%@", f"{name}_"))
        lines = [f"{name} = OpFunction {ret} None {ftype}"]
        lines += [f"{name}_p{i} = OpFunctionParameter {ty}" for i, ty in enumerate(ptypes)]
        lines.append(f"{name}_e = OpLabel")
        lines += [inst(s) for s in cbody]
        if ret == "%void":
            lines.append("OpReturn")
        elif len(rets) == 1:
            lines.append(f"OpReturnValue {inst(rets[0])}")
        else:
            lines.append(f"{name}_r = OpCompositeConstruct {ret} " + " ".join(inst(r) for r in rets))
            lines.append(f"OpReturnValue {name}_r")
        lines.append("OpFunctionEnd")
        self.functions[key] = (name, ftype, ret, lines)
        return self.functions[key]

    def call(self, info, new_id):
        """The caller's lines for one outlined span."""
        name, _, ret, _ = self.function_for(info)
        out, args = [], []
        for src, ty in info["params"]:
            if isinstance(src, tuple):
                v = new_id()
                out.append(f"{v} = OpLoad {ty} {src[1]}")
                args.append(v)
            else:
                args.append(src)
        results = info["results"]
        single_def = len(results) == 1 and results[0][2][0] == "def"
        r = results[0][2][1] if single_def else new_id()
        out.append(f"{r} = OpFunctionCall {ret} {name}" + "".join(" " + a for a in args))
        for i, (_, ty, (kind, what)) in enumerate(results):
            if len(results) == 1:
                v = r
            else:
                v = what if kind == "def" else new_id()
                out.append(f"{v} = OpCompositeExtract {ty} {r} {i}")
            if kind == "store":
                out.append(f"OpStore {what} {v}")
        for target, v in info["call_stores"]:
            if isinstance(v, tuple) and v[0] == "copy":      # target: an id read after the span
                out.append(f"{target} = OpCopyObject {v[2]} {args[v[1]]}")
            elif isinstance(v, tuple):
                out.append(f"OpStore {target} {args[v[1]]}")
            else:
                out.append(f"OpStore {target} {v}")
        self.calls[name] += 1
        return out


def outline_units(units, outliner, new_id):
    """units: list of (lines, spans, entry_label) for each region of %main (a basic block's lines, or the
    straight-line body). Returns the new lines of each unit. Two passes: count the bodies, then outline the ones
    that repeat and save words."""
    found = []
    counts = Counter()
    for lines, spans, entry in units:
        per = []
        label = entry
        pos = 0
        for s, e in spans:
            for l in lines[pos:s]:
                t = l.split()
                if len(t) >= 3 and t[1] == "=" and t[2] == "OpLabel":
                    label = t[0]
            info = outliner.analyze([l.strip() for l in lines[s:e]], label) if e > s else None
            per.append(info)
            if info is not None:
                counts[info["key"]] += 1
            for l in lines[s:e]:
                t = l.split()
                if len(t) >= 3 and t[1] == "=" and t[2] == "OpLabel":
                    label = t[0]
            pos = e
        found.append(per)

    def worth(info):
        n = counts[info["key"]]
        call = 5 + len(info["params"]) * 5 + len(info["results"]) * 8
        return n >= 2 and (n - 1) * info["words"] > n * call + 64

    result = []
    rename = {}
    for (lines, spans, entry), per in zip(units, found):
        out, pos = [], 0

        def put(l):
            if rename:
                t = l.split()
                if any(x in rename for x in t):
                    l = " ".join(rename.get(x, x) for x in t)
            out.append(l)
        cur_label = entry
        for (s, e), info in zip(spans, per):
            for l in lines[pos:s]:
                put(l)
                t = l.split()
                if len(t) >= 3 and t[1] == "=" and t[2] == "OpLabel":
                    cur_label = t[0]
            if info is not None and worth(info):
                for l in outliner.call(info, new_id):
                    put(l)
                outliner.outlined += 1
                if info["label_out"]:
                    rename[info["label_out"]] = cur_label
            else:
                for l in lines[s:e]:
                    put(l)
                    t = l.split()
                    if len(t) >= 3 and t[1] == "=" and t[2] == "OpLabel":
                        cur_label = t[0]
            pos = e
        for l in lines[pos:]:
            put(l)
        result.append(out)
    if rename:      # a label renamed after its unit was written (never seen, but labels are module-unique)
        for out in result:
            for i, l in enumerate(out):
                t = l.split()
                if any(x in rename for x in t):
                    out[i] = " ".join(rename.get(x, x) for x in t)
    return result
