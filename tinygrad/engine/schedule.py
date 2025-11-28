import time
from typing import cast
from dataclasses import dataclass, field, replace
from tinygrad.uop.ops import UOp, Ops, buffers, UOpMetaClass, print_uops
from tinygrad.uop.spec import type_verify, tensor_spec
from tinygrad.device import Buffer, MultiBuffer
from tinygrad.helpers import Metadata, DEBUG, cpu_profile, TracingKey, SPEC, flatten
from tinygrad.codegen.late.linearizer import linearize

# **** ScheduleItem return type

@dataclass(frozen=True)
class ScheduleItem:
  ast: UOp
  bufs: tuple[Buffer, ...]
  metadata: tuple[Metadata, ...] = ()
  fixedvars: dict[str, int] = field(default_factory=dict)
  bound_ranges: tuple[UOp, ...] = ()

# **** schedule linearizer

def create_schedule_with_vars(sched_sink:UOp) -> tuple[list[ScheduleItem], dict[str, int]]:
  with cpu_profile(TracingKey("linearize sched_sink")):
    var_vals: dict[str, int] = {}
    for u in sched_sink.toposort():
      if u.op is Ops.BIND and u.src[1].op is not Ops.RANGE:
        var, val = u.unbind()
        assert var.expr not in var_vals or var_vals[var.expr] == val, f"bind mismatch on {var}, {var_vals[var.expr]} != {val}"
        var_vals[var.expr] = val

    lst = linearize(sched_sink, tuple_order=False)
    if DEBUG >= 2: print_uops(lst)

  with cpu_profile(TracingKey("build schedule")):
    schedule: list[ScheduleItem|UOp] = []
    for k in lst:
      if k.op is Ops.KERNEL:
        ast = k.arg.ast
        if ast.op is Ops.BUFFER_VIEW:
          base = k.src[1].buf_uop.buffer
          assert isinstance(base, Buffer), "base can't be MultiBuffer"
          buffers[k.src[0]] = base.view(k.size, ast.dtype, ast.arg[1]*base.dtype.itemsize)
        ubufs = tuple(s.buf_uop.buffer for s in k.src if s.op is not Ops.BIND)
        bound_ranges = tuple(s for s in k.src if s.op is Ops.BIND and s.src[1].op is Ops.RANGE)
        if any(isinstance(x, MultiBuffer) for x in ubufs):
          assert all(isinstance(x, MultiBuffer) for x in ubufs), "kernel must all be multibuffer"
          dnums = [x for x in ast.variables() if x.arg[0] == '_device_num']
          for i,bufs in enumerate(zip(*[x.bufs for x in cast(tuple[MultiBuffer, ...], ubufs)])):
            schedule.append(ScheduleItem(ast, bufs, k.arg.metadata, {dnums[0].expr:i} if len(dnums) else {}, bound_ranges=bound_ranges))
        else:
          schedule.append(ScheduleItem(ast, cast(tuple[Buffer, ...], ubufs), k.arg.metadata, bound_ranges=bound_ranges))
      elif k.op in (Ops.RANGE, Ops.END):
        schedule.append(k)

  with cpu_profile(TracingKey("expand ranges")):
    real_schedule: list[ScheduleItem] = []
    sched_ptr = 0
    in_ranges = {}
    range_ptrs = {}
    while sched_ptr < len(schedule):
      si = schedule[sched_ptr]
      if isinstance(si, UOp):
        if si.op is Ops.RANGE:
          in_ranges[si] = 0
          range_ptrs[si] = sched_ptr + 1
        elif si.op is Ops.END:
          if in_ranges[si.src[1]] < si.src[1].vmax:
            in_ranges[si.src[1]] += 1
            sched_ptr = range_ptrs[si.src[1]]
            continue
      else:
        real_schedule.append(replace(si, fixedvars=si.fixedvars | {s.src[0].arg[0]:in_ranges[s.src[1]] for s in si.bound_ranges}, bound_ranges=()))
      sched_ptr += 1
  return real_schedule, var_vals

from tinygrad.engine.memory import memory_planner
from tinygrad.schedule.rangeify import get_rangeify_map
from tinygrad.schedule.multi import get_multi_map

def complete_create_schedule_with_vars(big_sink:UOp) -> tuple[dict[UOp, UOp], list[ScheduleItem], dict[str, int]]:
  # big_sink srcs are all the Tensors
  st = time.perf_counter()

  # verify Tensors match the spec
  if SPEC: type_verify(big_sink, tensor_spec)

  # tensor map is what we return
  tensor_map: dict[UOp, UOp] = {}

  if any(isinstance(x._device, tuple) for x in big_sink.toposort()):
    tensor_map |= get_multi_map(big_sink)
    big_sink = big_sink.substitute(tensor_map, name="Apply Multi Map")
    big_sink = UOp.sink(*flatten([x.src if x.op is Ops.MULTI else [x] for x in big_sink.src]))

  tensor_map |= get_rangeify_map(big_sink)
  big_sink = big_sink.substitute(tensor_map, name="Apply Kernelize Map")

  # create the schedule
  schedule, var_vals = create_schedule_with_vars(big_sink)
  with cpu_profile(TracingKey("memory planner")): schedule = memory_planner(schedule)

  # remove all AFTERs, after scheduling, the tensors are just buffers
  tensor_map |= {u:u.buf_uop for u in big_sink.toposort() if u.op is Ops.AFTER}

  if (DEBUG >= 1 and len(schedule) > 1) or DEBUG >= 3:
    print(f"scheduled {len(schedule)} kernels in {(time.perf_counter()-st)*1000:.2f} ms ({len(UOpMetaClass.ucache)} uops in cache)")
  return tensor_map, schedule, var_vals
