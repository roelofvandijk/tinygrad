from __future__ import annotations
from dataclasses import dataclass
import functools
from math import gcd
from itertools import product
from tinygrad.helpers import partition
from typing import List, Dict, Callable, Tuple, Type, Union, Optional, Any, Iterator

# NOTE: Python has different behavior for negative mod and floor div than c
# symbolic matches the Python behavior, but the code output is agnostic, and will never have negative numbers in div or mod

def is_sym_int(x: Any) -> bool: return isinstance(x, (int, Node))

class Node:
  a: Node
  b: Union[Node, int]
  min: int
  max: int
  def render(self, ops=None, ctx=None) -> str:
    if ops is None: ops = render_python
    assert self.__class__ in (Variable, NumNode) or self.min != self.max
    return ops[type(self)](self, ops, ctx)
  def vars(self): return []

  def expand_idx(self) -> VariableOrNum: return next((v for v in self.vars() if v.expr is None), NumNode(0))
  # expand a Node into List[Node] that enumerates the underlying Variables from min to max
  # expand increments earlier variables faster than later variables (as specified in the argument)
  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def expand(self, idxs:Optional[Tuple[VariableOrNum, ...]]=None) -> List[Node]:
    if idxs is None: idxs = (self.expand_idx(),)
    return [self.substitute(dict(zip(idxs, (NumNode(x) for x in rep)))) for rep in Node.iter_idxs(idxs)]
  @staticmethod
  def iter_idxs(idxs:Tuple[VariableOrNum, ...]) -> Iterator[Tuple[int,...]]:
    yield from (x[::-1] for x in product(*[[x for x in range(v.min, v.max + 1)] for v in reversed(idxs)]))
  # substitute Variables with the values in var_vals
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: raise RuntimeError(self.__class__.__name__)

  @functools.cached_property
  def key(self) -> str: return self.render(ctx="DEBUG")
  def __bool__(self): return not (self.__class__ is NumNode and self.b == 0)
  def __repr__(self): return "<"+self.key+">"
  def __neg__(self): return self*-1
  @functools.lru_cache(None)
  def __add__(self, b:Union[Node,int]): return Variable.sum([self, b if isinstance(b, Node) else Variable.num(b)])
  @functools.lru_cache(None)
  def __radd__(self, b:int): return self+b
  @functools.lru_cache(None)
  def __sub__(self, b:Union[Node,int]): return self+-b
  @functools.lru_cache(None)
  def __rsub__(self, b:int): return -self+b
  @functools.lru_cache(None)
  def __le__(self, b:Union[Node,int]): return self < (b+1)
  @functools.lru_cache(None)
  def __gt__(self, b:Union[Node,int]): return (-self) < (-b)
  @functools.lru_cache(None)
  def __ge__(self, b:Union[Node,int]): return (-self) < (-b+1)
  @functools.lru_cache(None)
  def __lt__(self, b:Union[Node,int]): return LtNode.create(self, b)
  @functools.lru_cache(None)
  def __mul__(self, b:Union[Node, int]):
    if b == 0: return NumNode(0)
    if b == 1: return self
    if self.__class__ is NumNode: return NumNode(self.b*b) if isinstance(b, int) else b*self.b
    return MulNode.create(self, b.b if isinstance(b, NumNode) else b)
  @functools.lru_cache(None)
  def __rmul__(self, b:int): return self*b

  # *** complex ops ***
  @functools.lru_cache(None)
  def __rfloordiv__(self, b:int):
    if self.min > b >= 0: return NumNode(0)
    if isinstance(self, NumNode): return NumNode(b // self.b)
    raise RuntimeError(f"not supported: {b} // {self}")
  @functools.lru_cache(None)
  def __floordiv__(self, b:Union[Node,int], factoring_allowed=True):
    if isinstance(b, Node):
      if b.__class__ is NumNode: return self // b.b
      if self == b: return NumNode(1)
      if (b - self).min > 0 and self.min >= 0: return NumNode(0) # b - self simplifies the node
      raise RuntimeError(f"not supported: {self} // {b}")
    assert b != 0
    if b < 0: return (self//-b)*-1
    if b == 1: return self

    # the numerator of div is not allowed to be negative
    if self.min < 0:
      offset = self.min//b
      # factor out an "offset" to make the numerator positive. don't allowing factoring again
      return (self + -offset*b).__floordiv__(b, factoring_allowed=False) + offset
    return DivNode.create(self, b)
  @functools.lru_cache(None)  
  def __rmod__(self, b:int):
    if self.min > b >= 0: return NumNode(b)
    if isinstance(self, NumNode): return NumNode(b % self.b)
    raise RuntimeError(f"not supported: {b} % {self}")
  @functools.lru_cache(None) 
  def __mod__(self, b:Union[Node,int]):
    if isinstance(b, Node):
      if b.__class__ is NumNode: return self % b.b
      if self == b: return NumNode(0)
      if (b - self).min > 0 and self.min >= 0: return self # b - self simplifies the node
      raise RuntimeError(f"not supported: {self} % {b}")
    assert b > 0
    if b == 1: return NumNode(0)
    if self.min >= 0 and self.max < b: return self
    if (self.min//b) == (self.max//b): return self - (b*(self.min//b))
    if self.min < 0: return (self - ((self.min//b)*b)) % b
    return ModNode.create(self, b)

  @staticmethod
  @functools.lru_cache(None)
  def num(num:int) -> NumNode: return NumNode(num)

  @staticmethod
  def factorize(nodes:List[Node]) -> List[Node]:
    mul_groups: Dict[Node, int] = {}
    for x in nodes:
      a,b = (x.a,x.b) if isinstance(x, MulNode) else (x,1)
      mul_groups[a] = mul_groups.get(a, 0) + b
    return [MulNode.create(a, b_sum) if b_sum != 1 else a for a, b_sum in mul_groups.items() if b_sum != 0]

  @staticmethod
  def sum(nodes:List[Node]) -> Node: 
    nodes = [x for x in nodes if x]
    if not nodes: return NumNode(0)
    if len(nodes) == 1: return nodes[0]
    return Node._sum(tuple(nodes))

  @staticmethod
  @functools.lru_cache(None)
  def _sum(nodes:List[Node]) -> Node:
    new_nodes: List[Node] = []
    num_node_sum = 0
    if (sn:=create_rednode(SumNode, nodes)).__class__ is NumNode: return sn
    for node in sn.flat_components:
      if node.__class__ is NumNode: num_node_sum += node.b
      else: new_nodes.append(node)

    if len(new_nodes) > 1 and len(set([x.a if isinstance(x, MulNode) else x for x in new_nodes])) < len(new_nodes):
      new_nodes = Node.factorize(new_nodes)
    if num_node_sum: new_nodes.append(NumNode(num_node_sum))
    return create_rednode(SumNode, new_nodes) if len(new_nodes) > 1 else new_nodes[0] if len(new_nodes) == 1 else NumNode(0)

  @staticmethod
  def ands(nodes:List[Node]) -> Node: 
    if not nodes: return NumNode(1)
    if len(nodes) == 1: return nodes[0]
    if any(not x for x in nodes): return NumNode(0)
    nodes = [x for x in nodes if x.min != x.max]
    return Node._ands(tuple(nodes))

  @staticmethod
  @functools.lru_cache(None)
  def _ands(nodes:List[Node]) -> Node:

    # filter 1s
    nodes = [x for x in nodes if x.min != x.max]
    return create_rednode(AndNode, nodes) if len(nodes) > 1 else (nodes[0] if len(nodes) == 1 else NumNode(1))

# 4 basic node types

@dataclass(frozen=True, repr=False)
class Variable(Node):
  def __new__(cls, expr:Optional[str], nmin:int, nmax:int):
    assert nmin >= 0 and nmin <= nmax
    if nmin == nmax: return NumNode(nmin)
    return super().__new__(cls)
  expr: Optional[str]
  min: int
  max: int
  def vars(self): return [self]
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return var_vals[self] if self in var_vals else self

@dataclass(frozen=True, repr=False)
class NumNode(Node):
  b: int
  @property
  def min(self): return self.b
  @property
  def max(self): return self.b
  def __eq__(self, other): return self.b == other
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return self


@dataclass(frozen=True, repr=False)
class OpNode(Node):
  a: Node
  b: Union[Node, int]
  min: int
  max: int
  @classmethod
  @functools.lru_cache(None)
  def create(cls,a,b):
    if cls is LtNode:
      if a.max < b: return NumNode(1)
      if a.min >= b: return NumNode(0)
      return cls(a,b, 0, 1)
    mmin, mmax = cls.get_bounds(a,b)
    if mmax == mmin: return NumNode(mmin)
    return cls(a,b, mmin, mmax)
  def vars(self): return self.a.vars() + (self.b.vars() if isinstance(self.b, Node) else [])
  @staticmethod
  def get_bounds(a: Node, b: Node | int) -> Tuple[int, int]: return (0,0)

class LtNode(OpNode):
  def __floordiv__(self, b: Union[Node, int], _=False): return (self.a//b) < (self.b//b)
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return self.a.substitute(var_vals) < (self.b if isinstance(self.b, int) else self.b.substitute(var_vals))

class MulNode(OpNode):
  def __mul__(self, b: Union[Node, int]): return self.a*(self.b*b) # two muls in one mul
  def __floordiv__(self, b: Union[Node, int], factoring_allowed=False): # NOTE: mod negative isn't handled right
    if self.b % b == 0: return self.a*(self.b//b)
    if b % self.b == 0 and self.b > 0: return self.a//(b//self.b)
    return Node.__floordiv__(self, b, factoring_allowed)
  def __mod__(self, b: Union[Node, int]):
    a = (self.a * (self.b%b))
    return Node.__mod__(a, b)
  @staticmethod
  def get_bounds(a: Node, b: Union[Node, int])  -> Tuple[int, int]: return (a.min*b, a.max*b) if b >= 0 else (a.max*b, a.min*b)
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return self.a.substitute(var_vals) * (self.b if isinstance(self.b, int) else self.b.substitute(var_vals))

class DivNode(OpNode):
  def __floordiv__(self, b: Union[Node, int], _=False): return self.a//(self.b*b) # two divs is one div
  @staticmethod
  def get_bounds(a: Node, b: Union[Node, int])  -> Tuple[int, int]:
    assert a.min >= 0 and isinstance(b, int)
    return a.min//b, a.max//b
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return self.a.substitute(var_vals) // self.b

class ModNode(OpNode):
  def __floordiv__(self, b: Union[Node, int], factoring_allowed=True):
    if (self.b % b == 0): return (self.a//b) % (self.b//b) # put the div inside mod
    return Node.__floordiv__(self, b, factoring_allowed)
  @staticmethod
  def get_bounds(a: Node, b: Union[Node, int]) -> Tuple[int, int]:
    assert a.min >= 0 and isinstance(b, int)
    return (0, b-1) if a.max - a.min >= b or (a.min != a.max and a.min%b >= a.max%b) else (a.min%b, a.max%b)
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return self.a.substitute(var_vals) % self.b

@dataclass(frozen=True)
class RedNode(Node):
  nodes: Tuple[Node, ...]
  min: int
  max: int
  def vars(self): return functools.reduce(lambda l,x: l+x.vars(), self.nodes, [])
  def __eq__(self, other): return self.__class__ == other.__class__ and len(self.nodes) == len(other.nodes) and self.key == other.key

class SumNode(RedNode):
  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def __mul__(self, b: Union[Node, int]): return Node.sum([x*b for x in self.nodes]) # distribute mul into sum
  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def __floordiv__(self, b: Union[Node, int], factoring_allowed=True):
    fully_divided: List[Node] = []
    rest: List[Node] = []
    if isinstance(b, SumNode):
      nu_num = sum(node.b for node in self.flat_components if node.__class__ is NumNode)
      de_num = sum(node.b for node in b.flat_components if node.__class__ is NumNode)
      if nu_num > 0 and de_num and (d:=nu_num//de_num) > 0: return NumNode(d) + (self-b*d) // b
    if isinstance(b, Node):
      for x in self.flat_components:
        if x % b == 0: fully_divided.append(x // b)
        else: rest.append(x)
      if (sum_fully_divided:=create_rednode(SumNode, fully_divided)) != 0: return sum_fully_divided + create_rednode(SumNode, rest) // b
      return Node.__floordiv__(self, b, False)
    if b == 1: return self
    if not factoring_allowed: return Node.__floordiv__(self, b, factoring_allowed)
    fully_divided, rest = [], []
    _gcd = b
    divisor = 1
    for x in self.flat_components:
      if x.__class__ in (NumNode, MulNode):
        if x.b%b == 0: fully_divided.append(x//b)
        else:
          rest.append(x)
          _gcd = gcd(_gcd, x.b)
          if x.__class__ == MulNode and divisor == 1 and b%x.b == 0: divisor = x.b
      else:
        rest.append(x)
        _gcd = 1
    if _gcd > 1: return Node.sum(fully_divided) + Node.sum(rest).__floordiv__(_gcd) // (b//_gcd)
    if divisor > 1: return Node.sum(fully_divided) + Node.sum(rest).__floordiv__(divisor) // (b//divisor)
    return Node.sum(fully_divided) + Node.__floordiv__(Node.sum(rest), b)

  @functools.lru_cache(maxsize=None)  # pylint: disable=method-cache-max-size-none
  def __mod__(self, b: Union[Node, int]):
    if isinstance(b, SumNode):
      nu_num = sum(node.b for node in self.flat_components if node.__class__ is NumNode)
      de_num = sum(node.b for node in b.flat_components if node.__class__ is NumNode)
      if nu_num > 0 and de_num and (d:=nu_num//de_num) > 0: return (self-b*d) % b
    if isinstance(b, Node) and (b - self).min > 0: return self # b - self simplifies the node
    new_nodes: List[Node] = []
    for x in self.nodes:
      if x.__class__ is NumNode: new_nodes.append(Variable.num(x.b%b))
      elif isinstance(x, MulNode): new_nodes.append(x.a * (x.b%b))
      else: new_nodes.append(x)
    return Node.__mod__(Node.sum(new_nodes), b)

  def __lt__(self, b:Union[Node,int]):
    lhs: Node = self
    if isinstance(b, int):
      new_sum = []
      for x in self.nodes:
        # TODO: should we just force the last one to always be the number
        if isinstance(x, NumNode): b -= x.b
        else: new_sum.append(x)
      lhs = Node.sum(new_sum)
      if isinstance(lhs, SumNode):
        muls, others = partition(lhs.nodes, lambda x: isinstance(x, MulNode) and x.b > 0 and x.max >= b)
        if muls:
          # NOTE: gcd in python 3.8 takes exactly 2 args
          mul_gcd = muls[0].b
          for x in muls[1:]: mul_gcd = gcd(mul_gcd, x.b)  # type: ignore  # mypy cannot tell x.b is int here
          if b%mul_gcd == 0:
            all_others = Variable.sum(others)
            if all_others.min >= 0 and all_others.max < mul_gcd:
              # TODO: should we divide both by mul_gcd here?
              lhs = Variable.sum(muls)
    return Node.__lt__(lhs, b)

  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node: return Variable.sum([node.substitute(var_vals) for node in self.nodes])

  @property
  def flat_components(self): # recursively expand sumnode components
    new_nodes = []
    for x in self.nodes: new_nodes += (x.flat_components if isinstance(x, SumNode) else [x])
    return new_nodes

class AndNode(RedNode):
  def __floordiv__(self, b: Union[Node, int], _=True): return Variable.ands([x//b for x in self.nodes])
  def substitute(self, var_vals: Dict[VariableOrNum, Node]) -> Node:
    subed = []
    for node in self.nodes:
      if not (sub:=node.substitute(var_vals)): return NumNode(0)
      subed.append(sub)
    return Variable.ands(subed)

def create_rednode(typ:Type[RedNode], nodes:List[Node]):
  if typ == SumNode: mmin, mmax = (sum([x.min for x in nodes]), sum([x.max for x in nodes]))
  elif typ == AndNode: mmin, mmax = (min([x.min for x in nodes]), max([x.max for x in nodes]))
  return typ(tuple(nodes), mmin, mmax) if mmax != mmin else NumNode(mmin)


@functools.lru_cache(maxsize=None)
def sym_rename(s) -> str: return f"s{sym_rename.cache_info().currsize}"
def sym_render(a: Union[Node, int], ops=None, ctx=None) -> str: return str(a) if isinstance(a, int) else a.render(ops, ctx)
def sym_infer(a: Union[Node, int], var_vals: Dict[Variable, int]) -> int:
  if isinstance(a, int): return a
  ret = a.substitute({k:Variable.num(v) for k, v in var_vals.items()})
  assert isinstance(ret, NumNode)
  return ret.b

# symbolic int
sint = Union[Node, int]
VariableOrNum = Union[Variable, NumNode]

render_python: Dict[Type, Callable] = {
  Variable: lambda self,ops,ctx: f"{self.expr}[{self.min}-{self.max}]" if ctx == "DEBUG" else f"{self.expr}",
  NumNode: lambda self,ops,ctx: f"{self.b}",
  MulNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}*{sym_render(self.b,ops,ctx)})",
  DivNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}//{self.b})",
  ModNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}%{self.b})",
  LtNode: lambda self,ops,ctx: f"({self.a.render(ops,ctx)}<{sym_render(self.b,ops,ctx)})",
  SumNode: lambda self,ops,ctx: f"({'+'.join(sorted([x.render(ops,ctx) for x in self.nodes]))})",
  AndNode: lambda self,ops,ctx: f"({' and '.join(sorted([x.render(ops,ctx) for x in self.nodes]))})"
}
