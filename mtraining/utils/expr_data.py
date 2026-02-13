import os
from dataclasses import dataclass

@dataclass
class ExprData:
    global_batch_size: int
    micro_batch_size: int
    reuse_type: str

EXPR_DATA = ExprData(64, 1, "match")

def update_expr_data(args):
    global EXPR_DATA
    EXPR_DATA = ExprData(args.global_batch_size, args.micro_batch_size, args.reuse_type)