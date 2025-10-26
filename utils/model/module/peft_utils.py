# utils/model/module/peft_utils.py
import re
import contextlib
from typing import List, Tuple, Optional

import torch
import torch.nn as nn

from .peft.osm_lora import LoRALinear


def _split_parent(root: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def dump_some_module_names(model: nn.Module, max_count: int = 200):
    print("------ [DUMP] first module names ------")
    for i, (n, m) in enumerate(model.named_modules()):
        print(f"{i:03d}: {n} :: {type(m).__name__}")
        if i + 1 >= max_count:
            break
    print("------ [DUMP END] ------")


def _fullmatch(name: str, pattern: re.Pattern) -> bool:
    # 與你原本一致：用 fullmatch，避免誤中太多層
    return pattern.fullmatch(name) is not None


def install_lora(
    model: nn.Module,
    include_name_regex: str,
    r: int,
    alpha: float,
    sparsity: float = 1.0,
) -> List[str]:
    """
    在符合 include_name_regex 的 nn.Linear 上安裝 LoRA。
    只做最小侵入：其它層不動。
    回傳命中的模組名稱列表。
    """
    if r is None or int(r) <= 0:
        print("[LoRA] rank<=0: skip installing LoRA.")
        return []

    pattern = re.compile(include_name_regex)
    matched: List[str] = []
    print(f"[LoRA] install_lora start | regex={include_name_regex} | r={r} alpha={alpha} sparsity={sparsity}")

    # 找出 fullmatch 的模組名稱
    candidates: List[str] = []
    for name, module in model.named_modules():
        if _fullmatch(name, pattern):
            candidates.append(name)

    # 替換（跳過已是 LoRA 的層）
    for name in candidates:
        parent, attr = _split_parent(model, name)
        leaf = getattr(parent, attr)
        if isinstance(leaf, LoRALinear):
            print(f"[LoRA] skip {name}: already LoRALinear")
            continue
        if isinstance(leaf, nn.Linear):
            lora = LoRALinear(leaf, rank=int(r), alpha=float(alpha), sparsity=float(sparsity))
            setattr(parent, attr, lora)
            matched.append(name)
            print(f"[LoRA] attached -> {name}  (type was nn.Linear)")
        else:
            print(f"[LoRA] skip {name}: not nn.Linear (got {type(leaf).__name__})")

    # Note: sparsity is applied at LoRA delta via element-wise masks; topology-aware OSM for CNN is out-of-scope here.

    print(f"[LoRA] matched count = {len(matched)}")
    return matched


def count_lora_modules(model: nn.Module) -> int:
    n = 0
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            n += 1
    return n


@contextlib.contextmanager
def merge_for_eval(model: nn.Module, enable: bool = True):
    """
    with merge_for_eval(model, True):   # 進入時 merge，退出還原
        ... validation ...
    """
    if not enable:
        yield
        return

    mods: List[LoRALinear] = []
    for _, m in model.named_modules():
        if isinstance(m, LoRALinear):
            mods.append(m)

    print(f"[LoRA] merge_for_eval: merging {len(mods)} modules ...")
    try:
        for m in mods:
            m.merge()
        yield
    finally:
        for m in mods:
            m.unmerge()
        print("[LoRA] merge_for_eval: unmerged.")


# =========================
#  LoRA-safe 擴張工具（關鍵）
# =========================

@torch.no_grad()
def expand_lora_linear_out_features(layer: LoRALinear, new_out: int) -> LoRALinear:
    """out_features 擴大時保留 base 與 LoRA（只初始化新增行）"""
    assert isinstance(layer, LoRALinear)
    if new_out <= layer.out_features:
        return layer

    # 建新層
    base_linear = nn.Linear(layer.in_features, new_out, bias=(layer.bias is not None))
    # 先用舊 base 權重/bias 初始化
    base_linear.weight[:layer.out_features].copy_(layer.weight)
    if layer.bias is not None:
        base_linear.bias[:layer.out_features].copy_(layer.bias)

    new = LoRALinear(base_linear, rank=layer.rank, alpha=layer.alpha, sparsity=layer.sparsity)
    # copy LoRA（只擴充新增輸出行）
    if layer.rank > 0:
        new.lora_A[:layer.rank, :].copy_(layer.lora_A)                 # [r, in]
        new.lora_B[:layer.out_features, :].copy_(layer.lora_B)         # [out_old, r]
        # 新增輸出行的 lora_B 已是 0 初始化

    # 允許 base weight 對新類別持續調整
    new.weight.requires_grad_(True)
    if new.bias is not None:
        new.bias.requires_grad_(True)
    if new.rank > 0:
        new.lora_A.requires_grad_(True)
        new.lora_B.requires_grad_(True)
    return new


def _strip_module_prefix(path: str) -> str:
    return path[7:] if path.startswith("module.") else path


def expand_classifier_preserve_lora(model: nn.Module, head_path: str, new_out: int, r: int, alpha: float):
    """
    擴張分類器並保留 LoRA（若是 Linear→先擴再換 LoRA；若本來就是 LoRA→用上面 expand）
    head_path 例如：'classifier.fc' 或 'module.classifier.fc'
    """
    base = model.module if isinstance(model, nn.DataParallel) else model
    head_path = _strip_module_prefix(head_path)
    parent, key = _split_parent(base, head_path)
    head = getattr(parent, key)

    if isinstance(head, LoRALinear):
        new_head = expand_lora_linear_out_features(head, new_out)
    elif isinstance(head, nn.Linear):
        # 先擴張 Linear，再包成 LoRA，並拷貝舊權重
        new_linear = nn.Linear(head.in_features, new_out, bias=(head.bias is not None))
        with torch.no_grad():
            new_linear.weight[:head.out_features].copy_(head.weight)
            if head.bias is not None:
                new_linear.bias[:head.out_features].copy_(head.bias)
    new_head = LoRALinear(new_linear, rank=r, alpha=alpha, sparsity=1.0)
        # 允許 base weight 與 bias 一併更新
        new_head.weight.requires_grad_(True)
        if new_head.bias is not None:
            new_head.bias.requires_grad_(True)
    else:
        raise TypeError(f'head "{head_path}" must be Linear/LoRALinear, got {type(head)}')

    setattr(parent, key, new_head)
    return new_head


def guess_head_path(model: nn.Module) -> Optional[str]:
    """
    嘗試推斷分類器 Linear 的路徑，例如 'classifier.fc'。
    規則：尋找 base.classifier 底下第一個 nn.Linear/LoRALinear。
    """
    base = model.module if isinstance(model, nn.DataParallel) else model
    if not hasattr(base, "classifier"):
        return None

    # 直接優先找 classifier.fc
    if hasattr(base.classifier, "fc") and isinstance(getattr(base.classifier, "fc"), (nn.Linear, LoRALinear)):
        return "classifier.fc"

    # 否則在 classifier 子模組裡找 leaf Linear
    for name, mod in base.classifier.named_modules():
        if isinstance(mod, (nn.Linear, LoRALinear)):
            # name 可能是 'fc'、'head' 等
            return f"classifier.{name}" if name else "classifier"
    return None
