"""
Model Adapter Layer (1.5)
=========================
唯一需要為「新的開源模型架構」寫程式碼的地方。
下游的 collector.py、schema、前端 HTML 一律只認識這裡定義的統一介面，
完全不需要知道底層是 Llama / Gemma / Mistral / Qwen / GPT-2 / MoE ...。

要支援一個新架構，只要新增一個 ModelAdapter 子類別，實作：
  layers()          -> list[nn.Module]   每一層 decoder layer
  mlp_down_proj(l)  -> nn.Module         MLP 輸出投影前一層（用來抓 top-k neuron）
  final_norm(h)      -> Tensor           套用模型最終的 norm
  lm_head(h)          -> Tensor           投影到 vocab logits（給 logit lens 用）
  embed_tokens(ids)   -> Tensor
  info()              -> AdapterInfo
並在 ModelAdapter.detect() 裡加一個判斷分支即可。
"""
from dataclasses import dataclass
import torch.nn as nn


@dataclass
class AdapterInfo:
    n_layers: int
    n_heads: int
    n_kv_heads: int
    hidden_size: int
    head_dim: int
    d_ff: int
    is_moe: bool = False


class ModelAdapter:
    name = "base"

    def __init__(self, model):
        self.model = model
        self.config = model.config

    def layers(self):
        raise NotImplementedError

    def mlp_down_proj(self, layer_idx) -> nn.Module:
        raise NotImplementedError

    def final_norm(self, hidden):
        raise NotImplementedError

    def lm_head(self, hidden):
        raise NotImplementedError

    def embed_tokens(self, input_ids):
        raise NotImplementedError

    def info(self) -> AdapterInfo:
        raise NotImplementedError

    @staticmethod
    def detect(model) -> "ModelAdapter":
        cname = model.__class__.__name__
        # 不同架構家族把「base model」包在不同屬性名下（.model / .transformer / 自己），
        # 依序試探即可，這裡本身就是 adapter 層唯一該碰「命名細節」的地方。
        candidates = [getattr(model, "model", None), getattr(model, "transformer", None), model]

        for base in candidates:
            if base is None:
                continue
            # Llama / Mistral / Qwen2 / Gemma / Gemma2 / StableLM / Phi 家族
            if hasattr(base, "layers") and len(base.layers) > 0:
                layer0 = base.layers[0]
                if hasattr(layer0, "mlp") and hasattr(layer0.mlp, "gate_proj") and hasattr(layer0.mlp, "down_proj"):
                    return LlamaFamilyAdapter(model)
                if hasattr(layer0, "mlp") and hasattr(layer0.mlp, "block_sparse_moe"):
                    raise ValueError(
                        f"{cname} 是 MoE 架構 (e.g. Mixtral)，目前沒有對應 adapter。"
                        " 請新增 MixtralAdapter 子類別（mlp_down_proj 需改成對選中 expert 的 w2）。"
                    )
            # GPT-2 家族：base.h[i].attn (combined c_attn) / .mlp.{c_fc,c_proj}
            if hasattr(base, "h") and len(base.h) > 0 and hasattr(base.h[0], "attn"):
                return GPT2Adapter(model)

        raise ValueError(
            f"沒有為架構 {cname} 註冊 adapter。"
            " 請在 model_adapter.py 新增一個 ModelAdapter 子類別（通常只要 20~30 行）。"
        )


class LlamaFamilyAdapter(ModelAdapter):
    """涵蓋 Llama / Mistral / Qwen2 / Gemma / Gemma2 / StableLM / Phi 等
    共用「gate/up/down_proj + RMSNorm」骨架的架構。"""
    name = "llama_family"

    def __init__(self, model):
        super().__init__(model)
        self.base = getattr(model, "model", model)

    def layers(self):
        return list(self.base.layers)

    def mlp_down_proj(self, layer_idx):
        return self.base.layers[layer_idx].mlp.down_proj

    def final_norm(self, hidden):
        return self.base.norm(hidden)

    def lm_head(self, hidden):
        return self.model.lm_head(hidden)

    def embed_tokens(self, input_ids):
        return self.base.embed_tokens(input_ids)

    def info(self) -> AdapterInfo:
        c = self.config
        n_heads = getattr(c, "num_attention_heads")
        n_kv = getattr(c, "num_key_value_heads", n_heads)
        hidden = getattr(c, "hidden_size")
        # 部分較新架構（例如 Gemma4：sliding-window 層與 global 層 head_dim 不同）
        # 把 head_dim 設計成「每層可能不一樣」，直接讀 config.head_dim 在這種情況下
        # 會主動丟例外，而不是回傳一個可能誤導的值。這裡退而求其次，只把它當成
        # metadata 顯示用的近似值，不影響 collector 實際抓資料的正確性
        # （attention/hidden state 一律直接讀模型輸出的真實 tensor shape）。
        try:
            head_dim = getattr(c, "head_dim", hidden // n_heads)
        except Exception:
            head_dim = hidden // n_heads
        try:
            d_ff = getattr(c, "intermediate_size", 4 * hidden)
        except Exception:
            d_ff = 4 * hidden
        return AdapterInfo(
            n_layers=getattr(c, "num_hidden_layers"),
            n_heads=n_heads, n_kv_heads=n_kv,
            hidden_size=hidden, head_dim=head_dim, d_ff=d_ff, is_moe=False,
        )


class GPT2Adapter(ModelAdapter):
    """示範第二種、結構明顯不同的架構：combined qkv (c_attn) + Conv1D MLP。
    用來證明「只改 adapter，其餘一行都不用動」這件事。"""
    name = "gpt2_family"

    def __init__(self, model):
        super().__init__(model)
        self.base = getattr(model, "transformer", model)

    def layers(self):
        return list(self.base.h)

    def mlp_down_proj(self, layer_idx):
        return self.base.h[layer_idx].mlp.c_proj

    def final_norm(self, hidden):
        return self.base.ln_f(hidden)

    def lm_head(self, hidden):
        return self.model.lm_head(hidden)

    def embed_tokens(self, input_ids):
        return self.base.wte(input_ids)

    def info(self) -> AdapterInfo:
        c = self.config
        hidden = c.n_embd
        n_heads = c.n_head
        d_ff = getattr(c, "n_inner", None) or 4 * hidden
        return AdapterInfo(
            n_layers=c.n_layer, n_heads=n_heads, n_kv_heads=n_heads,
            hidden_size=hidden, head_dim=hidden // n_heads, d_ff=d_ff, is_moe=False,
        )
