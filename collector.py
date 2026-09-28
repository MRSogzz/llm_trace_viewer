"""
採集層 (Hook / Tap) + 資料整形層
==================================
真的去跑模型、真的去抓張量的地方。輸出格式嚴格對應 SKILL 文件裡定義的
Run / Step / LayerTrace schema，前端 HTML 只讀這個 JSON，不管底層模型是什麼。

兩種取值手段並用：
  1. output_attentions=True / output_hidden_states=True
     -> HF 標準介面，幾乎所有 causal LM 架構通用，完全不用寫 hook
        (但需要 attn_implementation="eager"，sdpa/flash 不會回傳機率)
  2. 一個 forward_pre_hook 掛在 adapter.mlp_down_proj(l)
     -> 抓 MLP 在 down_proj 之前的 activation，做 top-k neuron
        (這是唯一需要走「真的 hook」的地方，因為 HF 標準輸出不包含它)
"""
import time
import torch
from model_adapter import ModelAdapter

TOPK_LOGITS = 8
TOPK_NEURONS = 8
TOPK_ATTN_KEYS = 8


class MLPTap:
    """掛在每一層 mlp_down_proj 上的 forward_pre_hook，抓「進 down_proj 前」的向量。"""

    def __init__(self, adapter, layer_indices):
        self.buf = {}
        self.handles = []
        for l in layer_indices:
            mod = adapter.mlp_down_proj(l)

            def make_hook(li):
                def hook(module, inputs):
                    self.buf[li] = inputs[0].detach()
                return hook

            self.handles.append(mod.register_forward_pre_hook(make_hook(l)))

    def pop(self, l):
        return self.buf.pop(l, None)

    def remove(self):
        for h in self.handles:
            h.remove()


def _topk_neurons(vec, k=TOPK_NEURONS):
    if vec is None:
        return []
    k = min(k, vec.numel())
    vals, idx = torch.topk(vec.abs(), k)
    out = [{"neuron_id": int(i), "activation": round(float(vec[i]), 4)} for i in idx.tolist()]
    out.sort(key=lambda d: -abs(d["activation"]))
    return out


def _candidates(logits_row, tokenizer, k=TOPK_LOGITS):
    probs = torch.softmax(logits_row.float(), dim=-1)
    vals, idx = torch.topk(probs, k)
    # 用 convert_ids_to_tokens 拿「原始 token piece」，不要逐一 decode()：
    # 中/日/韓文字常常一個字被切成好幾個 token（尤其 byte-level BPE），
    # 對單一 token id 呼叫 decode() 會因為 UTF-8 byte 不完整而吐出 � 亂碼。
    # convert_ids_to_tokens 永遠回傳穩定、正確對應該 token 的字串，不會亂碼。
    pieces = tokenizer.convert_ids_to_tokens(idx.tolist())
    return [
        {"token_id": int(i), "text": piece, "prob": round(float(v), 5), "rank": r}
        for r, (v, i, piece) in enumerate(zip(vals.tolist(), idx.tolist(), pieces))
    ]


def _attn_head_record(attn_row, k=TOPK_ATTN_KEYS):
    # attn_row: [n_key] 該 head、該 query 位置(通常取最後一個 query)的 softmax 權重
    k = min(k, attn_row.numel())
    vals, idx = torch.topk(attn_row, k)
    p = attn_row.clamp_min(1e-12)
    entropy = float(-(p * p.log()).sum())
    return {
        "topk_keys": [{"key_pos": int(i), "weight": round(float(v), 5)} for v, i in zip(vals.tolist(), idx.tolist())],
        "entropy": round(entropy, 4),
    }


def _tok_list(ids, offset, tokenizer):
    special = set(tokenizer.all_special_ids or [])
    # 同上：用 convert_ids_to_tokens 而不是逐一 decode()，避免多 byte 字元被拆散時的亂碼。
    pieces = tokenizer.convert_ids_to_tokens(list(ids))
    return [
        {"id": int(t), "text": piece, "position": offset + i, "is_special": int(t) in special}
        for i, (t, piece) in enumerate(zip(ids, pieces))
    ]


def _sample_next(logits_row, do_sample, temperature, top_p, top_k):
    """跟 HF generate() 同一套邏輯：temperature -> top_k -> top_p -> (greedy或取樣)。"""
    if not do_sample:
        return int(torch.argmax(logits_row))
    logits = logits_row.float() / max(temperature, 1e-5)
    if top_k and top_k > 0:
        kth_val = torch.topk(logits, min(top_k, logits.numel()))[0][..., -1]
        logits = torch.where(logits < kth_val, torch.full_like(logits, float("-inf")), logits)
    if top_p and top_p < 1.0:
        sorted_logits, sorted_idx = torch.sort(logits, descending=True)
        probs_sorted = torch.softmax(sorted_logits, dim=-1)
        cum = torch.cumsum(probs_sorted, dim=-1)
        remove = cum - probs_sorted > top_p
        sorted_logits[remove] = float("-inf")
        logits = torch.full_like(logits, float("-inf")).scatter(0, sorted_idx, sorted_logits)
    probs = torch.softmax(logits, dim=-1)
    return int(torch.multinomial(probs, 1))


@torch.no_grad()
def collect_trace(model, tokenizer, prompt, max_new_tokens=8, layer_stride=1, head_stride=1, device="cpu",
                   do_sample=False, temperature=1.0, top_p=1.0, top_k=0, seed=None):
    """
    do_sample=False（預設）：greedy，跟 model.generate(do_sample=False) 逐token完全一致，適合除錯/比對。
    do_sample=True：套用 temperature/top_p/top_k 取樣，如果你原本工具（llama.cpp/聊天前端等）
    預設就是用取樣在生成，想要「軌跡看起來跟原本輸出一樣」通常要把這幾個參數設成跟原本工具一致，
    並用同一個 --seed，否則兩邊本來就會抽到不同token、輸出自然對不上，這不是bug。
    """
    model.eval().to(device)
    if seed is not None:
        torch.manual_seed(seed)
    adapter = ModelAdapter.detect(model)
    info = adapter.info()
    layer_ids = list(range(0, info.n_layers, layer_stride))
    head_ids = list(range(0, info.n_heads, head_stride))

    tap = MLPTap(adapter, layer_ids)

    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    prompt_len = input_ids.shape[1]
    prompt_tokens = _tok_list(input_ids[0].tolist(), 0, tokenizer)

    past = None
    cur_input = input_ids
    generated_ids = []
    steps = []
    eos_id = tokenizer.eos_token_id

    for step_idx in range(1 + max_new_tokens):
        is_prefill = step_idx == 0
        out = model(
            input_ids=cur_input,
            past_key_values=past,
            use_cache=True,
            output_attentions=True,
            output_hidden_states=True,
        )
        past = out.past_key_values
        logits = out.logits  # [1, seq, vocab]
        next_id = _sample_next(logits[0, -1], do_sample, temperature, top_p, top_k)

        if hasattr(past, "get_seq_length"):
            cache_len_after = past.get_seq_length()
        else:  # 舊版 tuple((k,v), ...) 格式
            cache_len_after = past[0][0].shape[-2]
        cache_len_before = cache_len_after - cur_input.shape[1]

        hidden_states = out.hidden_states  # (embed, after_layer_0, after_layer_1, ...)
        attentions = out.attentions        # (layer_0_attn, layer_1_attn, ...)

        layer_traces = []
        for l in layer_ids:
            h_l = hidden_states[l + 1][0]                       # [seq, hidden]
            resid_norms = [round(float(x), 3) for x in h_l.norm(dim=-1)]

            normed = adapter.final_norm(h_l[-1:].unsqueeze(0))  # [1,1,hidden]
            logits_l = adapter.lm_head(normed)[0, -1]
            logit_lens = _candidates(logits_l, tokenizer)

            att_l = attentions[l][0]  # [n_heads, q, k] —— 某些新架構(如Gemma4)各層 head 設定可能不同，
            local_head_ids = [hh for hh in head_ids if hh < att_l.shape[0]]  # 用實際 tensor 形狀為準，不硬信全域設定
            heads_out = [{"head_index": hh, **_attn_head_record(att_l[hh, -1])} for hh in local_head_ids]

            act = tap.pop(l)  # [1, seq, d_ff] 或 None
            last_act = act[0, -1] if act is not None else None
            layer_traces.append({
                "layer_index": l,
                "residual_norms": resid_norms,
                "attn": {"heads": heads_out},
                "mlp": {
                    "topk_neurons": _topk_neurons(last_act),
                    "activation_norm": round(float(last_act.norm()), 3) if last_act is not None else None,
                },
                "logit_lens": logit_lens,
            })

        offset = 0 if is_prefill else (prompt_len + len(generated_ids) - 1)
        step_input_tokens = _tok_list(cur_input[0].tolist(), offset, tokenizer)
        output_distribution = _candidates(logits[0, -1], tokenizer)
        out_pos = prompt_len + len(generated_ids)

        steps.append({
            "index": step_idx,
            "type": "prefill" if is_prefill else "decode",
            "input_tokens": step_input_tokens,
            "cache_len_before": int(cache_len_before),
            "cache_len_after": int(cache_len_after),
            "output_token": {
                "id": next_id, "text": tokenizer.convert_ids_to_tokens([next_id])[0],
                "position": out_pos, "is_special": next_id == eos_id,
            },
            "output_distribution": output_distribution,
            "layers": layer_traces,
            "kv_cache_snapshot": {"length": int(cache_len_after)},
        })

        if eos_id is not None and next_id == eos_id:
            break
        generated_ids.append(next_id)
        if len(generated_ids) >= max_new_tokens:
            break
        cur_input = torch.tensor([[next_id]], device=device)

    tap.remove()

    return {
        "run_id": f"run_{int(time.time())}",
        "model_name": getattr(model.config, "name_or_path", None) or model.__class__.__name__,
        "backend": "transformers",
        "adapter_used": adapter.name,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": _tok_list(generated_ids, prompt_len, tokenizer),
        # 這兩個欄位是「真正的、跟原本輸出一致」的完整字串（一次性 decode 整段 ids），
        # 拿來跟你原本工具（llama.cpp / 聊天前端等）的輸出核對用；
        # 上面 tokens 裡逐個顯示的 text 是「token piece」，用來看模型內部切字方式，
        # 兩者故意不同，piece 不保證單獨看起來語意完整，這是預期行為不是bug。
        "prompt_text_decoded": tokenizer.decode(input_ids[0].tolist(), skip_special_tokens=True),
        "generated_text_decoded": tokenizer.decode(generated_ids, skip_special_tokens=True),
        "config": {"decoding": "greedy", "max_new_tokens": max_new_tokens,
                   "layer_stride": layer_stride, "head_stride": head_stride},
        "steps": steps,
        "layers_meta": [
            {"index": l, "n_heads": len(head_ids), "d_model": info.hidden_size,
             "d_head": info.head_dim, "d_ff": info.d_ff}
            for l in layer_ids
        ],
    }
