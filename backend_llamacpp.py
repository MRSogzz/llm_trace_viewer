"""
後端 2：llama.cpp + GGUF
=========================
專門處理「只有 GGUF 檔案」的情況，刻意不透過 transformers 讀 GGUF。

為什麼要分兩條後端、而不是都用 transformers：
  transformers 讀 GGUF 的本質是「把量化權重反量化回 HF 格式的 tensor，
  再用 transformers 自己的 attention/RoPE/MLP kernel 算」。這跟你原本用
  llama.cpp / ollama / LM Studio 實際跑出來的數值路徑不是同一套實作
  （量化反量化的細節、RoPE實作、attention kernel都可能有差異），
  遇到高度量化 (Q4_K_M等) 或較新/較冷門的架構時，容易出現「可視化跟原始
  輸出對不上」的狀況。
  這個後端改成直接呼叫 llama.cpp 本體 (透過 llama-cpp-python 綁定)，
  確保「畫出來的token」就是「llama.cpp真的選出來的token」，兩者保證一致。

代價（誠實列出，不要假裝有資料）：
  llama.cpp 是高度最佳化的 C++ 推論引擎，沒有 PyTorch hook 那種機制能拿到
  attention 機率矩陣、MLP 中間 activation、每層 residual/hidden state。
  這個後端只能提供：
    - 每一步實際被選中的 token（跟原始輸出保證一致）
    - 每一步 top-k 候選詞與其機率（llama.cpp 原生 logprobs 支援）
    - KV cache 長度
  拿不到的欄位（attn / mlp / logit_lens / residual_norms）一律回傳空值，
  前端偵測到空值會顯示「此後端不支援」，不會假裝畫出資料。
  如果你需要逐層 attention / MLP 可視化，那些細節只有後端1(transformers)
  能提供，代價是必須先把模型轉成/取得 safetensors 格式（非量化或bnb 4-bit）。
"""
import math
import time


def collect_trace_llamacpp(gguf_path, prompt, max_new_tokens=8, topk=8, n_ctx=4096, seed=0,
                            chat=True, system_prompt=None):
    """
    chat=True（預設）：用 llama-cpp-python 的 create_chat_completion()，
    它會自動讀取 GGUF 內嵌的 tokenizer.chat_template（Jinja2樣板，通常是原模型
    作者轉檔時一起包進去的），照聊天格式包裝 prompt 再丟給模型——這樣模型才會
    「回答」而不是把你的輸入當成純文字去續寫。這是先前版本輸出變成續寫的
    根本原因：原本直接把 prompt 字串丟給 create_completion()，完全沒有套用
    對話格式。
    chat=False：退回原本的純續寫模式（--raw），把 prompt 原封不動丟給模型。
    """
    try:
        from llama_cpp import Llama
    except ImportError as e:
        raise SystemExit(
            "需要 llama-cpp-python 才能用這個後端: pip install llama-cpp-python --break-system-packages\n"
            "（會現場編譯 llama.cpp 本體，第一次安裝需要幾分鐘）"
        ) from e

    llm = Llama(
        model_path=gguf_path,
        n_ctx=n_ctx,
        logits_all=True,
        verbose=False,
        seed=seed,
    )

    has_template = "tokenizer.chat_template" in (llm.metadata or {})
    if chat and not has_template:
        print("[警告] 這個 GGUF 沒有內嵌 chat_template（作者轉檔時沒帶），"
              "沒辦法自動套用聊天格式，退回純續寫模式。")
        chat = False

    if chat:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        out = llm.create_chat_completion(
            messages=messages,
            max_tokens=max_new_tokens,
            temperature=0.0,  # greedy，方便跟後端1比對
            logprobs=True,
            top_logprobs=topk,
        )
        choice = out["choices"][0]
        generated_text = choice["message"]["content"]
        lp_content = (choice.get("logprobs") or {}).get("content") or []

        final_cache_len = llm.n_tokens          # 呼叫完之後，這是「已處理格式化prompt + 全部生成token」的總長度
        n_generated = len(lp_content)
        prefill_cache_len = max(final_cache_len - n_generated, 0)

        def cand_list_chat(top_list):
            return [
                {"token_id": -1, "text": c["token"], "prob": round(math.exp(c["logprob"]), 5), "rank": r}
                for r, c in enumerate((top_list or [])[:topk])
            ]

        steps = []
        for i, entry in enumerate(lp_content):
            cache_before = prefill_cache_len + i
            cache_after = cache_before + 1
            steps.append({
                "index": i,
                "type": "prefill" if i == 0 else "decode",  # 第0個生成token代表prefill剛結束那一刻
                "input_tokens": [],
                "cache_len_before": int(cache_before),
                "cache_len_after": int(cache_after),
                "output_token": {"id": -1, "text": entry["token"], "position": cache_before, "is_special": False},
                "output_distribution": cand_list_chat(entry.get("top_logprobs")),
                "layers": [],
                "kv_cache_snapshot": {"length": int(cache_after)},
            })

        generated_tokens = [
            {"id": -1, "text": s["output_token"]["text"], "position": s["output_token"]["position"], "is_special": False}
            for s in steps
        ]

        return {
            "run_id": f"run_llamacpp_{int(time.time())}",
            "model_name": gguf_path,
            "backend": "llamacpp",
            "adapter_used": "llamacpp_native_chat",
            "prompt_tokens": [],
            "generated_tokens": generated_tokens,
            "prompt_text_decoded": prompt,
            "generated_text_decoded": generated_text,
            "config": {
                "decoding": "greedy", "max_new_tokens": max_new_tokens, "mode": "chat",
                "backend_limitations": "no attention/mlp/hidden-state access (llama.cpp doesn't expose per-layer internals)",
            },
            "steps": steps,
            "layers_meta": [],
        }

    # ---- chat=False：原本的純續寫模式 ----
    prompt_ids = llm.tokenize(prompt.encode("utf-8"))
    prompt_len = len(prompt_ids)

    out = llm.create_completion(
        prompt,
        max_tokens=max_new_tokens,
        temperature=0.0,
        logprobs=topk,
        echo=True,
    )
    choice = out["choices"][0]
    lp = choice.get("logprobs") or {}
    tokens_text = lp.get("tokens", [])
    top_logprobs_list = lp.get("top_logprobs", [])

    def cand_list(top_dict):
        if not top_dict:
            return []
        items = sorted(top_dict.items(), key=lambda kv: -(kv[1] if kv[1] is not None else -1e9))[:topk]
        return [
            {"token_id": -1, "text": t, "prob": round(math.exp(p), 5) if p is not None else 0.0, "rank": r}
            for r, (t, p) in enumerate(items)
        ]

    steps = []
    for i, text in enumerate(tokens_text):
        if i < prompt_len - 1:
            continue
        is_prefill = (i == prompt_len - 1)
        top_dict = top_logprobs_list[i] if i < len(top_logprobs_list) else None
        steps.append({
            "index": len(steps),
            "type": "prefill" if is_prefill else "decode",
            "input_tokens": [],
            "cache_len_before": i,
            "cache_len_after": i + 1,
            "output_token": {"id": -1, "text": text, "position": i, "is_special": False},
            "output_distribution": cand_list(top_dict),
            "layers": [],
            "kv_cache_snapshot": {"length": i + 1},
        })

    generated_tokens = [
        {"id": -1, "text": s["output_token"]["text"], "position": s["output_token"]["position"], "is_special": False}
        for s in steps if s["type"] == "decode"
    ]

    return {
        "run_id": f"run_llamacpp_{int(time.time())}",
        "model_name": gguf_path,
        "backend": "llamacpp",
        "adapter_used": "llamacpp_native_raw",
        "prompt_tokens": [],
        "generated_tokens": generated_tokens,
        "prompt_text_decoded": prompt,
        "generated_text_decoded": choice.get("text", "")[len(prompt):] if choice.get("text", "").startswith(prompt) else choice.get("text", ""),
        "config": {
            "decoding": "greedy", "max_new_tokens": max_new_tokens, "mode": "raw_completion",
            "backend_limitations": "no attention/mlp/hidden-state access (llama.cpp doesn't expose per-layer internals)",
        },
        "steps": steps,
        "layers_meta": [],
    }


def _tok_text(llm, tid):
    """單一 token 的顯示文字。中/日/韓字常被切成好幾個 token，單獨一個 token 可能只是
    不完整的 UTF-8 位元組，直接 decode 會變成 �（跟先前 transformers 後端同一個問題）。
    這種情況改顯示位元組本身，例如 <0xE4><0xBD>，不會出現亂碼，也不會跟別的 token 混淆。"""
    try:
        b = llm.detokenize([int(tid)])
    except Exception:
        return f"<{tid}>"
    try:
        return b.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(f"<0x{x:02X}>" for x in b)


def collect_trace_llamacpp_layers(gguf_path, prompt, max_new_tokens=8, topk=8, n_ctx=4096, seed=0,
                                   chat=True, system_prompt=None, lens=True):
    """
    含逐層資料的 llama.cpp 後端：用 ggml eval-callback 抓 attention / MLP / residual，
    並用手動 decode loop（llm.eval() 一次一步）讓每個 step 跟每次計算圖執行一一對應，
    跟後端1 的手寫 decode loop 同一種結構。greedy 解碼。

    注意：這條路為了拿到 kq_soft_max 節點會強制關閉 flash attention，
    速度比正常 llama.cpp 慢，而且沒有 logit lens（見 llamacpp_layers.py 說明）。
    chat=True 時用 GGUF 內嵌的 chat_template 組 prompt（Jinja2），否則直接用原文。
    """
    import numpy as np
    from llamacpp_layers import make_llama_with_layer_capture, layers_from_capture

    llm, cap = make_llama_with_layer_capture(gguf_path, n_ctx=n_ctx, seed=seed)

    text = prompt
    used_chat = False
    tmpl = (llm.metadata or {}).get("tokenizer.chat_template")
    if chat and tmpl:
        try:
            from jinja2 import Environment
            def tok_str(key):
                tid = (llm.metadata or {}).get(key)
                try:
                    return llm.detokenize([int(tid)]).decode("utf-8", "ignore") if tid is not None else ""
                except Exception:
                    return ""
            messages = ([{"role": "system", "content": system_prompt}] if system_prompt else []) + \
                       [{"role": "user", "content": prompt}]
            text = Environment().from_string(tmpl).render(
                messages=messages, add_generation_prompt=True,
                bos_token=tok_str("tokenizer.ggml.bos_token_id"),
                eos_token=tok_str("tokenizer.ggml.eos_token_id"),
            )
            used_chat = True
        except Exception as e:
            print(f"[警告] chat_template 渲染失敗，改用純續寫: {e}")
    elif chat and not tmpl:
        print("[警告] 這個 GGUF 沒有內嵌 chat_template，退回純續寫模式。")

    # add_bos=False：chat_template 通常已經自己帶了 bos_token 字串；純續寫才補 bos
    prompt_ids = llm.tokenize(text.encode("utf-8"), add_bos=not used_chat, special=True)
    prompt_len = len(prompt_ids)
    if prompt_len > n_ctx:
        raise SystemExit(f"prompt 有 {prompt_len} tokens，超過 --n_ctx {n_ctx}，請調大 --n_ctx")

    eos_id = llm.token_eos()
    steps, generated_ids = [], []
    cur = prompt_ids

    for step_idx in range(1 + max_new_tokens):
        cap.clear()
        llm.eval(cur)
        cache_after = llm.n_tokens
        cache_before = cache_after - len(cur)

        logits = np.array(llm.scores[cache_after - 1], dtype=np.float32)
        next_id = int(np.argmax(logits))
        z = logits - logits.max()
        probs = np.exp(z) / np.exp(z).sum()
        top_idx = np.argsort(-probs)[:topk]

        def piece(i):
            return _tok_text(llm, i)

        layers = layers_from_capture(cap, len(cur), valid_kv_len=cache_after, keep_hidden=lens)
        if step_idx == 0 and not layers:
            names = sorted(cap.seen)
            print("[警告] 逐層採集啟用了，但一個節點都沒抓到（attention/MLP/residual 都是空的）。")
            print("       這代表這個模型的計算圖節點名稱跟預期不同，或 callback 沒生效。")
            print(f"       計算圖裡實際出現的節點名稱（共 {len(names)} 種，前 60 個）：")
            print("       " + ", ".join(names[:60]))
            print("       把這段貼給我，我就能補上對應的節點名。也已寫入 run.json 的 config.debug_seen_nodes。")
        steps.append({
            "index": step_idx,
            "type": "prefill" if step_idx == 0 else "decode",
            "input_tokens": [{"id": int(t), "text": piece(t), "position": cache_before + j, "is_special": False}
                             for j, t in enumerate(cur)],
            "cache_len_before": int(cache_before),
            "cache_len_after": int(cache_after),
            "output_token": {"id": next_id, "text": piece(next_id), "position": cache_after,
                             "is_special": next_id == eos_id},
            "output_distribution": [{"token_id": int(i), "text": piece(i), "prob": round(float(probs[i]), 5), "rank": r}
                                    for r, i in enumerate(top_idx)],
            "layers": layers,
            "kv_cache_snapshot": {"length": int(cache_after)},
        })

        if next_id == eos_id:
            break
        generated_ids.append(next_id)
        if len(generated_ids) >= max_new_tokens:
            break
        cur = [next_id]

    lens_info = {"enabled": False}
    if lens:
        lens_info = _fill_logit_lens(llm, gguf_path, steps, topk)
    for st in steps:                      # 暫存的殘差向量不能進 json
        for ly in st["layers"]:
            ly.pop("_hidden", None)

    n_layers = len(steps[0]["layers"]) if steps else 0
    n_heads = len(steps[0]["layers"][0]["attn"]["heads"]) if n_layers and steps[0]["layers"][0]["attn"]["heads"] else 0
    return {
        "run_id": f"run_llamacpp_layers_{int(time.time())}",
        "model_name": gguf_path,
        "backend": "llamacpp",
        "adapter_used": "llamacpp_eval_callback",
        "prompt_tokens": [{"id": int(t), "text": _tok_text(llm, t),
                           "position": i, "is_special": False} for i, t in enumerate(prompt_ids)],
        "generated_tokens": [{"id": int(t), "text": _tok_text(llm, t),
                              "position": prompt_len + i, "is_special": False} for i, t in enumerate(generated_ids)],
        "prompt_text_decoded": text,
        "generated_text_decoded": llm.detokenize(generated_ids).decode("utf-8", "replace") if generated_ids else "",
        "config": {"decoding": "greedy", "max_new_tokens": max_new_tokens,
                   "mode": "chat" if used_chat else "raw_completion", "layers": "ggml_eval_callback",
                   "backend_limitations": "flash attention disabled (needed to expose kq_soft_max)",
                   "debug_seen_nodes": sorted(cap.seen)[:80], "logit_lens": lens_info},
        "steps": steps,
        "layers_meta": [{"index": s["layer_index"], "n_heads": n_heads, "d_model": 0, "d_head": 0, "d_ff": 0}
                        for s in (steps[0]["layers"] if steps else [])],
    }


def _fill_logit_lens(llm, gguf_path, steps, topk):
    """直接讀 GGUF 的輸出權重，對每個 (step, layer) 的殘差做 logit lens，填進 steps。詳見 llamacpp_lens.py。"""
    import numpy as np
    try:
        from llamacpp_lens import LogitLens
        lens = LogitLens(gguf_path)
        idx, vecs = [], []
        for si, st in enumerate(steps):
            for li, ly in enumerate(st["layers"]):
                h = ly.get("_hidden")
                if h is not None and h.shape[0] == lens.hidden:
                    idx.append((si, li)); vecs.append(h)
        if not vecs:
            print("[logit lens] 沒有可用的殘差向量（l_out 沒抓到），略過")
            return {"enabled": False, "reason": "no l_out captured"}
        print(f"[logit lens] 讀取 GGUF 輸出矩陣（{lens.out_t.tensor_type.name}，詞表 {lens.vocab}，"
              f"{'與詞嵌入共用' if lens.tied else '獨立 output.weight'}），對 {len(vecs)} 個 (step,layer) 向量做投影 ...")
        last_pct = [-1]
        def prog(done, total):
            pct = int(done * 100 / total) // 10 * 10
            if pct != last_pct[0]:
                last_pct[0] = pct; print(f"[logit lens] {pct}%", end="\r", flush=True)
        res = lens.compute(np.stack(vecs), topk=topk, progress=prog)
        print("[logit lens] 100%   ")

        def piece(i):
            return _tok_text(llm, i)
        for (si, li), cands in zip(idx, res):
            steps[si]["layers"][li]["logit_lens"] = [
                {"token_id": tid, "text": piece(tid), "prob": round(p, 5), "rank": r}
                for r, (tid, p) in enumerate(cands)]

        # 自我檢查：最後一層的 lens 理論上就是模型真實輸出 → top-1 應該等於實際選出的 token
        agree = total = 0
        for st in steps:
            if st["layers"] and st["layers"][-1]["logit_lens"]:
                total += 1
                agree += st["layers"][-1]["logit_lens"][0]["token_id"] == st["output_token"]["id"]
        info = {"enabled": True, "arch": lens.arch, "weight_type": lens.out_t.tensor_type.name, "tied": lens.tied,
                "softcap": lens.softcap, "eps": lens.eps, "last_layer_top1_matches_output": f"{agree}/{total}"}
        if total and agree < total:
            print(f"[警告] logit lens 自我檢查：最後一層的 top-1 只有 {agree}/{total} 步跟實際輸出一致。")
            print("       正常情況下最後一層的 lens 就是模型輸出，應該全部一致；不一致代表這個架構在最終 norm 前後")
            print("       有我沒處理的額外運算，中間層的 lens 數值請當作參考、不要當精確值。")
        else:
            print(f"[logit lens] 自我檢查通過：最後一層 top-1 與實際輸出一致 {agree}/{total} 步")
        return info
    except Exception as e:
        print(f"[警告] logit lens 計算失敗，略過（其他逐層資料不受影響）：{type(e).__name__}: {e}")
        return {"enabled": False, "reason": f"{type(e).__name__}: {e}"}
