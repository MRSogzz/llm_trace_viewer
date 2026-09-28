"""
llama.cpp 後端的 logit lens
===========================
logit lens = 把「第 l 層之後的殘差」當成最終隱藏狀態，套最終 norm、乘輸出矩陣，看模型
「如果在這一層就結束」會預測什麼。

需要的兩樣東西是模型的靜態權重（output_norm.weight、output.weight；沒有 output.weight
時輸出矩陣與詞嵌入共用，token_embd.weight），不是計算圖節點，ggml eval-callback 拿不到，
所以這裡直接用 `gguf` 套件從 GGUF 檔案讀出來，自己算。
每層的殘差向量來自 callback 抓到的 `l_out-{層}`。

實作重點
- 輸出矩陣通常是量化的（Q4_K、Q6_K…）而且很大（詞表 26 萬 × 隱藏維度 2560 展開成 float32 約 2.7GB），
  所以用「分塊反量化、一次掃過整個詞表」的串流算法：每塊反量化後立刻跟「全部 (step,layer) 向量」
  做矩陣乘，同時維護每個向量的 running top-k 與 log-sum-exp（softmax 分母），不需要整個矩陣同時在記憶體。
- 最終 norm：有 output_norm.bias 視為 LayerNorm，否則 RMSNorm（epsilon 來自 GGUF metadata）。
  Gemma 系的 norm 權重在轉檔時已經把 +1 燒進去，所以直接相乘即可，跟 llama.cpp 的算法一致。
- 有 final_logit_softcapping（Gemma2/3/4）時套用 cap * tanh(logits / cap)，機率才會跟實際輸出一致。
- 自我檢查：最後一層的 lens 理論上就是模型的真實輸出，程式會拿它跟實際輸出 token 比對並回報一致率。
"""
import numpy as np


def _field(reader, key, default=None):
    f = reader.fields.get(key)
    if f is None:
        return default
    try:
        v = f.parts[f.data[0]]
        if v.dtype == np.uint8:  # 字串
            return bytes(v).decode("utf-8", "ignore")
        return v.item() if v.size == 1 else v
    except Exception:
        return default


class LogitLens:
    def __init__(self, gguf_path, chunk_rows=4096):
        import gguf
        from gguf import quants

        self._quants = quants
        self.chunk_rows = chunk_rows
        r = gguf.GGUFReader(gguf_path)
        arch = _field(r, "general.architecture", "llama")
        by_name = {t.name: t for t in r.tensors}

        if "output_norm.weight" not in by_name:
            raise RuntimeError("GGUF 裡找不到 output_norm.weight，無法做 logit lens")
        norm_t = by_name["output_norm.weight"]
        self.norm_w = self._to_f32(norm_t).reshape(-1)
        self.norm_b = self._to_f32(by_name["output_norm.bias"]).reshape(-1) if "output_norm.bias" in by_name else None

        out_t = by_name.get("output.weight") or by_name.get("token_embd.weight")
        if out_t is None:
            raise RuntimeError("GGUF 裡找不到 output.weight 也找不到 token_embd.weight")
        self.tied = "output.weight" not in by_name
        self.out_t = out_t
        self.hidden = int(out_t.shape[0])
        self.vocab = int(out_t.shape[1])

        self.eps = float(_field(r, f"{arch}.attention.layer_norm_rms_epsilon",
                                _field(r, f"{arch}.attention.layer_norm_epsilon", 1e-5)))
        cap = _field(r, f"{arch}.final_logit_softcapping", None)
        self.softcap = float(cap) if cap not in (None, 0, 0.0) else None
        self.arch = arch
        self._reader = r  # 保持 memmap 存活

    def _to_f32(self, t):
        return self._quants.dequantize(t.data, t.tensor_type).astype(np.float32) \
            if t.tensor_type.name not in ("F32",) else np.asarray(t.data, dtype=np.float32)

    def _final_norm(self, H):
        H = H.astype(np.float32)
        if self.norm_b is not None:  # LayerNorm
            H = H - H.mean(axis=1, keepdims=True)
            var = (H * H).mean(axis=1, keepdims=True)
        else:                        # RMSNorm
            var = (H * H).mean(axis=1, keepdims=True)
        H = H / np.sqrt(var + self.eps) * self.norm_w
        if self.norm_b is not None:
            H = H + self.norm_b
        return H

    def _rows(self, start, stop):
        """取輸出矩陣的第 start..stop 列（每列 = 一個詞的 hidden 維向量）並反量化成 float32。"""
        raw = self.out_t.data[start:stop]
        if self.out_t.tensor_type.name == "F32":
            return np.asarray(raw, dtype=np.float32)
        return self._quants.dequantize(raw, self.out_t.tensor_type).astype(np.float32)

    def compute(self, H, topk=8, progress=None):
        """
        H: [N, hidden] 一堆殘差向量（每個 (step, layer) 一條）。
        回傳 N 個候選詞清單，每個是 [(token_id, prob), ...] 依機率由大到小。
        """
        N = H.shape[0]
        Hn = self._final_norm(H)                      # [N, hidden]
        k = min(topk, self.vocab)
        best_v = np.full((N, k), -np.inf, dtype=np.float32)
        best_i = np.zeros((N, k), dtype=np.int64)
        run_max = np.full(N, -np.inf, dtype=np.float64)
        run_sum = np.zeros(N, dtype=np.float64)

        for start in range(0, self.vocab, self.chunk_rows):
            stop = min(start + self.chunk_rows, self.vocab)
            W = self._rows(start, stop)               # [rows, hidden]
            L = W @ Hn.T                              # [rows, N]
            if self.softcap:
                L = self.softcap * np.tanh(L / self.softcap)
            # log-sum-exp 串流更新
            cmax = L.max(axis=0).astype(np.float64)
            new_max = np.maximum(run_max, cmax)
            run_sum = run_sum * np.exp(run_max - new_max) + np.exp(L - new_max).sum(axis=0)
            run_max = new_max
            # running top-k：先取本塊每欄的 top-k，再跟既有的合併
            kk = min(k, L.shape[0])
            idx = np.argpartition(-L, kk - 1, axis=0)[:kk]                  # [kk, N]
            vals = np.take_along_axis(L, idx, axis=0).T                     # [N, kk]
            ids = (idx + start).T                                           # [N, kk]
            cv = np.concatenate([best_v, vals], axis=1)
            ci = np.concatenate([best_i, ids], axis=1)
            sel = np.argpartition(-cv, k - 1, axis=1)[:, :k]
            best_v = np.take_along_axis(cv, sel, axis=1)
            best_i = np.take_along_axis(ci, sel, axis=1)
            if progress:
                progress(stop, self.vocab)

        out = []
        for n in range(N):
            order = np.argsort(-best_v[n])
            probs = np.exp(best_v[n][order].astype(np.float64) - run_max[n]) / run_sum[n]
            out.append([(int(best_i[n][o]), float(p)) for o, p in zip(order, probs)])
        return out
