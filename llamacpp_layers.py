"""
後端 2 的逐層資料採集：ggml eval-callback
============================================
llama.cpp 沒有 PyTorch 那種 forward hook，但 ggml（llama.cpp 的底層計算函式庫）
本身有一個對等機制：`ggml_backend_sched_eval_callback`。這是 llama.cpp 官方
`llama-eval-callback`範例debug工具用的同一個機制——每個計算圖節點算完之後，
排程器都會呼叫這個callback一次，把該節點的 ggml_tensor 指標交給你，你可以用
`ggml_backend_tensor_get()` 把資料讀回host記憶體。這就是 backend2 版本的
「hook」，只是掛在 ggml 計算圖節點上，不是 PyTorch nn.Module 上。

節點命名（來自 llama.cpp 原始碼 src/llama-graph.cpp 與 src/models/*.cpp，
這些命名是所有走共用 build_attn_mha() / build_ffn() 的架構都會有的，
不是特定模型才有）：
  "kq_soft_max-{il}"   attention softmax後的機率矩陣 (跟後端1的 attn.heads 對應)
  "ffn_gate_par-{il}"  gated MLP 在 down_proj 之前的 activation (跟後端1的 mlp.topk_neurons 對應)
  "l_out-{il}"         該層最終殘差輸出 (跟後端1的 residual_norms 對應)

拿不到的東西，誠實列出：
  - logit lens：需要模型的 output_norm/output 權重張量做投影，這兩個是
    「持久權重」而不是計算圖節點，這個callback機制拿不到，目前沒做
    （可以另外直接讀gguf檔案本身的權重來算，是可行的擴充方向，但還沒實作）
  - MoE / 非 gated-MLP 架構節點命名可能不同，目前只認 build_ffn() 的
    "ffn_gate_par" 這個共用節點名
"""
import ctypes
import math
import os
import re
import sys
import time

import numpy as np

GGML_MAX_DIMS = 4
GGML_MAX_OP_PARAMS = 64
GGML_MAX_SRC = 10
GGML_MAX_NAME = 64

GGML_TYPE_F32 = 0
GGML_TYPE_F16 = 1

WANT_PREFIXES = (
    "kq_soft_max-", "l_out-",
    # MLP「pre-down_proj」activation，依 llm.cpp 用的activation函式不同，
    # 節點名稱也不同（都是 build_ffn() 這個共用函式產生的，只是走不同分支）：
    "ffn_swiglu_limited-", "ffn_swiglu_oai-", "ffn_swiglu-",  # SiLU-gated (Llama/Mistral/Qwen/Gemma系列最常見)
    "ffn_geglu-", "ffn_reglu-",                                # GELU/ReLU-gated
    "ffn_gate_par-",                                           # 少數走「先分別算gate/up，最後才相乘」路徑的架構
)
MLP_TAGS = WANT_PREFIXES[2:]


class _GgmlTensor(ctypes.Structure):
    pass


_GgmlTensor._fields_ = [
    ("type", ctypes.c_int),
    ("buffer", ctypes.c_void_p),
    ("ne", ctypes.c_int64 * GGML_MAX_DIMS),
    ("nb", ctypes.c_size_t * GGML_MAX_DIMS),
    ("op", ctypes.c_int),
    ("op_params", ctypes.c_int32 * (GGML_MAX_OP_PARAMS // 4)),
    ("flags", ctypes.c_int32),
    ("src", ctypes.POINTER(_GgmlTensor) * GGML_MAX_SRC),
    ("view_src", ctypes.POINTER(_GgmlTensor)),
    ("view_offs", ctypes.c_size_t),
    ("data", ctypes.c_void_p),
    ("name", ctypes.c_char * GGML_MAX_NAME),
    ("extra", ctypes.c_void_p),
    ("padding", ctypes.c_char * 8),
]


def _ggml_lib_candidates(lib_dir, platform=None):
    """依平台列出可能放著 ggml_* 符號的動態函式庫檔名（Windows 是 .dll 沒有 lib 前綴）。
    新版 llama-cpp-python 把 ggml 拆成 ggml-base；舊版可能在 ggml 或 llama 裡，都試。"""
    platform = platform or sys.platform
    names = []
    for base in ("ggml-base", "ggml", "llama"):
        if platform == "win32":
            names += [f"{base}.dll", f"lib{base}.dll"]
        elif platform == "darwin":
            names += [f"lib{base}.dylib", f"lib{base}.so"]
        else:
            names += [f"lib{base}.so"]
    return [os.path.join(lib_dir, n) for n in names]


def _load_ggml_lib():
    import llama_cpp
    lib_dir = os.path.join(os.path.dirname(llama_cpp.__file__), "lib")
    if sys.platform == "win32":
        # 跟 llama-cpp-python 自己載入時一樣，要先把資料夾加進 DLL 搜尋路徑，
        # 不然 ggml-base.dll 依賴的其他 dll（ggml-cpu.dll 等）找不到
        os.add_dll_directory(lib_dir)
        os.environ["PATH"] = lib_dir + os.pathsep + os.environ.get("PATH", "")
    tried, lib = [], None
    for path in _ggml_lib_candidates(lib_dir):
        if not os.path.isfile(path):
            continue
        try:
            cand = ctypes.CDLL(path, winmode=ctypes.RTLD_GLOBAL) if sys.platform == "win32" else ctypes.CDLL(path)
        except OSError as e:
            tried.append(f"{path}: {e}")
            continue
        if hasattr(cand, "ggml_backend_tensor_get"):
            lib = cand
            break
        tried.append(f"{path}: 沒有匯出 ggml_backend_tensor_get")
    if lib is None:
        listing = sorted(os.listdir(lib_dir)) if os.path.isdir(lib_dir) else "(資料夾不存在)"
        raise RuntimeError(
            "找不到含 ggml_backend_tensor_get 的動態函式庫。\n"
            f"  搜尋資料夾: {lib_dir}\n  資料夾內容: {listing}\n  嘗試結果: {tried}\n"
            "  請把以上貼給開發者。")
    lib.ggml_backend_tensor_get.argtypes = [ctypes.POINTER(_GgmlTensor), ctypes.c_void_p, ctypes.c_size_t, ctypes.c_size_t]
    lib.ggml_backend_tensor_get.restype = None
    lib.ggml_nbytes.argtypes = [ctypes.POINTER(_GgmlTensor)]
    lib.ggml_nbytes.restype = ctypes.c_size_t
    _check_struct_layout(lib)
    return lib


def _check_struct_layout(lib):
    """ggml_tensor 的欄位配置會隨 ggml 版本變動；我的 ctypes 結構是照特定版本的 ggml.h 寫的。
    ggml_tensor_overhead() = sizeof(ggml_object) + sizeof(ggml_tensor)，sizeof(ggml_object)
    在 64 位元平台固定 32 bytes，可以反推出這個 ggml 實際的 sizeof(ggml_tensor)，
    跟我們的結構比對；不一致就直接報錯，避免讀到錯位的記憶體產生亂七八糟的結果甚至崩潰。"""
    try:
        lib.ggml_tensor_overhead.restype = ctypes.c_size_t
        real = lib.ggml_tensor_overhead() - 32
    except AttributeError:
        return  # 拿不到就跳過檢查
    mine = ctypes.sizeof(_GgmlTensor)
    if real != mine:
        raise RuntimeError(
            f"你安裝的 llama-cpp-python 內建的 ggml 版本，ggml_tensor 結構大小是 {real} bytes，"
            f"但本程式是照 {mine} bytes 的版本寫的，欄位配置對不上，強行讀取會讀到錯位資料。\n"
            "  請把 `pip show llama-cpp-python` 的版本貼給開發者，我照那個版本的 ggml.h 調整結構。\n"
            "  暫時想避開這個問題可以加 --no_layers（只要 token 與候選詞，不抓逐層）。")


def _parse_layer(name: str):
    for p in WANT_PREFIXES:
        if name.startswith(p):
            tail = name[len(p):]
            try:
                return p[:-1], int(tail)
            except ValueError:
                return p[:-1], -1
    return None, -1


class LayerCapture:
    """一次 llm.eval(tokens) 呼叫期間，把感興趣的節點抓下來存這裡。"""

    def __init__(self):
        self.data = {}  # (base_name, layer_idx) -> (numpy array, ne tuple)
        self.seen = {}  # 計算圖裡實際出現過的節點名稱(去掉層號) -> 次數，診斷用
        self._lib = _load_ggml_lib()
        self._callback_holder = None  # 要留著參照，不然 ctypes callback 會被 GC 回收

    def clear(self):
        self.data = {}

    def make_callback(self):
        import llama_cpp

        def _cb(t_ptr, ask, user_data):
            tensor = ctypes.cast(t_ptr, ctypes.POINTER(_GgmlTensor)).contents
            raw_name = bytes(tensor.name).split(b"\x00")[0]
            name = raw_name.decode("utf-8", "ignore")
            if ask and len(self.seen) < 400:
                key = re.sub(r"-\d+$", "", name)
                self.seen[key] = self.seen.get(key, 0) + 1
            base, il = _parse_layer(name)
            if base is None:
                return False
            if ask:
                return True
            nbytes = self._lib.ggml_nbytes(ctypes.byref(tensor))
            buf = (ctypes.c_uint8 * nbytes)()
            self._lib.ggml_backend_tensor_get(ctypes.byref(tensor), buf, 0, nbytes)
            raw = bytes(buf)
            if tensor.type == GGML_TYPE_F32:
                arr = np.frombuffer(raw, dtype=np.float32)
            elif tensor.type == GGML_TYPE_F16:
                arr = np.frombuffer(raw, dtype=np.float16).astype(np.float32)
            else:
                return True  # 量化型別的中間張量很少見，暫不支援，跳過不報錯
            ne = tuple(int(tensor.ne[i]) for i in range(GGML_MAX_DIMS))
            self.data[(base, il)] = (arr.copy(), ne)
            return True

        self._callback_holder = llama_cpp.ggml_backend_sched_eval_callback(_cb)
        return self._callback_holder


def make_llama_with_layer_capture(gguf_path, n_ctx=4096, seed=0, capture: "LayerCapture | None" = None):
    """
    建一個 Llama 物件，但注入 cb_eval callback 讓 ggml 排程器在每個計算圖節點
    算完之後都通知我們一次。做法是暫時 monkeypatch
    llama_cpp.llama_context_default_params，讓 Llama.__init__ 內部拿到的
    「預設參數」裡就已經帶著我們的 callback（Llama.__init__ 本身沒有任何
    參數可以直接傳 cb_eval，這是目前版本唯一乾淨的注入點）。
    同時強制關閉 flash attention：flash attention kernel 會把
    QK^T-softmax-V 融合成一個運算，"kq_soft_max" 節點就不會存在，
    行為上跟後端1要求 attn_implementation="eager" 是同一個道理。
    """
    import llama_cpp
    import llama_cpp.llama_cpp as _capi  # llama.py 內部實際呼叫的是這個子模組，
    # 不是頂層 package 重新匯出的那個名字，兩者是不同的物件參照，只 patch 頂層的話
    # llama.py 內部呼叫到的還是原本沒被改過的函式，完全沒用。

    if capture is None:
        capture = LayerCapture()
    cb = capture.make_callback()

    original_default_params = _capi.llama_context_default_params

    def patched_default_params():
        p = original_default_params()
        p.cb_eval = cb
        p.cb_eval_user_data = None
        if hasattr(p, "flash_attn_type"):
            p.flash_attn_type = 0  # LLAMA_FLASH_ATTN_TYPE_DISABLED
        return p

    _capi.llama_context_default_params = patched_default_params
    try:
        llm = llama_cpp.Llama(model_path=gguf_path, n_ctx=n_ctx, n_batch=n_ctx,
                               logits_all=True, verbose=False, seed=seed)
    finally:
        _capi.llama_context_default_params = original_default_params

    return llm, capture


def layers_from_capture(capture: LayerCapture, n_query_positions: int, topk_neurons=8, topk_attn_keys=8,
                         valid_kv_len=None, keep_hidden=False):
    """
    valid_kv_len: 這次 eval() 之後「實際」的 KV cache 長度（例如 llm.n_tokens）。
    ggml 為了對齊/效能，kq_soft_max 張量的 key 維度常常會被 padding 到比實際
    cache長度更大的邊界，多出來的位置是無意義的殘留資料，不能拿來當「真的
    attention 分佈」呈現，用這個參數裁掉超出實際長度的部分。

    把這次 eval() 呼叫抓到的原始 ggml tensor 轉成跟後端1同樣的 LayerTrace schema。
    n_query_positions: 這次 eval() 處理了幾個新位置（prefill時是整個prompt長度，
    decode時通常是1），用來把攤平的 buffer 正確 reshape 回 [n_head, n_query, n_key] 等形狀。
    """
    layer_indices = sorted({il for (_, il) in capture.data.keys() if il >= 0})
    layers = []
    for il in layer_indices:
        entry = {"layer_index": il, "residual_norms": [], "attn": {"heads": []},
                 "mlp": {"topk_neurons": [], "activation_norm": None}, "logit_lens": []}

        if ("kq_soft_max", il) in capture.data:
            arr, ne = capture.data[("kq_soft_max", il)]
            n_kv, n_q, n_head = ne[0], ne[1], ne[2]
            try:
                mat = arr.reshape(n_head, n_q, n_kv)
            except ValueError:
                mat = None
            if mat is not None:
                heads = []
                for h in range(n_head):
                    row = mat[h, -1]  # 這次eval()裡「最後一個新位置」對key的注意力權重
                    if valid_kv_len is not None:
                        row = row[:valid_kv_len]  # 裁掉 ggml padding 出來的無效位置
                    k = min(topk_attn_keys, row.shape[0])
                    idx = np.argsort(-row)[:k]
                    p = np.clip(row, 1e-12, None)
                    entropy = float(-(p * np.log(p)).sum())
                    heads.append({
                        "head_index": h,
                        "topk_keys": [{"key_pos": int(i), "weight": round(float(row[i]), 5)} for i in idx],
                        "entropy": round(entropy, 4),
                    })
                entry["attn"]["heads"] = heads

        if ("l_out", il) in capture.data:
            arr, ne = capture.data[("l_out", il)]
            hidden, n_tok = ne[0], ne[1]
            try:
                mat = arr.reshape(n_tok, hidden)
                entry["residual_norms"] = [round(float(np.linalg.norm(mat[i])), 3) for i in range(n_tok)]
                if keep_hidden:
                    entry["_hidden"] = mat[-1].copy()  # 最後位置的殘差向量，logit lens 用（之後會移除，不進 json）
            except ValueError:
                pass

        mlp_tag = next((t for t in MLP_TAGS if (t.rstrip("-"), il) in capture.data), None)
        if mlp_tag:
            arr, ne = capture.data[(mlp_tag.rstrip("-"), il)]
            d_ff, n_tok = ne[0], ne[1]
            try:
                mat = arr.reshape(n_tok, d_ff)
                last = mat[-1]
                k = min(topk_neurons, last.shape[0])
                idx = np.argsort(-np.abs(last))[:k]
                entry["mlp"]["topk_neurons"] = [
                    {"neuron_id": int(i), "activation": round(float(last[i]), 4)} for i in idx
                ]
                entry["mlp"]["activation_norm"] = round(float(np.linalg.norm(last)), 3)
            except ValueError:
                pass

        layers.append(entry)
    return layers
