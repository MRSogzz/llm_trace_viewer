# LLM Trace｜開源大型語言模型推論可視化工具

**逐層看見 LLM 如何處理每一個 token：Attention 注意力、Logit Lens、MLP 神經元、殘差流、KV Cache。同時支援 Hugging Face Transformers（safetensors / fp16 / bnb 4-bit）與 llama.cpp（GGUF 量化模型）。**

> **English:** LLM Trace is an open-source LLM interpretability and visualization toolkit. It runs a model, records what happens at every layer and every decoding step (attention weights, logit lens, MLP activations, residual stream norms, KV cache length), and writes a single `run.json` that a standalone HTML viewer renders. Two separate backends: **Hugging Face Transformers** (exact per-layer tensors, safetensors / fp16 / bnb 4-bit) and **llama.cpp** (GGUF quantized models, per-layer data captured through the ggml eval callback, logit lens computed from the GGUF weights).

適用情境：想**視覺化 Transformer attention**、想在 **GGUF / llama.cpp** 模型上看**每一層的 hidden state 與 logit lens**、想理解 **Llama、Qwen、Gemma、Mistral** 等模型「在第幾層決定答案」的 **LLM 可解釋性 (interpretability)** 研究與教學。

---

## 目錄

- [功能特色](#功能特色)
- [兩條後端怎麼選](#兩條後端怎麼選)
- [快速開始](#快速開始)
- [安裝](#安裝)
- [完整命令參考](#完整命令參考)
- [可視化介面](#可視化介面)
- [專案架構](#專案架構)
- [輸出格式 run.json](#輸出格式-runjson)
- [支援的模型架構與驗證狀態](#支援的模型架構與驗證狀態)
- [運作原理](#運作原理)
- [常見問題 FAQ](#常見問題-faq)
- [疑難排解](#疑難排解)
- [已知限制](#已知限制)
- [擴充：新增模型架構](#擴充新增模型架構)

---

## 功能特色

| 功能 | Transformers 後端 | llama.cpp (GGUF) 後端 |
|---|---|---|
| **Attention 注意力**（每層、每個 head） | ✅ | ✅ |
| **Logit Lens**（每一層對下一個 token 的預測） | ✅ | ✅（讀 GGUF 權重自行計算） |
| **MLP 神經元 top-k**（down_proj 前的激活） | ✅ | ✅（僅常見 gated-MLP 節點名） |
| **殘差流範數 (residual norm)** | ✅ | ✅ |
| **KV Cache 長度變化**（prefill / decode） | ✅ | ✅ |
| **每一步輸出機率分佈 top-k** | ✅ | ✅ |
| **Chat 對話模式**（自動套用 chat template） | ✅ | ✅（GGUF 內嵌樣板） |
| **取樣控制**（temperature / top-p / top-k / seed） | ✅ | ❌（固定 greedy） |
| **量化模型** | bnb 4-bit | Q4_K / Q6_K / Q8_0 等 GGUF 量化 |

其他特色：

- **單一輸出格式**：兩個後端都寫出同一種 `run.json`，前端完全不依賴任何特定模型或框架。
- **模型適配層 (Model Adapter)**：不同架構的差異集中在 `model_adapter.py` 一個檔案，新增架構不必改採集邏輯與前端。
- **跨模型**：Llama 系（Llama / Mistral / Qwen / Gemma / Phi / StableLM 等共用 gate-up-down MLP 骨架）與 GPT-2 系。
- **中文友善**：token 顯示避免多位元組字元被切開造成的 `�` 亂碼，並提供整段一次解碼的 `generated_text_decoded` 供核對。
- **單檔前端**：一個 HTML 檔，雙擊即可開啟，不需要伺服器。

---

## 兩條後端怎麼選

| 你手上有什麼 | 用哪條 | 原因 |
|---|---|---|
| HF 格式模型（`config.json` + safetensors），要**精確逐層數值** | **transformers** | 用 PyTorch 直接取得完整張量 |
| **只有 `.gguf` 檔**（Ollama / LM Studio / llama.cpp 常見格式） | **llamacpp** | 直接用 llama.cpp 本體推論，輸出與原工具一致 |

`--backend auto`（預設）會自動判斷：`--model` 是 `.gguf` 檔、是「裡面有 `.gguf` 且沒有 `config.json` 的資料夾」、或有給 `--gguf_file` 時走 llama.cpp，其餘走 transformers。

**為什麼不用 transformers 讀 GGUF？** transformers 讀 GGUF 是把量化權重反量化回 PyTorch 張量，再用自己的 kernel 計算，數值路徑與 llama.cpp 不同，量化模型上容易出現「可視化與原工具輸出對不上」。llama.cpp 後端直接用 llama.cpp 本體推論，畫出的 token 就是它真正選出的 token。

---

## 快速開始

```bash
# 1) 跑一次模型，產生 run.json（Hugging Face 模型，會自動下載）
python run_trace.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "你好，請介紹你自己"

# 2) 或用本機的 GGUF 檔
python run_trace.py --model "D:\models\gemma-Q4_K_M.gguf" --prompt "你好"

# 3) 用瀏覽器開啟 llm_trace_viewer.html，按右上角「載入 run.json」，選 runs/ 底下剛產生的檔案
```

不載入任何檔案時，前端會顯示內建的**示範資料**，可以先熟悉介面。

---

## 安裝

需要 Python 3.10 以上。建議使用虛擬環境：

```bash
python -m venv trace
# Windows:  trace\Scripts\activate        Linux/macOS:  source trace/bin/activate
```

**後端 1（transformers）：**

```bash
pip install torch transformers numpy
pip install bitsandbytes        # 只有要用 --load_in_4bit 才需要
```

**後端 2（llama.cpp / GGUF）：**

```bash
pip install llama-cpp-python gguf jinja2 numpy
pip install huggingface_hub     # 只有要從 HF Hub 下載單一 GGUF 檔才需要
```

- `llama-cpp-python` 沒有現成 wheel 的平台會在本機編譯 llama.cpp（需要 C/C++ 編譯工具與 CMake），第一次安裝可能要數分鐘到十幾分鐘；請依 llama-cpp-python 官方說明處理你的平台。
- `gguf` 用來讀取 GGUF 權重（Logit Lens 需要），`jinja2` 用來渲染 GGUF 內嵌的 chat template。
- 在 Debian / Ubuntu 的**系統** Python（非虛擬環境）安裝時，pip 可能要求加 `--break-system-packages`。

---

## 完整命令參考

```
python run_trace.py --model <模型> --prompt <文字> [選項]
```

### 通用參數（兩個後端）

| 參數 | 預設 | 說明 |
|---|---|---|
| `--model` | 必填 | 後端 1：HF repo id、`models/local/` 下的資料夾名、或本機路徑。後端 2：`.gguf` 檔路徑、放 `.gguf` 的資料夾、或 `models/local/` 下的資料夾名 |
| `--prompt` | 必填 | 輸入文字（chat 模式下作為使用者訊息） |
| `--backend` | `auto` | `auto` / `transformers` / `llamacpp` |
| `--max_new_tokens` | `8` | 生成步數（另外還有 1 步 prefill） |
| `--raw` | 關 | 不套用聊天格式，純續寫 |
| `--system` | 無 | system prompt（需有 chat template 才生效） |
| `--seed` | 無 | 隨機種子 |
| `--models_dir` | `./models` | 模型根目錄（含 `local/` 與 `hf_cache/`） |
| `--runs_dir` | `./runs` | 輸出資料夾 |
| `--out` | 自動 | 指定輸出檔；不指定則存成 `runs/run_<毫秒時間戳>.json` |

### 後端 1（transformers）專用

| 參數 | 預設 | 說明 |
|---|---|---|
| `--device` | `cpu` | 例如 `cuda` |
| `--load_in_4bit` | 關 | 以 bitsandbytes NF4 4-bit 載入 |
| `--layer_stride` | `1` | 每隔幾層取一層（大模型降低資料量） |
| `--head_stride` | `1` | 每隔幾個 head 取一個 |
| `--do_sample` | 關 | 開啟取樣；預設為 greedy |
| `--temperature` | `0.8` | 僅 `--do_sample` 時有效 |
| `--top_p` | `0.95` | 同上 |
| `--top_k` | `40` | 同上 |

### 後端 2（llama.cpp）專用

| 參數 | 預設 | 說明 |
|---|---|---|
| `--gguf_file` | 無 | GGUF 檔名（`--model` 是資料夾或 HF repo id 時使用；資料夾內有多個 `.gguf` 時必填） |
| `--n_ctx` | `4096` | context window；prompt 超過會報錯 |
| `--no_layers` | 關 | 不抓逐層資料，只要 token 與候選詞（走原生 llama.cpp，快很多，並保留 flash attention） |
| `--no_lens` | 關 | 不算 Logit Lens（省下讀取輸出矩陣的時間與記憶體） |
| `--topk_llamacpp` | `8` | 每步輸出候選詞數量 |
| `--layers` | — | 已預設開啟，保留僅為相容舊指令 |

llama.cpp 後端一律 greedy 解碼，`--do_sample` / `--temperature` / `--top_p` / `--top_k` 不會作用於它。

### 常用範例

```bash
# ── 後端 1：transformers ──────────────────────────────
python run_trace.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "你好"             # 預設 chat + greedy
python run_trace.py --model my-local-model --prompt "你好"                         # models/local/my-local-model/
python run_trace.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "Once upon" --raw   # 純續寫
python run_trace.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "你好" \
    --system "你是一位嚴謹的助理" --max_new_tokens 32
python run_trace.py --model Qwen/Qwen2.5-1.5B-Instruct --prompt "你好" \
    --do_sample --temperature 0.7 --top_p 0.9 --top_k 40 --seed 42                  # 取樣並固定種子
python run_trace.py --model my-local-model --prompt "你好" --load_in_4bit --device cuda
python run_trace.py --model my-local-model --prompt "你好" --layer_stride 2 --head_stride 2

# ── 後端 2：llama.cpp / GGUF ─────────────────────────
python run_trace.py --model "D:\models\gemma-Q4_K_M.gguf" --prompt "你好"           # 直接給 .gguf 路徑
python run_trace.py --model "D:\models\gemma-folder" --prompt "你好"                # 資料夾內只有一個 .gguf
python run_trace.py --model my-gguf-dir --gguf_file model-Q4_K_M.gguf --prompt "你好"  # 指定檔名
python run_trace.py --model some-user/some-model-GGUF --gguf_file model-Q4_K_M.gguf \
    --prompt "你好"                                                                   # 從 HF Hub 下載單一 GGUF
python run_trace.py --model "D:\models\gemma-Q4_K_M.gguf" --prompt "你好" --no_layers  # 只要 token 與候選詞（最快）
python run_trace.py --model "D:\models\gemma-Q4_K_M.gguf" --prompt "你好" --no_lens    # 逐層但不算 logit lens
python run_trace.py --model "D:\models\gemma-Q4_K_M.gguf" --prompt "你好" --raw        # 純續寫
python run_trace.py --model "D:\models\gemma-Q4_K_M.gguf" --prompt "..." --n_ctx 8192 --max_new_tokens 64

# ── 強制指定後端 ──────────────────────────────────────
python run_trace.py --backend llamacpp --model "D:\models\x.gguf" --prompt "你好"
python run_trace.py --backend transformers --model my-local-model --prompt "你好"
```

### 模型放置路徑

```
llm_trace/models/
├── local/<模型名>/     自己放的模型（HF 資料夾或含 .gguf 的資料夾）→ --model <模型名>
└── hf_cache/           從 Hugging Face Hub 下載的快取（已導向專案內，不會散落到 ~/.cache）
```

---

## 可視化介面

開啟 `llm_trace_viewer.html`（單檔、離線可用），按「載入 run.json」。

| 面板 | 內容 |
|---|---|
| **Token Strip** | prompt 與生成的 token，點擊選取位置（橘色為生成） |
| **Step Timeline** | prefill 與每個 decode 步，可點選；可自動播放並調速 |
| **Layer / Head 控制** | 切換層與 attention head，所有面板同步更新 |
| **Attention** | 最後一個 query 位置對各 key 的 top-k 注意力，顯示位置、權重與 entropy |
| **Logit Lens** | 該層殘差直接投影到詞表後的 top-k 候選詞 |
| **MLP Top-K** | 該層 down_proj 前激活最強的神經元 |
| **Residual Norm** | 該層殘差流的範數 |
| **候選詞演化** | 橫軸 layer、縱軸 rank、顏色為機率，看答案在第幾層浮現 |
| **KV Cache / 輸出分布** | cache 長度變化與該步輸出機率 top-k |
| **驗證面板** | 整段一次解碼的文字，用來與原工具輸出核對 |

若資料缺少某類內容（例如用了 `--no_layers`），對應面板會說明原因，而不是留白或報錯。

---

## 專案架構

```
llm_trace/
├── run_trace.py            CLI 進入點：解析參數、自動選後端、寫出 run.json
├── model_adapter.py        模型適配層：各架構差異集中於此（後端 1）
├── collector.py            後端 1 採集：手寫 decode loop + output_attentions / hidden_states / pre-hook
├── backend_llamacpp.py     後端 2 採集：llama.cpp 推論、chat 模板、逐層與 logit lens 串接
├── llamacpp_layers.py      後端 2 逐層：ggml eval-callback（讀 attention / MLP / residual 節點）
├── llamacpp_lens.py        後端 2 logit lens：直接讀 GGUF 權重、分塊反量化串流計算
├── llm_trace_viewer.html   前端可視化（讀 run.json，與模型無關）
├── models/{local,hf_cache}/
└── runs/                   輸出的 run.json
```

```
                       run_trace.py  ──►  --backend auto / transformers / llamacpp
                      ┌──────────────────────┴───────────────────────┐
            後端 1: transformers                             後端 2: llama.cpp (GGUF)
      model_adapter.py（偵測架構）                     backend_llamacpp.py（chat 模板 / decode loop）
      collector.py（PyTorch 取值）                     llamacpp_layers.py（ggml 計算圖 callback）
                                                       llamacpp_lens.py（讀 GGUF 權重算 logit lens）
                      └──────────────────────┬───────────────────────┘
                               run.json（統一 schema）
                                             │
                                  llm_trace_viewer.html
```

設計原則：**採集與渲染分離**。模型只跑一次，把需要的資料寫成 `run.json`；前端只讀檔，不重跑模型。

---

## 輸出格式 run.json

```
Run
├─ run_id, model_name, backend, adapter_used
├─ prompt_tokens[], generated_tokens[]          { id, text, position, is_special }
├─ prompt_text_decoded, generated_text_decoded  整段一次解碼的文字（拿來核對）
├─ config                                       解碼設定、模式、限制、config.logit_lens 自我檢查結果
├─ layers_meta[]                                { index, n_heads, d_model, d_head, d_ff }
└─ steps[]                                      每一次前向傳播
   ├─ index, type ("prefill" | "decode")
   ├─ input_tokens[], output_token
   ├─ cache_len_before, cache_len_after, kv_cache_snapshot.length
   ├─ output_distribution[]                     { token_id, text, prob, rank }
   └─ layers[]
      ├─ layer_index, residual_norms[]
      ├─ attn.heads[]                           { head_index, topk_keys[{key_pos, weight}], entropy }
      ├─ mlp                                    { topk_neurons[{neuron_id, activation}], activation_norm }
      └─ logit_lens[]                           { token_id, text, prob, rank }
```

- **Layer 與 Step 是兩個獨立維度**：一次生成 = 多個 step，每個 step 經過所有 layer。
- Prefill 一次處理整段 prompt；decode 每步只處理 1 個新 token，但 attention 看整個 KV cache。
- attention 只保留最後一個 query 位置的 top-k key（不存 O(S²) 全矩陣）；MLP top-k 同樣針對最後位置。
- token 的 `text` 是「token piece」，單看可能不成字（中日韓字常被切開）；要核對輸出請看 `generated_text_decoded`。

---

## 支援的模型架構與驗證狀態

**後端 1（transformers）**

| 架構 | 狀態 |
|---|---|
| Llama | 隨機初始化小模型，greedy 輸出與 `model.generate()` 逐 token 一致 |
| Gemma 4 | 隨機初始化小模型，採集流程跑通（含每層 head_dim 不同的異質層設定）；未與 `generate()` 比對 |
| GPT-2 | 隨機初始化小模型，採集流程跑通；未與 `generate()` 比對 |
| Mistral / Qwen2·3 / Gemma 2·3 / Phi-3 / StableLM | **預期可用但未逐一測試**：模組結構同為 `gate_proj / up_proj / down_proj`，會被同一個 adapter 認出 |
| MoE（Mixtral 等） | **不支援**，載入時會報明確錯誤，需另寫 adapter |

**後端 2（llama.cpp）**

- 已用手動組出的 llama 架構 GGUF 驗證：F32、Q8_0、Q4_0 輸出矩陣、共用詞嵌入（無 `output.weight`）、帶 Jinja2 chat template 的檔案。
- 逐層節點名稱來自 llama.cpp 原始碼（`kq_soft_max`、`l_out`、`ffn_swiglu` 等，由共用的 attention / FFN 建圖函式產生）。
- **尚未用真實 Gemma 4 GGUF 驗證。**

---

## 運作原理

**後端 1：** 不用 `model.generate()`，而是手寫 decode loop，讓每次前向傳播都對應一個 step。attention 與每層 hidden state 來自 HF 標準的 `output_attentions` / `output_hidden_states`（因此需要 `attn_implementation="eager"`，已內建）；MLP 激活用 `forward_pre_hook` 掛在 `down_proj` 前；Logit Lens 是把各層 hidden state 套最終 norm 後乘 `lm_head`。

**後端 2：** llama.cpp 沒有 PyTorch 那樣的 hook，但底層 ggml 有等價機制 `ggml_backend_sched_eval_callback`（llama.cpp 官方 `llama-eval-callback` 除錯工具使用的同一個）。每個計算圖節點算完會通知 callback，程式用 `ggml_backend_tensor_get` 把資料讀回：

| 資料 | ggml 節點 |
|---|---|
| attention 機率 | `kq_soft_max-{層}` |
| MLP 激活 | `ffn_swiglu-{層}`（GELU / ReLU 版為 `ffn_geglu` / `ffn_reglu`） |
| 殘差輸出 | `l_out-{層}` |

Logit Lens 需要最終 norm 權重與輸出矩陣，這是靜態權重而非計算圖節點，所以直接用 `gguf` 套件讀取，以「分塊反量化、一次掃過詞表」的串流算法計算（不需把約 2.7GB 的 float32 矩陣整個展開），並套用 GGUF metadata 裡的 `final_logit_softcapping`（Gemma 系）。這段的串流算法已與直接計算比對一致，但**沒有跟真實 Gemma 模型的輸出比對過**。程式會用「最後一層的 lens 應等於模型真實輸出」做自我檢查，並印出一致率。

---

## 常見問題 FAQ

### 如何視覺化 LLM 每一層的 attention？
執行 `python run_trace.py --model <模型> --prompt "..."` 產生 `run.json`，用 `llm_trace_viewer.html` 載入，切換 Layer / Head 即可。每個 head 顯示最後一個 query 對各 key 的 top-k 注意力與 entropy。

### GGUF / llama.cpp 量化模型可以看 attention 嗎？
可以。llama.cpp 後端預設就抓逐層資料（透過 ggml eval callback）。代價是為了讓 `kq_soft_max` 節點存在，會強制關閉 flash attention，速度比一般 llama.cpp 慢；只要 token 與候選詞的話加 `--no_layers`。

### 什麼是 Logit Lens？怎麼在 llama.cpp 上用？
Logit Lens 是把第 N 層的殘差直接當成最終隱藏狀態，套最終 norm 並乘輸出矩陣，看「模型若在這一層就結束會預測什麼」，用來觀察答案在第幾層浮現。llama.cpp 後端預設開啟（`--no_lens` 可關），權重從 GGUF 檔讀取計算。

### 怎麼看 KV Cache 的變化？
每個 step 記錄 `cache_len_before` / `cache_len_after`，前端「KV Cache」面板顯示長度變化（prefill 一次寫入整段 prompt，之後每步 +1）。目前**只記錄長度**，不記錄各層 K/V 向量本身。

### 為什麼輸出是續寫而不是對話？
沒有套用聊天格式。預設會自動套用 chat template；若你的 GGUF 沒有內嵌 `tokenizer.chat_template`，程式會印警告並退回續寫。也檢查是否誤加了 `--raw`。

### 為什麼可視化結果和我原本工具的輸出不一樣？
常見原因：(1) 原工具用取樣、這裡是 greedy（transformers 後端可用 `--do_sample` 與相同參數/種子對齊）；(2) 沒套用同樣的 chat template 或 system prompt；(3) 量化與非量化模型本來就會有不同輸出；(4) 逐個 token 解碼中文會亂碼，請看驗證面板的 `generated_text_decoded`。

### 支援 Gemma 4、Qwen、Llama、Mistral 嗎？
見[支援的模型架構與驗證狀態](#支援的模型架構與驗證狀態)。Llama、Gemma 4、GPT-2 用隨機初始化小模型驗證過採集流程；其他同骨架架構預期可用但未逐一測試；

### 需要 GPU 嗎？
不需要，CPU 可跑（`--device` 預設 `cpu`）。GPU速度更快。

### 量化模型上的 Logit Lens 準確嗎？
以合成小模型驗證：F32 與 Q4_0 末層與 llama.cpp 實際輸出一致；Q8_0 機率誤差約 1e-6 量級，機率幾乎並列的項目可能換位。另外程式每次都會自我檢查末層 top-1 是否等於實際輸出。

---

## 疑難排解

| 現象 | 原因與做法 |
|---|---|
| `401 ... Repository Not Found`，`--model` 填的是 `.gguf` 檔名 | GGUF 不是 HF repo。給檔案的**完整路徑**，或給放 `.gguf` 的資料夾，或 `--model <資料夾> --gguf_file <檔名>` |
| 卡在 `import transformers`、走錯後端 | 確認 `--model` 指向 GGUF 檔或含 `.gguf`（且無 `config.json`）的資料夾；也可加 `--backend llamacpp` |
| `找不到 GGUF 檔案` | 資料夾內有多個 `.gguf` 時要加 `--gguf_file`；或路徑有誤 |
| 逐層採集初始化失敗（找不到 ggml 函式庫） | 程式會印出搜尋資料夾與內容並**自動退回不含逐層的模式**；把訊息貼給開發者，或加 `--no_layers` |
| `ggml_tensor 結構大小 ... 對不上` | 你的 llama-cpp-python 內建 ggml 版本與程式預期不同（避免讀到錯位記憶體而中止）。回報 `pip show llama-cpp-python` 版本，或先加 `--no_layers` |
| 逐層資料是空的（「一個節點都沒抓到」） | 該模型節點名稱不同。程式會印出計算圖實際出現的節點，也寫入 `run.json` 的 `config.debug_seen_nodes`，貼出即可補上 |
| Logit Lens 自我檢查警告 | 最後一層 top-1 與實際輸出不一致，代表該架構在最終 norm 前後有未處理的運算；中間層數值僅供參考 |
| 中文 token 顯示成 `<0xE4><0xBD>` | 該 token 只是不完整的 UTF-8 位元組，屬正常；完整文字看 `generated_text_decoded` |
| 前端顯示「這次是不含逐層的模式」 | 用了 `--no_layers` 或載入了舊版 `run.json`，重跑即可 |
| Logit Lens 計算很久 | 詞表大時需掃過整個輸出矩陣，會顯示進度；不需要就加 `--no_lens` |

---

## 已知限制

- **llama.cpp 逐層採集**：強制關閉 flash attention（較慢）；只支援 f32 / f16 的中間張量；MLP 僅認 gated-MLP 的常見節點名，MoE 分支不會被抓到；靠 monkeypatch `llama_cpp.llama_cpp.llama_context_default_params` 注入 callback，llama-cpp-python 版本差異可能使其失效（載入時有結構大小檢查與自動退回）。
- **Windows / macOS 的 ggml 函式庫載入**：檔名與 DLL 搜尋邏輯是照 llama-cpp-python 自己的載入器寫的，**開發環境為 Linux，這兩個平台未實機測過**。
- **llama.cpp 後端**：僅 greedy；`--no_layers` 模式下不記錄各步的輸入 token（`input_tokens` 為空），逐層模式則有記錄。
- **兩個後端**：attention 與 MLP 只保留最後一個 query 位置的 top-k；KV Cache 只記錄長度。
- **transformers 後端**：MoE 未支援；真實 HF 模型下載、bnb 4-bit、GPU 路徑未實測。

---

## 擴充：新增模型架構

只有 `model_adapter.py` 需要為新架構寫程式碼（通常 20~30 行）。新增一個 `ModelAdapter` 子類別，實作：

```python
layers()            # 各層 decoder layer 模組
mlp_down_proj(l)    # MLP 輸出投影模組（用來抓 pre-down_proj 激活）
final_norm(h)       # 套用最終 norm
lm_head(h)          # 投影到詞表
embed_tokens(ids)   # 詞嵌入
info()              # 層數、head 數、hidden 大小等
```

再到 `ModelAdapter.detect()` 加一個判斷分支。`collector.py`、`run_trace.py`、前端都不用改。llama.cpp 後端不需要 adapter，節點名稱由 llama.cpp 共用的建圖函式決定。

---

**關鍵字：** LLM visualization · LLM interpretability · transformer attention visualization · logit lens · llama.cpp · GGUF · Hugging Face Transformers · safetensors · KV cache · Gemma · Qwen · Llama · Mistral · 大型語言模型可視化 · 注意力視覺化 · 模型可解釋性