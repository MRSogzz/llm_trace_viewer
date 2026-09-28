r"""
兩條後端，明確分開，不混用：

  後端 1 --backend transformers   給要精確逐層數值(attention/MLP/logit lens)的人
      支援 HF repo id / 本地 safetensors 資料夾 / bnb 4-bit 量化
      範例：
        python run_trace.py --backend transformers \
            --model Qwen/Qwen2.5-1.5B-Instruct --prompt "你好"
        python run_trace.py --backend transformers \
            --model my-local-model --load_in_4bit --prompt "你好"

  後端 2 --backend llamacpp       給只有 GGUF 檔的人
      直接呼叫 llama.cpp 本體推論，保證輸出跟你原本工具一致；
      代價是拿不到逐層 attention/MLP 內部資料（llama.cpp 不暴露這些）。
      範例：
        python run_trace.py --backend llamacpp \
            --model "D:\path\to\model-Q4_K_M.gguf" --prompt "你好"
        python run_trace.py --backend llamacpp \
            --model my-model --gguf_file model-Q4_K_M.gguf --prompt "你好"

不給 --backend 時會自動判斷：--model 或 --gguf_file 指向 .gguf -> llamacpp，否則 transformers。
"""
import argparse
import json
import os
import time

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODELS_DIR = os.path.join(BASE_DIR, "models")
# 必須在 import transformers 之前設，把 HF Hub 下載快取導向專案內，
# 不要散落到使用者家目錄的 ~/.cache/huggingface。
os.environ.setdefault("HF_HOME", os.path.join(DEFAULT_MODELS_DIR, "hf_cache"))


# ---------- 後端自動判斷 ----------

def looks_like_gguf_target(model_arg: str, gguf_file_arg: str, models_dir: str) -> bool:
    """
    判斷這次呼叫應該走哪條後端。原本只認「--model 本身以 .gguf 結尾」或
    「有給 --gguf_file」，但使用者常見的用法是直接把 --model 指到「裝著
    .gguf 檔的資料夾」（沒有副檔名），這種情況原本會被誤判成 transformers，
    導致去 import transformers、嘗試當成 HF repo 處理，一路卡住或報怪錯誤。
    這裡多檢查：--model 若是資料夾，裡面有沒有 .gguf 檔、有沒有 config.json
    （HF 格式模型一定有這個檔，GGUF 資料夾通常沒有）。
    """
    if gguf_file_arg or model_arg.lower().endswith(".gguf"):
        return True
    candidates = [model_arg, os.path.join(models_dir, "local", model_arg)]
    for d in candidates:
        if os.path.isdir(d):
            has_config = os.path.isfile(os.path.join(d, "config.json"))
            has_gguf = any(f.lower().endswith(".gguf") for f in os.listdir(d))
            if has_gguf and not has_config:
                return True
    return False


# ---------- 後端 1：transformers ----------

def resolve_transformers_model(model_arg: str, models_dir: str) -> str:
    if model_arg.lower().endswith(".gguf"):
        raise SystemExit(
            f"{model_arg} 是 GGUF 檔，請改用 --backend llamacpp（或不指定 --backend 讓程式自動判斷），"
            "這個後端(transformers)故意不處理 GGUF。"
        )
    if os.path.isdir(model_arg):
        has_config = os.path.isfile(os.path.join(model_arg, "config.json"))
        has_gguf = any(f.lower().endswith(".gguf") for f in os.listdir(model_arg))
        if has_gguf and not has_config:
            raise SystemExit(
                f"{model_arg} 裡面是 GGUF 檔、沒有 config.json，這不是 HF 格式模型。"
                " 請改用 --backend llamacpp（或不指定 --backend 讓程式自動判斷）。"
            )
        return model_arg
    local_candidate = os.path.join(models_dir, "local", model_arg)
    if os.path.isdir(local_candidate) and os.listdir(local_candidate):
        print(f"[models] 使用本地模型: {local_candidate}")
        return local_candidate
    print(f"[models] 視為 HF repo id，快取到: {os.path.join(models_dir, 'hf_cache')}")
    return model_arg


def run_transformers_backend(args):
    # 先做路徑檢查，確認不是誤把 GGUF 資料夾當成 HF 模型，
    # 避免在正式 import transformers（有時候要幾十秒甚至更久）之後才報錯。
    model_path = resolve_transformers_model(args.model, args.models_dir)

    from transformers import AutoModelForCausalLM, AutoTokenizer
    from collector import collect_trace

    print(f"[1/3] 載入 tokenizer / model: {model_path}")

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    load_kwargs = {"attn_implementation": "eager", "torch_dtype": "auto"}
    if args.load_in_4bit:
        from transformers import BitsAndBytesConfig
        load_kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype="float16", bnb_4bit_quant_type="nf4"
        )
        print("[models] 使用 bnb 4-bit 量化載入")
    model = AutoModelForCausalLM.from_pretrained(model_path, **load_kwargs)

    prompt = args.prompt
    if not args.raw and getattr(tokenizer, "chat_template", None):
        messages = []
        if args.system:
            messages.append({"role": "system", "content": args.system})
        messages.append({"role": "user", "content": args.prompt})
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        print("[prompt] 偵測到 chat_template，已自動套用（跟大多數聊天前端行為一致；"
              "如果你原本是拿這個模型當純續寫用，請加 --raw）")

    print(f"[2/3] 執行 prefill + {args.max_new_tokens} 步 decode 並採集 trace ...")
    run = collect_trace(
        model, tokenizer, prompt,
        max_new_tokens=args.max_new_tokens,
        layer_stride=args.layer_stride,
        head_stride=args.head_stride,
        device=args.device,
        do_sample=args.do_sample,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        seed=args.seed,
    )
    return run


# ---------- 後端 2：llama.cpp ----------

def resolve_gguf_path(model_arg: str, gguf_file_arg: str, models_dir: str) -> str:
    if model_arg.lower().endswith(".gguf") and os.path.isfile(model_arg):
        return os.path.abspath(model_arg)

    if gguf_file_arg:
        if os.path.isdir(model_arg):
            candidate = os.path.join(model_arg, gguf_file_arg)
            if os.path.isfile(candidate):
                return os.path.abspath(candidate)
        local_candidate = os.path.join(models_dir, "local", model_arg, gguf_file_arg)
        if os.path.isfile(local_candidate):
            return os.path.abspath(local_candidate)
        try:
            from huggingface_hub import hf_hub_download
        except ImportError:
            raise SystemExit("需要 huggingface_hub 才能從 HF repo 下載單一 gguf 檔："
                              "pip install huggingface_hub --break-system-packages")
        cache_dir = os.path.join(models_dir, "hf_cache")
        print(f"[models] 從 HF repo 下載單一 GGUF 檔: {model_arg} / {gguf_file_arg}")
        return hf_hub_download(repo_id=model_arg, filename=gguf_file_arg, cache_dir=cache_dir)

    # --model 直接給一個資料夾（不管是絕對路徑還是 models/local/ 下的名字），
    # 裡面剛好只有一個 .gguf 檔 -> 直接用，不用多打 --gguf_file
    for d in (model_arg, os.path.join(models_dir, "local", model_arg)):
        if os.path.isdir(d):
            ggufs = [f for f in os.listdir(d) if f.lower().endswith(".gguf")]
            if len(ggufs) == 1:
                print(f"[models] 在資料夾中找到唯一的 GGUF 檔: {os.path.join(d, ggufs[0])}")
                return os.path.abspath(os.path.join(d, ggufs[0]))
            if len(ggufs) > 1:
                raise SystemExit(f"{d} 底下有多個 .gguf 檔，請用 --gguf_file 指定其中一個: {ggufs}")

    raise SystemExit(
        "找不到 GGUF 檔案。請用下列其中一種方式指定：\n"
        "  1) --model 直接填 .gguf 檔案的完整路徑\n"
        "  2) --model 填一個資料夾路徑，裡面只放一個 .gguf 檔\n"
        "  3) 資料夾內有多個 .gguf 時，加 --gguf_file <檔名> 指定其中一個\n"
        "  4) --model <HF repo id> --gguf_file <檔名>，從 Hub 下載單一 gguf 檔"
    )


def run_llamacpp_backend(args):
    from backend_llamacpp import collect_trace_llamacpp

    gguf_path = resolve_gguf_path(args.model, args.gguf_file, args.models_dir)
    print(f"[1/2] llama.cpp 原生推論: {gguf_path}")
    print(f"[2/2] 執行 prefill + {args.max_new_tokens} 步 decode ...")
    if not args.no_layers:
        from backend_llamacpp import collect_trace_llamacpp_layers
        print("[layers] 用 ggml eval-callback 抓逐層 attention / MLP / residual（會強制關閉 flash attention，較慢）")
        try:
            return collect_trace_llamacpp_layers(
                gguf_path, args.prompt, max_new_tokens=args.max_new_tokens,
                topk=max(args.topk_llamacpp, 1), n_ctx=args.n_ctx, seed=args.seed or 0,
                chat=not args.raw, system_prompt=args.system, lens=not args.no_lens,
            )
        except RuntimeError as e:
            # 逐層採集的前置設定失敗（找不到 ggml 函式庫、ggml 版本對不上…），
            # 不要讓整個指令崩潰：明講原因，退回不含逐層的模式，至少 token 與候選詞照樣拿得到
            print("\n[警告] 逐層採集初始化失敗，改用不含逐層的模式繼續：")
            print(f"       {e}\n")
    run = collect_trace_llamacpp(
        gguf_path, args.prompt, max_new_tokens=args.max_new_tokens,
        topk=max(args.topk_llamacpp, 1), n_ctx=args.n_ctx, seed=args.seed or 0,
        chat=not args.raw, system_prompt=args.system,
    )
    return run


# ---------- CLI ----------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--backend", choices=["auto", "transformers", "llamacpp"], default="auto",
                     help="auto：依 --model/--gguf_file 是否指向 .gguf 自動判斷")
    ap.add_argument("--model", required=True,
                     help="後端1: HF repo id / models/local/ 下的資料夾名 / 本地路徑\n"
                          "後端2: .gguf 檔完整路徑，或 models/local/ 下的資料夾名（配合 --gguf_file）")
    ap.add_argument("--gguf_file", default=None, help="後端2專用：GGUF 檔名（--model 是資料夾或repo id時要給）")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new_tokens", type=int, default=8)

    # 後端1 專用
    ap.add_argument("--layer_stride", type=int, default=1, help="[transformers] 只取每隔幾層")
    ap.add_argument("--head_stride", type=int, default=1, help="[transformers] 只取每隔幾個 head")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--load_in_4bit", action="store_true", help="[transformers] 用 bitsandbytes 4-bit 量化載入")
    ap.add_argument("--raw", action="store_true",
                     help="不套用聊天格式：[transformers] 不套用 chat_template；[llamacpp] 不用 create_chat_completion，"
                          "退回純續寫的 create_completion。想看模型單純續寫文字時用這個")
    ap.add_argument("--system", default=None, help="system prompt（兩個後端都支援；llamacpp需要gguf內嵌chat_template才有效）")
    ap.add_argument("--do_sample", action="store_true", help="[transformers] 開啟取樣，預設greedy")
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--top_k", type=int, default=40)
    ap.add_argument("--seed", type=int, default=None)

    # 後端2 專用
    ap.add_argument("--n_ctx", type=int, default=4096, help="[llamacpp] context window")
    ap.add_argument("--layers", action="store_true",
                     help="[llamacpp] 抓逐層資料（現在預設就開，這個旗標保留只是為了相容舊指令）")
    ap.add_argument("--no_layers", action="store_true",
                     help="[llamacpp] 不抓逐層，只要 token 與候選詞（會用原生 llama.cpp，快很多，也開著 flash attention）")
    ap.add_argument("--no_lens", action="store_true",
                     help="[llamacpp] 不算 logit lens（會省下讀取輸出矩陣的時間與記憶體）")
    ap.add_argument("--topk_llamacpp", type=int, default=8, help="[llamacpp] 每步輸出候選詞數量")

    ap.add_argument("--models_dir", default=DEFAULT_MODELS_DIR)
    ap.add_argument("--runs_dir", default=os.path.join(BASE_DIR, "runs"))
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    os.makedirs(os.path.join(args.models_dir, "local"), exist_ok=True)
    os.makedirs(os.path.join(args.models_dir, "hf_cache"), exist_ok=True)
    os.makedirs(args.runs_dir, exist_ok=True)

    backend = args.backend
    if backend == "auto":
        backend = "llamacpp" if looks_like_gguf_target(args.model, args.gguf_file, args.models_dir) else "transformers"
        print(f"[backend] 自動判斷為: {backend}")

    run = run_llamacpp_backend(args) if backend == "llamacpp" else run_transformers_backend(args)

    out_path = args.out or os.path.join(args.runs_dir, f"run_{int(time.time()*1000)}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(run, f, ensure_ascii=False, indent=1)

    print(f"[完成] backend={run.get('backend')} steps={len(run['steps'])} layers={len(run['layers_meta'])}")
    print(f"generated_text_decoded: {run.get('generated_text_decoded', '')!r}")
    print(f"寫出: {out_path}")
    print("打開 llm_trace_viewer.html，按「載入 run.json」選這個檔案")


if __name__ == "__main__":
    main()
