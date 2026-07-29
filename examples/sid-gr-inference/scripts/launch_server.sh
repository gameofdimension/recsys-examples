set -xeuo pipefail

beam=$1
# 2nd arg toggles item-constrained decode (off=0, on=1/on/true/yes).
# Precedence: positional arg > GR_ITEM_CONSTRAINED env > default off.
item_constrained=${2:-${GR_ITEM_CONSTRAINED:-0}}
case "${item_constrained,,}" in
  1|on|true|yes|y) item_constrained=1 ;;
  *) item_constrained=0 ;;
esac

export GR_INFERENCE_DISABLE_PREFILL_CUDA_GRAPH=1 # 关闭 prefill CUDA graph
export GR_INFERENCE_DISABLE_DECODE_CUDA_GRAPH=1 # 关闭 decode CUDA graph

export GR_MODEL_DIR="/warehouse/Qwen3-1.7B"
export GR_CONTEXT_LEN=4000
export GR_DECODE_STEPS=3
export GR_BEAM_WIDTH="$beam"
export GR_MAX_BATCH_SIZE=4
export GR_BEAM_KV_POOL_CAPACITY=4
export GR_CONTEXT_KV_POOL_CAPACITY=4
export GR_DECODE_BACKEND=real
export GR_DEVICE=cuda
export GR_ENABLE_PREFILL_CACHE=0
export GR_HTTP_HOST=0.0.0.0
export GR_HTTP_PORT=${GR_HTTP_PORT:-8000}
export GR_WARMUP_ONLINE_SHAPES=1
export GR_WARMUP_ONLINE_POOL_WINDOWS=1
export GR_WARMUP_ONLINE_MAX_CASES=64
export GR_FREEZE_CUDA_GRAPHS_AFTER_WARMUP=1
export GR_DECODE_CUDA_GRAPH_BATCH_BUCKETS=1,2,4,8
export GR_ENABLE_PREFILL_CACHE=0

# ---- Item-constrained decode (optional) ----
# ON:  server loads the catalog JSONL as a token trie and masks every
#      /generate decode step so beams can only follow legal catalog token
#      paths. Also exposes /catalog/status|reload|rollback.
# OFF: catalog env is unset so decoding is unconstrained (authoritative).
# Defaults target Qwen3-1.7B (vocab 151936, eos 151645). Override the catalog
# path / vocab / eos via env, e.g. GR_CATALOG_JSONL=/abs/path launch_server.sh 256 1
if [[ "$item_constrained" == "1" ]]; then
  export GR_CATALOG_JSONL="${GR_CATALOG_JSONL:-$(realpath ./benchmark_artifacts/item_constrained_benchmark/20260715_031600/catalog_5000000_seed42.jsonl)}"
  export GR_CATALOG_VOCAB_SIZE="${GR_CATALOG_VOCAB_SIZE:-151936}"
  export GR_CATALOG_EOS_TOKEN_ID="${GR_CATALOG_EOS_TOKEN_ID:-151645}"
  export GR_ALLOW_CATALOG_RELOAD="${GR_ALLOW_CATALOG_RELOAD:-0}"
  echo "[launch_server] item-constrained ON  catalog=${GR_CATALOG_JSONL} vocab=${GR_CATALOG_VOCAB_SIZE} eos=${GR_CATALOG_EOS_TOKEN_ID}"
else
  unset GR_CATALOG_JSONL GR_CATALOG_VOCAB_SIZE GR_CATALOG_EOS_TOKEN_ID GR_ALLOW_CATALOG_RELOAD
  echo "[launch_server] item-constrained OFF"
fi

bash scripts/serve_qwen3_gr_http.sh
