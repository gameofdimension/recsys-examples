set -xeuo pipefail

beam=$1
concurrency=$2

pip install uv -i http://pypi.vip.vip.com/simple

SGLANG_VENV=/opt/sglang-venv
export PATH=${SGLANG_VENV}/bin:${PATH}
if python -c "import sglang.bench_serving" &> /dev/null; then
    echo "sglang.bench_serving is available"
else
    rm -fr $SGLANG_VENV/
    export RUSTUP_DIST_SERVER=https://mirrors.tuna.tsinghua.edu.cn/rustup
    rm -rf /root/.rustup /root/.cargo \
        && (apt-get remove -y rustc cargo 2>/dev/null || true) \
        && curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
            | sh -s -- -y --default-toolchain stable \
        && . "$HOME/.cargo/env" \
        && rustc --version && cargo --version
    export PATH=/root/.cargo/bin:${PATH}
    
    old_dir=`pwd`
    cd /root/cswuyg-sglang/
    uv venv ${SGLANG_VENV}
    uv pip install --python ${SGLANG_VENV}/bin/python -e "python" kernels==0.14.1 kernels-data==0.14.1 -i http://pypi.vip.vip.com/simple
    cd $old_dir
fi

python -m sglang.bench_serving \
  --backend sglang \
  --host 127.0.0.1 \
  --port 8000 \
  --model /warehouse/Qwen3-1.7B \
  --tokenizer /warehouse/Qwen3-1.7B \
  --dataset-name random \
  --random-input-len 3000 \
  --random-output-len 3 \
  --random-range-ratio 1 \
  --num-prompts 200 \
  --request-rate inf \
  --max-concurrency "$concurrency" \
  --warmup-requests 10 \
  --disable-stream \
  --disable-ignore-eos \
  --tokenize-prompt \
  --extra-request-body '{"sampling_params":{"temperature":0,"max_new_tokens":3,"ignore_eos":true,"n":'"$beam"'}}' \
  --output-details \
  --output-file bw${beam}_mc${concurrency}.jsonl
