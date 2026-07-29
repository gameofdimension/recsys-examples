#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Bash/curl port of tools/demo_update_weights_from_disk.py.
#
# Demo client: hot-swap model weights from disk against a running GR HTTP
# server, driving the slime-style disk weight-sync flow end to end with real
# workload probes around each step:
#
#    1. GET  /get_weight_version + /get_weights_by_name   (record "before" state)
#    2. POST /generate                                    (output A: original weights)
#    3. POST /pause_generation                            (mode=abort)
#    4. POST /generate                                    (probe: NO service while
#       paused -- the server rejects it with 503 code=paused)
#    5. GET  /flush_cache                                 (retry until 200, slime-style)
#    6. POST /update_weights_from_disk                    (swap to the new checkpoint)
#    7. POST /continue_generation
#    8. GET  /get_weight_version + /get_weights_by_name   (state changed)
#    9. POST /generate                                    (output B: must DIFFER from A)
#   10. pause -> flush -> update_weights_from_disk -> continue
#                                                        (restore the ORIGINAL checkpoint)
#   11. GET  /get_weights_by_name                         (sample must match the
#       original -- restore verified at the weight level)
#   12. POST /generate                                    (output C: must EQUAL A)
#
# Deterministic sampling (temperature=0, ignore_eos), so identical weights +
# inputs yield identical output_ids across rounds -- output C must equal A.
#
# Requires: curl, jq.

set -uo pipefail

# --------------------------------------------------------------------------- #
# Defaults (mirror tools/demo_update_weights_from_disk.py)
# --------------------------------------------------------------------------- #

BASE_URL="http://127.0.0.1:8000"
MODEL_PATH=""
WEIGHT_VERSION=""
RESTORE_MODEL_PATH=""
RESTORE_WEIGHT_VERSION=""
PARAM_NAME="embed_tokens.weight"
TRUNCATE_SIZE=2
API_KEY=""
FLUSH_RETRIES=60

DEFAULT_TIMEOUT=300        # seconds; matches the python client default
PROBE_TIMEOUT=3            # paused-probe client timeout (a fast 503 is expected)

# Reference workload (a real GR request shape), used verbatim for all probes:
#   input_ids: [9707, 887, 525, 263, 590, 34561, 13, 198, 9707, 887, 525, 263]
#   sampling_params: {temperature: 0, max_new_tokens: 3, ignore_eos: true, n: 256}
PROBE_INPUT_IDS_JSON='[9707,887,525,263,590,34561,13,198,9707,887,525,263]'
PROBE_MAX_NEW_TOKENS=3
PROBE_BEAM_WIDTH=256

# HTTP result globals, refreshed by curl_request.
HTTP_STATUS=0   # 0 == connection failure / curl error
HTTP_BODY=""
CURL_RC=0

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #

die() {
    echo "ERROR: $*" >&2
    exit 1
}

usage() {
    cat <<'EOF'
Demo client: hot-swap model weights from disk against a running GR HTTP server
(slime-style disk weight-sync flow, workload-verified).
Bash/curl port of tools/demo_update_weights_from_disk.py.

Usage:
  tools/demo_update_weights_from_disk.sh \
      --base-url http://127.0.0.1:8000 \
      --model-path /models/Qwen3-GR-plus1 --weight-version plus1 \
      --restore-model-path /models/Qwen3-GR --restore-weight-version base

Options:
  --base-url URL                 server base URL (default: http://127.0.0.1:8000)
  --model-path PATH              NEW checkpoint dir, readable by the *server* (required)
  --weight-version LABEL         version label for the new checkpoint (default: dir name)
  --restore-model-path PATH      ORIGINAL checkpoint dir, restored in round 2 (required)
  --restore-weight-version LABEL version label for restore (default: startup version,
                                 else the restore dir name)
  --param-name NAME              module parameter to sample (default: embed_tokens.weight)
  --truncate-size N              rows to fetch per sample (default: 2)
  --api-key KEY                  X-GR-API-Key value (default: none)
  --flush-retries N              flush_cache retries while busy, slime uses 60 (default: 60)
  -h, --help                     show this help and exit

Requires: curl, jq.
EOF
}

is_uint() { [[ "$1" =~ ^[0-9]+$ ]]; }

# 8-hex-char request suffix (mirrors python's uuid4().hex[:8]).
gen_id() {
    if command -v uuidgen >/dev/null 2>&1; then
        uuidgen | tr -d '-' | cut -c1-8
    elif [[ -r /proc/sys/kernel/random/uuid ]]; then
        cut -c1-8 /proc/sys/kernel/random/uuid | tr -d '-'
    else
        od -An -tx1 -N4 /dev/urandom | tr -d ' \n'
    fi
}

TMP_BODY="$(mktemp)"
trap 'rm -f "$TMP_BODY"' EXIT

# curl_request METHOD PATH [JSON_PAYLOAD] [TIMEOUT_S]
#   Sets HTTP_STATUS (0 on connection failure / curl error), HTTP_BODY, CURL_RC.
#   Always returns 0 so it composes with the flow's own status checks.
curl_request() {
    local method="$1"
    local path="$2"
    local payload="${3:-}"
    local timeout="${4:-$DEFAULT_TIMEOUT}"
    local url="${BASE_URL%/}/${path#/}"

    local -a args=(
        -sS -o "$TMP_BODY" -w '%{http_code}'
        --max-time "$timeout"
        -X "$method"
        -H 'Content-Type: application/json'
    )
    [[ -n "${API_KEY:-}" ]] && args+=(-H "X-GR-API-Key: ${API_KEY}")
    [[ -n "$payload" ]] && args+=(--data "$payload")

    local code curl_rc=0
    code="$(curl "${args[@]}" "$url" 2>/dev/null)" || curl_rc=$?

    HTTP_BODY="$(<"$TMP_BODY")"
    CURL_RC="$curl_rc"
    if (( curl_rc != 0 )) || [[ "$code" == "000" || -z "$code" ]]; then
        HTTP_STATUS=0
    else
        HTTP_STATUS="$code"
    fi
    return 0
}

body_preview() {
    printf '%s' "$HTTP_BODY" | head -c 400 || true
}

# require_ok STEP -- the previous curl_request must have returned 2xx.
require_ok() {
    local step="$1"
    (( HTTP_STATUS / 100 == 2 )) || die "[$step] failed: HTTP ${HTTP_STATUS} $(body_preview)"
}

json_get() {
    jq -r "$1" 2>/dev/null <<<"$HTTP_BODY"
}

# get_weights_by_name -> canonical "[v1, v2, ...]" (first row, up to 8 values).
# Equal weights produce an identical string, so before/restore compare cleanly.
weight_sample() {
    local name="$1" trunc="$2"
    curl_request GET "/get_weights_by_name?name=${name}&truncate_size=${trunc}"
    require_ok get_weights_by_name
    local nrows
    nrows="$(jq -r '.parameter | length' 2>/dev/null <<<"$HTTP_BODY")"
    [[ "$nrows" -gt 0 ]] || die "[get_weights_by_name] empty parameter for '$name'"
    jq -r '
        (.parameter[0]) as $row
        | (if ($row | type) == "array" then $row[0:8] else [$row] end)
        | "[" + (map(. | tostring) | join(", ")) + "]"
    ' 2>/dev/null <<<"$HTTP_BODY"
}

# generate -> canonical "[id, id, ...]" (top-beam output_ids).
generate() {
    local input_ids="$1" max_new="$2" beam="$3"
    local req_id payload ids
    req_id="demo-generate-$(gen_id)"
    payload="$(jq -c -n \
        --arg rid "$req_id" \
        --argjson ids "$input_ids" \
        --argjson mnt "$max_new" \
        --argjson n "$beam" \
        '{request_id:$rid, input_ids:$ids,
          sampling_params:{temperature:0.0, ignore_eos:true, max_new_tokens:$mnt, n:$n}}')"
    curl_request POST /generate "$payload"
    require_ok generate
    ids="$(jq -r '[.output_ids[]?] | map(tostring) | join(", ")' 2>/dev/null <<<"$HTTP_BODY")"
    [[ -n "$ids" ]] || die "[generate] unexpected response shape: $(body_preview)"
    echo "[$ids]"
}

# While paused the server rejects inference with 503 (code=paused, retryable);
# that 503 IS the "no service while paused" proof. A 2xx is a failure (engine
# kept serving); a hang/timeout or any other status is also a failure.
probe_no_service_while_paused() {
    local req_id payload code
    req_id="demo-paused-probe-$(gen_id)"
    payload="$(jq -c -n \
        --arg rid "$req_id" \
        --argjson ids "$PROBE_INPUT_IDS_JSON" \
        '{request_id:$rid, input_ids:$ids,
          sampling_params:{temperature:0.0, ignore_eos:true, max_new_tokens:1, n:1}}')"
    curl_request POST /generate "$payload" "$PROBE_TIMEOUT"

    if (( HTTP_STATUS == 503 )); then
        code="$(json_get '.error.code // empty')"
        if [[ "$code" == "paused" ]]; then
            echo "   paused probe: /generate rejected with 503 (code=paused) -> no service while paused"
            return 0
        fi
        die "[paused probe] got 503 but unexpected error code '$code': $(body_preview)"
    fi
    if (( HTTP_STATUS / 100 == 2 )); then
        die "[paused probe] /generate SUCCEEDED while paused -- the engine kept serving after pause_generation"
    fi
    if (( HTTP_STATUS == 0 && CURL_RC == 28 )); then
        die "[paused probe] /generate hung ${PROBE_TIMEOUT}s instead of being rejected with 503 -- the server did not reject inference while paused"
    fi
    die "[paused probe] unexpected HTTP ${HTTP_STATUS} (expected 503 code=paused, curl rc=${CURL_RC}): $(body_preview)"
}

flush_cache_with_retry() {
    local retries="$1" attempt
    for (( attempt = 1; attempt <= retries; attempt++ )); do
        curl_request GET /flush_cache
        if (( HTTP_STATUS == 200 )); then
            echo "   flush_cache: $HTTP_BODY (attempt $attempt)"
            return 0
        fi
        sleep 1
    done
    die "[flush_cache] still busy after $retries attempts: $(body_preview)"
}

# pause_generation -> flush_cache -> update_weights_from_disk -> continue
update_round() {
    local model_path="$1" weight_version="$2" flush_retries="$3" label="$4"
    local payload
    curl_request POST /pause_generation '{}'
    require_ok "pause_generation($label)"
    echo "   pause_generation: $HTTP_BODY"
    flush_cache_with_retry "$flush_retries"
    payload="$(jq -c -n --arg mp "$model_path" --arg wv "$weight_version" \
        '{model_path:$mp, weight_version:$wv}')"
    curl_request POST /update_weights_from_disk "$payload"
    require_ok "update_weights_from_disk($label)"
    echo "   update_weights_from_disk: $HTTP_BODY"
    curl_request POST /continue_generation '{}'
    require_ok "continue_generation($label)"
    echo "   continue_generation: $HTTP_BODY"
}

# --------------------------------------------------------------------------- #
# Argument parsing
# --------------------------------------------------------------------------- #

while [[ $# -gt 0 ]]; do
    case "$1" in
        --base-url)               BASE_URL="${2:-}"; shift 2 ;;
        --model-path)             MODEL_PATH="${2:-}"; shift 2 ;;
        --weight-version)         WEIGHT_VERSION="${2:-}"; shift 2 ;;
        --restore-model-path)     RESTORE_MODEL_PATH="${2:-}"; shift 2 ;;
        --restore-weight-version) RESTORE_WEIGHT_VERSION="${2:-}"; shift 2 ;;
        --param-name)             PARAM_NAME="${2:-}"; shift 2 ;;
        --truncate-size)          TRUNCATE_SIZE="${2:-}"; shift 2 ;;
        --api-key)                API_KEY="${2:-}"; shift 2 ;;
        --flush-retries)          FLUSH_RETRIES="${2:-}"; shift 2 ;;
        -h|--help)                usage; exit 0 ;;
        *)                        die "unknown argument: $1 (try --help)" ;;
    esac
done

[[ -n "$MODEL_PATH" ]] || { usage; die "--model-path is required"; }
[[ -n "$RESTORE_MODEL_PATH" ]] || { usage; die "--restore-model-path is required"; }
is_uint "$TRUNCATE_SIZE" || die "--truncate-size must be a non-negative integer"
is_uint "$FLUSH_RETRIES" || die "--flush-retries must be a non-negative integer"

for dep in curl jq; do
    command -v "$dep" >/dev/null 2>&1 || die "required tool not found: $dep"
done

# Default weight-version = the checkpoint dir name (mirrors the python client).
WEIGHT_VERSION="${WEIGHT_VERSION:-$(basename "${MODEL_PATH%/}")}"

# --------------------------------------------------------------------------- #
# Flow
# --------------------------------------------------------------------------- #

echo "== target: $BASE_URL"

# --- 1. record the "before" state -------------------------------------------
curl_request GET /get_weight_version
require_ok get_weight_version
before_body="$HTTP_BODY"
before_weight_version="$(json_get '.weight_version // empty')"
echo "1. weight_version (before): $before_body"
before_sample="$(weight_sample "$PARAM_NAME" "$TRUNCATE_SIZE")"
echo "   $PARAM_NAME (before): $before_sample"

# --- 2. workload output with the ORIGINAL weights ---------------------------
output_a="$(generate "$PROBE_INPUT_IDS_JSON" "$PROBE_MAX_NEW_TOKENS" "$PROBE_BEAM_WIDTH")"
echo "2. generate (original weights): output_ids=$output_a"

# --- 3-7. pause (prove no service) -> flush -> update -> continue -----------
echo "3. update round 1: swap to the new checkpoint"
curl_request POST /pause_generation '{}'
require_ok pause_generation
echo "   pause_generation: $HTTP_BODY"
probe_no_service_while_paused
flush_cache_with_retry "$FLUSH_RETRIES"
update1_payload="$(jq -c -n --arg mp "$MODEL_PATH" --arg wv "$WEIGHT_VERSION" \
    '{model_path:$mp, weight_version:$wv}')"
curl_request POST /update_weights_from_disk "$update1_payload"
require_ok update_weights_from_disk
echo "   update_weights_from_disk: $HTTP_BODY"
curl_request POST /continue_generation '{}'
require_ok continue_generation
echo "   continue_generation: $HTTP_BODY"

# --- 8. confirm state changed -----------------------------------------------
curl_request GET /get_weight_version
require_ok get_weight_version
echo "8. weight_version (after): $HTTP_BODY"
after_weight_version="$(json_get '.weight_version // empty')"
if [[ "$after_weight_version" != "$WEIGHT_VERSION" ]]; then
    die "weight_version mismatch: expected '$WEIGHT_VERSION', got '$after_weight_version'"
fi
after_sample="$(weight_sample "$PARAM_NAME" "$TRUNCATE_SIZE")"
echo "   $PARAM_NAME (after): $after_sample"

# --- 9. workload output with the NEW weights: must differ -------------------
output_b="$(generate "$PROBE_INPUT_IDS_JSON" "$PROBE_MAX_NEW_TOKENS" "$PROBE_BEAM_WIDTH")"
echo "9. generate (new weights): output_ids=$output_b"
if [[ "$output_b" == "$output_a" ]]; then
    die "output UNCHANGED after the weight swap -- the new checkpoint does not seem to be in effect"
fi
echo "   -> output changed after the weight swap (as expected)"

# --- 10. restore the ORIGINAL weights ---------------------------------------
restore_version="${RESTORE_WEIGHT_VERSION:-${before_weight_version:-$(basename "${RESTORE_MODEL_PATH%/}")}}"
echo "10. update round 2: restore the original checkpoint"
update_round "$RESTORE_MODEL_PATH" "$restore_version" "$FLUSH_RETRIES" "restore"

# --- 11. weight sample after restore: must match the original sample --------
restored_sample="$(weight_sample "$PARAM_NAME" "$TRUNCATE_SIZE")"
echo "11. $PARAM_NAME (restored): $restored_sample"
if [[ "$restored_sample" != "$before_sample" ]]; then
    die "weight sample DIFFERS from the original after restore -- expected $before_sample, got $restored_sample"
fi
echo "   -> weight sample matches the original (restore verified)"

# --- 12. workload output after restore: must match the original -------------
output_c="$(generate "$PROBE_INPUT_IDS_JSON" "$PROBE_MAX_NEW_TOKENS" "$PROBE_BEAM_WIDTH")"
echo "12. generate (restored weights): output_ids=$output_c"
if [[ "$output_c" != "$output_a" ]]; then
    die "output DIFFERS from the original after restoring weights -- expected $output_a, got $output_c"
fi
echo "   -> output matches the original weights exactly (restore verified)"

echo "OK: disk weight update + restore completed and verified."
