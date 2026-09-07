#!/system/bin/sh
# Run on the prepared phone with the benchmark binaries and models already installed.
set -eu
bin=/data/local/tmp/sdar-ar-crossover-current
out=/mnt/nvme/sdar-ar-crossover-$(date +%Y%m%d-%H%M%S)-$$
export LD_LIBRARY_PATH=/vendor/lib64
mkdir "$out"
test -e /sys/bus/platform/drivers/pwm-fan/c42d000.qcom,spmi:qcom,pm8550@1:pwm-fan

prompt='Continue writing numbered facts about mobile processors until the requested token budget is exhausted. Begin with fact one and do not conclude early'
ar_instruction=$prompt
ar_suffix=''
tag=normal

run() {
    log=$1; shift
    while [ "$(cat /sys/class/thermal/thermal_zone*/temp | sort -nr | head -n 1)" -gt 42000 ]; do sleep 5; done
    echo "$log"
    "$@" > "$out/$log.log" 2>&1
}

sdar() { # block, steps, output tokens, repeat, flash attention
    run "sdar-b$1-s$2-n$3-r$4-fa$5" "$bin/llama-diffusion-cli" \
        -m /mnt/nvme/SDAR-4B-Chat-Q4_K_M.gguf -p "$prompt" \
        -c 512 -b 512 -t 8 -ngl 99 -ub 512 -n "$3" --ignore-eos --flash-attn "$5" \
        --diffusion-block-length "$1" --diffusion-steps "$2" --diffusion-algorithm 4 \
        --temp 0 --seed 1 --perf --log-colors off --verbosity 4
}

ar() { # model name, output tokens, repeat, flash attention
    run "ar-$1-n$2-r$3-fa$4-$tag" "$bin/llama-completion" \
        -m "/mnt/nvme/$1-Q4_K_M.gguf" -p "<|im_start|>user
${ar_instruction}<|im_end|>
<|im_start|>assistant
${ar_suffix}" \
        -no-cnv --flash-attn "$4" -c 512 -ub 512 -b 512 -n "$2" \
        -t 8 -ngl 99 --ignore-eos --temp 0 --seed 1 \
        --no-display-prompt --no-warmup --perf --log-colors off --verbosity 4 --simple-io
}

# Main grid: four AR repeats; three SDAR repeats at the crossover/reference cells.
for n in 96 224 480; do
    for r in 1 2 3 4; do
        ar Qwen3-4B "$n" "$r" auto
        ar SDAR-4B-Chat "$n" "$r" auto
    done
    for b in 4 8 16 32 64 128; do
        for s in 1 2 4; do
            repeats=1
            case "$b:$s:$n" in
                4:*:*|8:2:*|16:4:*|32:4:*|32:1:96|128:1:224|128:1:480) repeats=3 ;;
            esac
            for r in $(seq 1 "$repeats"); do sdar "$b" "$s" "$n" "$r" auto; done
        done
    done
done

# Short-output crossover and Flash Attention checks.
for r in 1 2 3; do
    for n in 16 32 40 64; do
        ar Qwen3-4B "$n" "$r" auto
        ar SDAR-4B-Chat "$n" "$r" auto
        sdar 4 1 "$n" "$r" auto
    done
    for n in 32 64; do sdar 32 4 "$n" "$r" auto; done
    sdar 128 1 480 "$r" off
    ar Qwen3-4B 224 "$r" off
done
ar SDAR-4B-Chat 224 1 off

# Separate Qwen direct-answer checks: still exactly 32 input tokens.
ar_instruction='Continue writing numbered facts about mobile processors until the budget is exhausted. Begin with one and do not conclude'
ar_suffix='<think>

</think>

'
tag=no-thinking
for n in 96 224 480; do ar Qwen3-4B "$n" 1 auto; done
echo "Results: $out"
