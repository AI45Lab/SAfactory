#!/usr/bin/env bash
# 收集最新 patcheval run 的效率指标，用于 POOL_SIZE 扫描对比。
# 用法: bash rl/collect_pool_metrics.sh <POOL_SIZE标签>
#   例如: bash rl/collect_pool_metrics.sh 8
set -euo pipefail

LABEL="${1:-unknown}"
cd /mnt/shared-storage-user/leishanzhe/repo/SAfactory

f="$(ls -t logs/patcheval_qwen3_8_27b/*/slime.log 2>/dev/null | head -1)"
if [[ -z "$f" ]]; then
  echo "[collect] 找不到 slime.log" >&2
  exit 1
fi

echo "=========================================="
echo " POOL_SIZE=${LABEL}  run: $f"
echo "=========================================="

# 时间范围
t0="$(grep -oE '2026-[0-9-]+ [0-9:]+' "$f" | head -1)"
t1="$(grep -oE '2026-[0-9-]+ [0-9:]+' "$f" | tail -1)"
echo "时间: $t0 -> $t1"

# 启动配置
echo "--- 启动配置 ---"
grep -E "mem_fraction_static=[0-9]" "$f" | grep -v repeated | head -1 | grep -oE "mem_fraction_static=[0-9.]+" || true
grep -E "KV Cache is alloc" "$f" | grep -v repeated | head -1 | grep -oE "#tokens: [0-9]+, K size: [0-9.]+ GB, V size: [0-9.]+ GB" || true
grep -E "max_total_num_tokens=" "$f" | grep -v repeated | head -1 | grep -oE "max_total_num_tokens=[0-9]+, chunked_prefill_size=[0-9]+, max_prefill_tokens=[0-9]+, max_running_requests=[0-9]+, context_len=[0-9]+, available_gpu_mem=[0-9.]+ GB" || true

# cached-token 分布
echo "--- cached-token (KV 复用) ---"
grep -oE "#cached-token: [0-9]+" "$f" | awk '{print $2}' | awk '
{a[NR]=$1; n=NR} END{
  if(n==0){print "no prefill data"; exit}
  c0=0; sum=0
  for(i=1;i<=n;i++){v=a[i]; sum+=v; if(v==0)c0++}
  print "samples="n
  print "cached=0(无复用): "c0" ("int(c0*100/n)"%)"
  print "有复用: "(n-c0)" ("int((n-c0)*100/n)"%)"
  print "avg="int(sum/n)" max="a[n]
}'

# full token usage
echo "--- KV 占用率 ---"
grep -oE "full token usage: [0-9.]+" "$f" | awk '{print $4}' | sort -n | awk '
{a[NR]=$1; n=NR} END{if(n>0)print "p50="a[int(n/2)]" p90="a[int(n*0.9)]" max="a[n]}'

# running-req
echo "--- 并发请求数 ---"
grep -oE "#running-req: [0-9]+" "$f" | awk '{print $2}' | sort -n | awk '
{a[NR]=$1; n=NR} END{if(n>0){c=0; for(i=1;i<=n;i++)if(a[i]>=1)c++; print "p50="a[int(n/2)]" p90="a[int(n*0.9)]" max="a[n]" 有并发="c"("int(c*100/n)"%)"}}'

# queue-req
echo "--- 排队 ---"
grep -oE "#queue-req: [0-9]+" "$f" | awk '{print $2}' | sort -n | awk '
{a[NR]=$1; n=NR} END{if(n>0){c=0; for(i=1;i<=n;i++)if(a[i]>=1)c++; print "max="a[n]" 有排队="c"("int(c*100/n)"%)"}}'

# throughput
echo "--- 吞吐 ---"
grep -oE "gen throughput \(token/s\): [0-9.]+" "$f" | awk '{print $4}' | sort -n | awk '
{a[NR]=$1; n=NR} END{if(n>0)print "decode p50="a[int(n/2)]" max="a[n]}'
grep -oE "input throughput \(token/s\): [0-9.]+" "$f" | awk '{print $4}' | sort -n | awk '
{a[NR]=$1; n=NR} END{if(n>0)print "prefill p50="a[int(n/2)]" max="a[n]}'

# batch 计数
echo "--- 批次计数 ---"
echo "Prefill batches: $(grep -c 'Prefill batch' "$f")  Decode batches: $(grep -c 'Decode batch' "$f")"
echo "=========================================="
