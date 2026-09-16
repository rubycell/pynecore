#!/usr/bin/env bash
# Step-5 runner (sep16 plan): WS BROKER-channel dual capture (#121/#107 WS half).
# Starts the extended WS watcher (short + broker order channels, per-channel counters),
# then generates ONE guaranteed account order event by running the L0 gate (its probe
# orders place+cancel inside the capture window). Read-only apart from L0's own probes.
set -u
cd "$(dirname "$0")/../../../.."
PY=.venv/bin/python
LOGDIR=plugins/dnse/testing/live_test/logs; mkdir -p "$LOGDIR"
TS=$(date +%H%M%S); WSLOG="$LOGDIR/ws_dual_$TS.log"

echo "=== [1/3] start dual-channel WS watcher (background, 180s) ==="
$PY plugins/dnse/testing/live_test/probe_ws_market_data.py --trading --seconds 180 \
    > "$WSLOG" 2>&1 &
WSPID=$!
sleep 8   # auth + subscribe before any event

echo "=== [2/3] generate order events: L0 gate (places+cancels probe orders) ==="
$PY plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py \
    2>&1 | tee "$LOGDIR/ws_dual_l0_$TS.log"
L0=${PIPESTATUS[0]}
echo "  L0 exit=$L0 (its place/cancel events are the capture payload)"

echo "=== [3/3] wait for the watcher window, then per-channel verdict ==="
wait "$WSPID" 2>/dev/null
strip(){ sed 's/\x1b\[[0-9;]*m//g'; }
echo "---------------- WS DUAL CAPTURE RESULT ----------------"
strip < "$WSLOG" | tail -30
echo
echo "VERDICT guide: frames on BOTH channels -> dual-subscribe is redundant-safe (keep)."
echo "  short-only -> broker channel is dead on prod: ws_order_source should PREFER short."
echo "  broker-only (unexpected) -> the 09-15 short capture needs re-examination."
echo "  neither -> the event may have missed the window; re-run (empty != conclusive)."
exit 0
