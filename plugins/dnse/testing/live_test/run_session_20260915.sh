#!/usr/bin/env bash
# One-shot runner for the 2026-09-15 live session (runbook: docs/plan/live_session_20260915_runbook.md)
# Order: WS-watcher(bg) -> L0 gate -> (1) #124 OCO isolation -> (3) #116 -> (4) #117 -> teardown.
# The WS watcher runs across (1), so (1)'s own OCO place+cancel IS the account event
# step (2) needs -- one run answers both.
set -u
cd "$(dirname "$0")/../../../.."           # repo root
PY=.venv/bin/python
LOGDIR=plugins/dnse/testing/live_test/logs
mkdir -p "$LOGDIR"
TS=$(date +%H%M%S)

echo "=== [0/6] starting prod trading-WS watcher (background, 300s) ==="
$PY plugins/dnse/testing/live_test/probe_ws_market_data.py --trading --seconds 300 \
    > "$LOGDIR/session0915_ws_trading_$TS.log" 2>&1 &
WSPID=$!
sleep 8    # let it auth+subscribe before any order event happens

echo "=== [1/6] L0 venue-semantics gate ==="
$PY plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py \
    2>&1 | tee "$LOGDIR/session0915_l0_$TS.log"
L0=${PIPESTATUS[0]}
if [ "$L0" -ne 0 ]; then
    echo "!!! L0 FAILED (exit $L0) — ABORTING the session. Nothing else runs."
    kill "$WSPID" 2>/dev/null; exit "$L0"
fi

echo "=== [2/6] (1) #124 OCO lifecycle isolation (--yes, watch 120s) ==="
$PY plugins/dnse/testing/live_test/probe_124_oco_lifecycle_isolation.py --yes \
    2>&1 | tee "$LOGDIR/session0915_probe124_$TS.log"
P124=${PIPESTATUS[0]}

echo "=== [3/6] (3) #116 cancel-a-FILLED-id ==="
$PY plugins/dnse/testing/live_test/probe_116_117_prod_premises.py A \
    2>&1 | tee "$LOGDIR/session0915_p116_$TS.log"

echo "=== [4/6] (4) #117 STOCK amend semantics ==="
$PY plugins/dnse/testing/live_test/probe_116_117_prod_premises.py B \
    2>&1 | tee "$LOGDIR/session0915_p117_$TS.log"

echo "=== [5/6] waiting for the WS watcher to finish its window ==="
wait "$WSPID" 2>/dev/null
echo "--- WS watcher tail ---"; tail -20 "$LOGDIR/session0915_ws_trading_$TS.log"

echo "=== [6/6] teardown: account must be flat+clean ==="
$PY plugins/dnse/tools/venue.py status 2>&1 | tee "$LOGDIR/session0915_teardown_$TS.log"
$PY plugins/dnse/tools/venue.py flat;  FLAT=$?
echo
echo "================ SESSION SUMMARY ================"
echo "L0 gate      : exit $L0"
echo "probe #124   : exit $P124   (VERDICT line is in session0915_probe124_$TS.log)"
echo "venue flat   : exit $FLAT   (0 = truly flat+clean)"
echo "logs         : $LOGDIR/session0915_*_$TS.log"
[ "$FLAT" -ne 0 ] && echo "!!! ACCOUNT NOT PROVABLY CLEAN — check venue.py status output above !!!"
exit 0
