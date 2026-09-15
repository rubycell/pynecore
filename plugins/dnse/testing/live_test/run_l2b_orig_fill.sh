#!/usr/bin/env bash
# FILL test for the ORIGINAL l2b_fill_protect_flatten (pre-placed entry-bar bracket).
# Goal: confirm the SL/TP arms FAST after the fill (#121 wake on a pre-armed exit), not a bar late.
# Real position; operator watches; app open. 540s; continuous window (to 14:30).
set -u
cd "$(dirname "$0")/../../../.."
PY=.venv/bin/python; PYNE=.venv/bin/pyne
LOGDIR=plugins/dnse/testing/live_test/logs; mkdir -p "$LOGDIR"
TS=$(date +%H%M%S); LOG="$LOGDIR/l2b_orig_$TS.log"

echo "=== [1/4] pre-flight: token GOOD + FLAT ==="
$PY plugins/dnse/tools/token_status.py 2>&1 | tail -2
$PY plugins/dnse/tools/venue.py flat >/dev/null 2>&1; FLAT=$?
[ "$FLAT" -ne 0 ] && { echo "!!! NOT flat/clean ($FLAT) — ABORTING."; $PY plugins/dnse/tools/venue.py status 2>&1|sed 's/\x1b\[[0-9;]*m//g'; exit "$FLAT"; }
echo "  FLAT + clean."

echo "=== [2/4] L0 gate (exit 0) ==="
$PY plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py 2>&1 | tee "$LOGDIR/l2b_orig_l0_$TS.log"
L0=${PIPESTATUS[0]}; [ "$L0" -ne 0 ] && { echo "!!! L0 FAILED ($L0) — ABORTING."; exit "$L0"; }

echo "=== [3/4] l2b_fill_protect_flatten --broker @1m, 540s ==="
echo "    watch: 'event FILLED ... leg=entry' then how SOON 'dispatching EXIT'/'event CREATED ... leg=tp' follows"
echo "    FAST (same bar / seconds) = #121 arm-on-fill WORKS on the pre-placed exit"
timeout 540 $PYNE run plugins/dnse/testing/live_test/l2b_fill_protect_flatten.py \
    dnse_broker:VN30F1M@1 --broker 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}; echo "  pyne exit=$RC"

echo "=== [4/4] teardown ==="
$PY plugins/dnse/tools/venue.py status 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
$PY plugins/dnse/tools/venue.py flat; FLATEND=$?
strip(){ sed 's/\x1b\[[0-9;]*m//g'; }
echo; echo "================ ARM-LATENCY SUMMARY ================"
echo "entry fill:"; strip < "$LOG" | grep -aoE "\[2026[^]]*\] bar: *[0-9]+ .*event FILLED[^[]*leg=entry" | tail -1
echo "bracket arm:"; strip < "$LOG" | grep -aoE "\[2026[^]]*\] bar: *[0-9]+ .*(dispatching EXIT|event CREATED[^[]*leg=tp)" | tail -3
echo "#124 rearm/quarantine:"; strip < "$LOG" | grep -aicE "protective_exit_rearm|re-arming|quarantined" | xargs echo "  count:"
echo "flat@end: exit $FLATEND"
[ "$FLATEND" -ne 0 ] && echo "!!! NOT FLAT — flatten in the DNSE app or venue.py cancel <id>."
exit 0
