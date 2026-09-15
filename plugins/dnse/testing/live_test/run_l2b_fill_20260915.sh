#!/usr/bin/env bash
# PHASE 2 (FILL) runner for l2b_entry_update at padPct=0 — take a REAL position on a breakout,
# then observe whether the SL/TP OCO bracket is placed RIGHT AFTER the fill (arm-on-fill), and
# whether it STAYS armed (today's #124 bounded-re-arm fix on a live conditional exit).
# 360s timeout so it ends at the 11:30 lunch, never straddles it. Operator watches; app open.
set -u
cd "$(dirname "$0")/../../../.."
PY=.venv/bin/python
PYNE=.venv/bin/pyne
LOGDIR=plugins/dnse/testing/live_test/logs
mkdir -p "$LOGDIR"
TS=$(date +%H%M%S)
LOG="$LOGDIR/l2b_fill_$TS.log"

echo "=== [1/4] pre-flight: token GOOD + account FLAT ==="
$PY plugins/dnse/tools/token_status.py 2>&1 | tail -2
$PY plugins/dnse/tools/venue.py flat > /dev/null 2>&1; FLAT=$?
if [ "$FLAT" -ne 0 ]; then
    echo "!!! account NOT flat/clean (exit $FLAT) — ABORTING."; $PY plugins/dnse/tools/venue.py status 2>&1 | sed 's/\x1b\[[0-9;]*m//g'; exit "$FLAT"
fi
echo "  FLAT + clean."

echo "=== [2/4] L0 gate (must exit 0) ==="
$PY plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py 2>&1 | tee "$LOGDIR/l2b_fill_l0_$TS.log"
L0=${PIPESTATUS[0]}; [ "$L0" -ne 0 ] && { echo "!!! L0 FAILED ($L0) — ABORTING."; exit "$L0"; }

echo "=== [3/4] l2b_entry_update --broker @1m, padPct=0 FILL, 360s (ends ~lunch) ==="
echo "    watch: STOP fill -> 'dispatching EXIT' + 'event CREATED ... leg=tp' (bracket armed)"
echo "           then does it STAY armed, or 'protective_exit_rearm'/'quarantined' (#124)?"
timeout 360 $PYNE run plugins/dnse/testing/live_test/l2b_entry_update.py \
    dnse_broker:VN30F1M@1 --broker 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}
echo "  pyne exit=$RC"

echo "=== [4/4] teardown: account MUST end flat + clean ==="
$PY plugins/dnse/tools/venue.py status 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
$PY plugins/dnse/tools/venue.py flat; FLATEND=$?
echo
echo "================ l2b FILL SUMMARY ================"
strip(){ sed 's/\x1b\[[0-9;]*m//g'; }
echo "entry fill:"; strip < "$LOG" | grep -aoE "event FILLED[^[]*leg=entry" | tail -2
echo "bracket arm:"; strip < "$LOG" | grep -aoE "dispatching EXIT[^[]*|event CREATED[^[]*leg=tp" | tail -4
echo "protective re-arm / quarantine (#124):"; strip < "$LOG" | grep -aicE "protective_exit_rearm|quarantined" | xargs echo "  count:"
echo "flat@end: exit $FLATEND"
[ "$FLATEND" -ne 0 ] && echo "!!! NOT FLAT — FLATTEN IN THE DNSE APP or venue.py cancel <id>."
exit 0
