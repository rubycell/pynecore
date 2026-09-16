#!/usr/bin/env bash
# Step-3 runner (sep16 plan): l3_fill_fixed_sl_tp LIVE — does the SL/TP actually FIRE?
# maxHoldBars=45 via the .toml (a 0.20% leg gets room to resolve at 1m). Real 1-lot fill;
# operator watches; app open. Timeout 3300s (~55 min) so the 45-bar hold + resolution fits.
# NEVER run into ATC: fire this no later than ~13:30 so 45 bars + teardown end before 14:30.
set -u
cd "$(dirname "$0")/../../../.."
PY=.venv/bin/python; PYNE=.venv/bin/pyne
LOGDIR=plugins/dnse/testing/live_test/logs; mkdir -p "$LOGDIR"
TS=$(date +%H%M%S); LOG="$LOGDIR/l3_sep16_$TS.log"

echo "=== [1/4] pre-flight: token GOOD + FLAT ==="
$PY plugins/dnse/tools/token_status.py 2>&1 | tail -2
$PY plugins/dnse/tools/venue.py flat >/dev/null 2>&1; FLAT=$?
[ "$FLAT" -ne 0 ] && { echo "!!! NOT flat/clean ($FLAT) — ABORTING."; $PY plugins/dnse/tools/venue.py status 2>&1|sed 's/\x1b\[[0-9;]*m//g'; exit "$FLAT"; }
echo "  FLAT + clean."

echo "=== [2/4] L0 gate (exit 0) ==="
$PY plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py 2>&1 | tee "$LOGDIR/l3_sep16_l0_$TS.log"
L0=${PIPESTATUS[0]}; [ "$L0" -ne 0 ] && { echo "!!! L0 FAILED ($L0) — ABORTING."; exit "$L0"; }

echo "=== [3/4] l3_fill_fixed_sl_tp --broker @1m (maxHoldBars=45, timeout 3300s) ==="
echo "    watch: FILLED leg=entry -> next bar EXIT dispatch (reactive, expected) -> SL/TP RESTS"
echo "    then: a leg FIRES (conditional Activates -> child fills -> position closes) = THE PROOF"
echo "    or: venue cancels bracket -> #124 re-arm + #128-OBS metadata (also a win)"
timeout 3300 $PYNE run plugins/dnse/testing/live_test/l3_fill_fixed_sl_tp.py \
    dnse_broker:VN30F1M@1 --broker 2>&1 | tee "$LOG"
RC=${PIPESTATUS[0]}; echo "  pyne exit=$RC"

echo "=== [4/4] teardown ==="
$PY plugins/dnse/tools/venue.py status 2>&1 | sed 's/\x1b\[[0-9;]*m//g'
$PY plugins/dnse/tools/venue.py flat; FLATEND=$?
strip(){ sed 's/\x1b\[[0-9;]*m//g'; }
echo; echo "================ l3 SEP16 SUMMARY ================"
echo "-- entry fill:";        strip < "$LOG" | grep -aoE "event FILLED[^[]*leg=entry" | tail -2
echo "-- bracket dispatch:";  strip < "$LOG" | grep -aoE "dispatching EXIT[^[]*" | tail -3
echo "-- leg fired? (Activated->child / exit fill):"; strip < "$LOG" | grep -aoE "(conditional ACTIVATED[^[]*|event FILLED[^[]*leg=(tp|sl|close)[^[]*)" | tail -5
echo "-- #124 re-arm / #128-OBS:"; strip < "$LOG" | grep -aE "re-arming|#124-OBS|#128-OBS|quarantined" | tail -6
echo "flat@end: exit $FLATEND"
[ "$FLATEND" -ne 0 ] && echo "!!! NOT FLAT — flatten in the DNSE app or venue.py cancel <id>."
exit 0
