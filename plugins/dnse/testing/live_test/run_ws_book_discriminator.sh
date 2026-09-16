#!/usr/bin/env bash
# #121/#107 WS book discriminator (sep16 step 5, second run).
#
# THE QUESTION the first run could not answer: on 2026-09-16 10:59 the SHORT
# order channel stayed silent while L0 placed+cancelled 8 orders inside the
# capture window — but every one of those was CONDITIONAL-book. On 09-15 the
# same channel delivered 4 `do` frames for an OCO, whose NORMAL-book child is
# born Activated. So "silent" may mean "conditional-book events never stream".
#
# The payload here is direct_probes --case t18, which touches BOTH books in one
# short run: a NORMAL LO -5% place->visible->cancel, then a conditional STOP +5%
# place->visible->cancel. One capture window, one of each. Whichever book
# produces frames IS the answer; frames from both refutes the hypothesis.
# Nothing can fill (>=4.5% away, 1 contract); t18 cleans up its own orders.
set -u
cd "$(dirname "$0")/../../../.."
PY=.venv/bin/python
LOGDIR=plugins/dnse/testing/live_test/logs; mkdir -p "$LOGDIR"
TS=$(date +%H%M%S); WSLOG="$LOGDIR/ws_book_disc_$TS.log"
SECONDS_CAP=${1:-120}

echo "=== [1/3] dual-channel WS watcher (background, ${SECONDS_CAP}s) ==="
$PY plugins/dnse/testing/live_test/probe_ws_market_data.py --trading \
    --seconds "$SECONDS_CAP" > "$WSLOG" 2>&1 &
WSPID=$!
sleep 10   # auth + subscribe MUST complete before the payload fires

echo "=== [2/3] payload: t18 — NORMAL leg then CONDITIONAL leg ==="
$PY plugins/dnse/testing/live_test/direct_probes_t14_t15_t17.py --case t18 \
    2>&1 | tee "$LOGDIR/ws_book_disc_t18_$TS.log" | sed 's/\x1b\[[0-9;]*m//g' | tail -25
T18=${PIPESTATUS[0]}
echo "  t18 exit=$T18 (its place/cancel events on BOTH books are the payload)"

echo "=== [3/3] wait out the window, then per-channel verdict ==="
wait "$WSPID" 2>/dev/null
sed 's/\x1b\[[0-9;]*m//g' "$WSLOG" | tail -30
echo
echo "READ IT AS: frames whose T-code/marketType match the NORMAL leg only"
echo "  -> conditional-book events do NOT stream; the 10:59 silence was the BOOK,"
echo "     not a broken channel. Frames for BOTH legs -> hypothesis refuted, and"
echo "     the 10:59 silence needs another explanation. NO frames at all after a"
echo "     confirmed both-book payload -> the short channel is dead TODAY, which"
echo "     would contradict 09-15 and is itself the finding."
echo "logs: $WSLOG  +  $LOGDIR/ws_book_disc_t18_$TS.log"
exit 0
