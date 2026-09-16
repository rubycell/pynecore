#!/usr/bin/env bash
# Live-L3-F13-WsVsPollLatency — ONE ARM per invocation.
#
# THE QUESTION: how much sooner does a NORMAL-book fill reach the engine via the
# WS push than via the 0.5 s REST poll?
#
# WHY ONE ARM PER RUN: the engine dedups a fill across transports (_scan_row is
# the single read-modify-write of the shared _last_seen watermark), so whichever
# transport arrives SECOND produces no event at all — the poll's arrival time in
# a WS-enabled run is invisible BY CONSTRUCTION. The two arms are therefore two
# runs, one config flag apart. Running both in one invocation was rejected: a
# crash between arms would leave the live config flipped.
#
#   run_f13_latency.sh --arm ws    [--fills N]   # enable_ws_order_events = true
#   run_f13_latency.sh --arm poll  [--fills N]   # enable_ws_order_events = false
#
# VEHICLE: l2_fill_flatten (Live-L2-SingleFill, PASS 08-12) — a `var traded`
# enter-once latch: exactly ONE market entry per run, then flatten. A market
# entry is a NORMAL-book order, which #130 requires (conditional-book events
# never stream, so a stop entry would measure nothing on the WS arm). N samples
# = N runs of that proven vehicle; no new strategy is authored for this.
#
# GRADING RULE (#134, formal): arm ws is graded on DELIVERY — the
# "WS ORDER SOURCE FIRST LIVE FRAME" line — never on the subscribe line, which
# the venue can ACK for a channel it never honours (#131).
#
# CONFIG SAFETY: this edits an UNTRACKED live config. It backs the file up
# first, refuses to start from an ambiguous baseline, and restores + verifies
# byte-identical on EVERY exit path including Ctrl-C. A crash must never leave
# the live config in the wrong transport mode — a flipped flag that survives is
# indistinguishable from a deliberate setting.
set -u
cd "$(dirname "$0")/../../../.."
PY=.venv/bin/python; PYNE=.venv/bin/pyne
CONFIG=workdir/config/plugins/dnse_broker.toml
FLAG=enable_ws_order_events
BACKUP_DIR="$(git rev-parse --show-toplevel)/backup/f13_config"
MARKER="$BACKUP_DIR/.in_flight"
LOGDIR=plugins/dnse/testing/live_test/logs; mkdir -p "$LOGDIR" "$BACKUP_DIR"
TS=$(date +%H%M%S)

ARM=""; FILLS=1
while [ $# -gt 0 ]; do
    case "$1" in
        --arm)   ARM="${2:-}"; shift 2 ;;
        --fills) FILLS="${2:-1}"; shift 2 ;;
        *) echo "unknown argument: $1"; exit 2 ;;
    esac
done
case "$ARM" in
    ws)   FLAG_VALUE=true ;;
    poll) FLAG_VALUE=false ;;
    *) echo "usage: $0 --arm ws|poll [--fills N]"; exit 2 ;;
esac

# ---------------------------------------------------------------- guards ----
echo "=== [0/4] config guards ==="
[ -f "$CONFIG" ] || { echo "!!! $CONFIG missing — ABORTING."; exit 2; }
if [ -e "$MARKER" ]; then
    echo "!!! An IN-FLIGHT marker exists: $MARKER"
    echo "    A previous run did not restore the config. REFUSING to touch it —"
    echo "    inspect the config and the backup named in that marker by hand."
    cat "$MARKER"; exit 2
fi
CURRENT=$(grep -oE "^[[:space:]]*${FLAG}[[:space:]]*=[[:space:]]*\w+" "$CONFIG" || true)
if [ -n "$CURRENT" ]; then
    echo "!!! Baseline is AMBIGUOUS: $CONFIG already carries -> ${CURRENT# }"
    echo "    Expected the flag ABSENT (it defaults to true in code). This is"
    echo "    either a leftover from a crashed run or a deliberate operator"
    echo "    setting, and this script cannot tell which. REFUSING — restoring"
    echo "    onto an unknown baseline is how a live config silently drifts."
    exit 2
fi
# NOTE for whoever opens the toml: it carries a COMMENTED template line
# (`#enable_ws_order_events = true`). The check above is line-start anchored, so
# a commented line is inert and correctly ignored — only an ACTIVE assignment
# counts as a dirty baseline.
echo "  baseline clean: no ACTIVE '$FLAG' assignment (code default = true;"
echo "                  a commented template line may exist in the toml — inert)"

BACKUP="$BACKUP_DIR/dnse_broker.toml.$(date +%Y%m%d_%H%M%S).bak"
cp -p "$CONFIG" "$BACKUP" || { echo "!!! backup FAILED — ABORTING."; exit 2; }
echo "  backup: $BACKUP"

RESTORED=0
restore() {
    local rc=$?
    # Idempotent: the trap fires on INT *and then* on EXIT, and a restore that
    # logs twice reads like it ran twice. Copying the same backup twice is
    # harmless; the misleading log is not.
    [ "$RESTORED" -eq 1 ] && exit "$rc"
    RESTORED=1
    if [ -f "$BACKUP" ]; then
        cp -p "$BACKUP" "$CONFIG"
        if cmp -s "$BACKUP" "$CONFIG"; then
            echo "=== config RESTORED and verified byte-identical to $BACKUP ==="
            [ -e "$MARKER" ] && mv "$MARKER" "$MARKER.completed.$(date +%H%M%S)"
        else
            echo "!!! RESTORE VERIFICATION FAILED — $CONFIG does NOT match"
            echo "!!! $BACKUP — fix by hand BEFORE any further live run."
        fi
    fi
    exit "$rc"
}
trap restore EXIT INT TERM

printf 'arm=%s flag=%s=%s started=%s backup=%s\n' \
    "$ARM" "$FLAG" "$FLAG_VALUE" "$(date +%H:%M:%S)" "$BACKUP" > "$MARKER"

# ---------------------------------------------------------------- gates ----
echo "=== [1/4] session gates ==="
$PY plugins/dnse/tools/token_status.py 2>&1 | grep -E 'VERDICT'
$PY plugins/dnse/tools/venue.py flat >/dev/null 2>&1; FLATRC=$?
[ "$FLATRC" -ne 0 ] && { echo "!!! not provably flat (exit $FLATRC; 2 = COULD NOT DETERMINE, never 'no') — ABORTING."; exit "$FLATRC"; }
echo "  FLAT + clean."
$PY plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py 2>&1 | tee "$LOGDIR/f13_${ARM}_l0_$TS.log" | tail -2
L0=${PIPESTATUS[0]}; [ "$L0" -ne 0 ] && { echo "!!! L0 FAILED ($L0) — ABORTING."; exit "$L0"; }

# ------------------------------------------------------------- set arm ----
echo "=== [2/4] set the arm ==="
printf '%s = %s\n' "$FLAG" "$FLAG_VALUE" >> "$CONFIG"
WROTE=$(grep -oE "^[[:space:]]*${FLAG}[[:space:]]*=[[:space:]]*\w+" "$CONFIG")
[ -z "$WROTE" ] && { echo "!!! flag write did NOT read back — ABORTING."; exit 2; }
echo "  wrote and read back: ${WROTE# }"

# --------------------------------------------------------------- fills ----
echo "=== [3/4] $FILLS fill(s) via l2_fill_flatten (ONE market entry each) ==="
for i in $(seq 1 "$FILLS"); do
    LOG="$LOGDIR/f13_${ARM}_fill${i}_$TS.log"
    echo "--- fill $i/$FILLS -> $LOG ---"
    timeout 900 $PYNE run plugins/dnse/testing/live_test/l2_fill_flatten.py \
        dnse_broker:VN30F1M@1 --broker 2>&1 | tee "$LOG" >/dev/null
    echo "  pyne exit=${PIPESTATUS[0]}"
    $PY plugins/dnse/tools/venue.py flat >/dev/null 2>&1; F=$?
    [ "$F" -ne 0 ] && { echo "!!! NOT FLAT after fill $i (exit $F) — STOPPING the ladder."; break; }
    echo "  flat after fill $i."
done

# -------------------------------------------------------------- report ----
echo "=== [4/4] evidence ($ARM arm) ==="
strip(){ sed 's/\x1b\[[0-9;]*m//g' "$1"; }
for LOG in "$LOGDIR"/f13_${ARM}_fill*_$TS.log; do
    [ -f "$LOG" ] || continue
    echo "-- $(basename "$LOG")"
    echo "   transport milestone (#134 DELIVERY evidence):"
    strip "$LOG" | grep -aoE '\[BROKER\] WS ORDER SOURCE FIRST LIVE FRAME[^[]{0,60}' | head -1
    strip "$LOG" | grep -aoE 'WS order feed (disabled by config|subscribe REQUESTED)[^[]{0,40}' | head -1
    echo "   WS frame arrivals:"
    strip "$LOG" | grep -aE 'order frame via WS' | head -4
    echo "   engine fill event:"
    strip "$LOG" | grep -aoE 'event FILLED[^[]*leg=entry' | head -2
done
echo
echo "GRADE IT LIKE THIS:"
echo "  arm ws   -> the FIRST LIVE FRAME line MUST be present. If it is absent the"
echo "              WS transport did NOT work, whatever the subscribe line says (#134)."
echo "  arm poll -> 'WS order feed disabled by config' MUST be present, and there"
echo "              must be NO WS frame lines at all; otherwise the arm is not poll-only."
echo "  latency  = T(event FILLED) - T(venue fill). Take the venue fill time from"
echo "             the order record: venue.py order <id>. Arm ws also splits into"
echo "             venue->frame and frame->event using the 'order frame via WS' line."
if [ "$FILLS" -eq 1 ]; then
echo "  n=1 CAVEAT: the POLL arm's latency is ~uniform over its 0.5 s poll grid, so a"
echo "             single sample can land anywhere in [0, 0.5 s] + RTT. Report n=1 as an"
echo "             ANECDOTE with that bound stated — it is not a mean and must not be"
echo "             presented as one. Use --fills 3+ for a comparison worth the name."
fi
