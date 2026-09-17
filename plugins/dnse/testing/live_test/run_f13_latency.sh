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
#   run_f13_latency.sh --arm ws    [--fills N] [--vehicle l2b|l2]
#   run_f13_latency.sh --arm poll  [--fills N] [--window-bars N] [--no-fallback]
#
# VEHICLE (operator, 2026-09-17): l2b_fill_protect_flatten is the DEFAULT.
# l2_fill_flatten (Live-L2-SingleFill, PASS 08-12) stays as the fallback.
#   * l2  — one MARKET entry per run, then flatten. A market entry is a
#           NORMAL-book order, so it measures fill-event TRANSPORT only.
#   * l2b — pre-places the bracket on the entry bar, so each run measures the
#           chain that moves money: venue fill -> engine fill event ->
#           EXIT dispatched -> SL/TP resting. Its entry is a STOP, i.e. the
#           conditional book, which is precisely the open #130 question for the
#           entry type real strategies use: does the WS arm deliver the
#           normal-book CHILD's frame and attribute it to our entry? Either
#           answer is a measurement we need.
# A stop entry only fills on a breakout, hence --window-bars and --fallback.
#
# GRADING is NOT done here. This script launches and gates; f13_grade.py reads
# the logs afterwards. That split is deliberate: this file places live orders,
# so it carries no test affordance and no offline mode, while the grader is
# exercised against canned logs in plugins/dnse/tests/test_f13_grade.py.
#
# GRADING RULE (#134, formal): arm ws is graded on DELIVERY — the
# "WS ORDER SOURCE FIRST LIVE FRAME" line — never on the subscribe line, which
# the venue can ACK for a channel it never honours (#131).
#
# THE CLOCK (measured 2026-09-17, #146): [BROKER] lines are stamped with the
# PINE BAR time (lib/log.py:66-81), identical for every event inside a bar — a
# latency computed from them is bar-grid noise. So every run is piped through a
# wall-clock prefixer before the tee, and PYNE_NO_COLOR_LOG=1 forces the plain
# logging.StreamHandler (flushes per record) instead of Rich's batching console.
# The prefix is when OUR PROCESS PRINTED the line: an upper bound on arrival,
# including the logging path. Venue-side times come from the venue record.
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

ARM=""; FILLS=1; VEHICLE=l2b; WINDOW_BARS=6; FALLBACK=1
TIMEFRAME=5; BAR_SECONDS=300          # live-test rule: 5m+, NEVER 1m

usage() {
    cat <<'USAGE'
usage: run_f13_latency.sh --arm ws|poll [options]

  --arm ws|poll        REQUIRED. ws = enable_ws_order_events true, poll = false.
  --fills N            runs per arm (default 1).
  --vehicle l2b|l2     l2b (default) = stop entry + pre-placed bracket, measures
                       the whole fill->arm chain; l2 = market entry, transport only.
  --window-bars N      l2b only: bars to wait for the stop to fill (default 6).
                       Unfilled -> cancel our entry, log NO-SAMPLE, count 0.
  --no-fallback        do NOT run one l2 in place of a NO-SAMPLE slot
                       (default: the fallback IS run, labelled transport-only).
  --help               this text. Contacts nothing.

Grading is a separate READ-ONLY tool: f13_grade.py --arm <arm> <logs...>
USAGE
}

while [ $# -gt 0 ]; do
    case "$1" in
        --arm)         ARM="${2:-}"; shift 2 ;;
        --fills)       FILLS="${2:-1}"; shift 2 ;;
        --vehicle)     VEHICLE="${2:-}"; shift 2 ;;
        --window-bars) WINDOW_BARS="${2:-6}"; shift 2 ;;
        --fallback)    FALLBACK=1; shift ;;
        --no-fallback) FALLBACK=0; shift ;;
        # --help exits BEFORE any gate, config read or venue contact. The
        # flatten_api lesson: a loose argument check let --help fall through
        # into a live account.
        --help|-h)     usage; exit 0 ;;
        *) echo "unknown argument: $1"; usage; exit 2 ;;
    esac
done
case "$ARM" in
    ws)   FLAG_VALUE=true ;;
    poll) FLAG_VALUE=false ;;
    *) echo "!!! --arm is required."; usage; exit 2 ;;
esac
case "$VEHICLE" in
    l2b) SCRIPT=plugins/dnse/testing/live_test/l2b_fill_protect_flatten.py ;;
    l2)  SCRIPT=plugins/dnse/testing/live_test/l2_fill_flatten.py ;;
    *) echo "!!! --vehicle must be l2b or l2 (got '$VEHICLE')"; exit 2 ;;
esac
FALLBACK_SCRIPT=plugins/dnse/testing/live_test/l2_fill_flatten.py
case "$WINDOW_BARS" in (*[!0-9]*|"") echo "!!! --window-bars must be a positive integer"; exit 2 ;; esac
case "$FILLS" in (*[!0-9]*|"") echo "!!! --fills must be a positive integer"; exit 2 ;; esac
# window + 3 bars, per the F13 plan §1.
RUN_TIMEOUT=$(( (WINDOW_BARS + 3) * BAR_SECONDS ))

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
# Each run is piped through a wall-clock prefixer BEFORE the tee (#146): the
# engine's own bracket is the PINE BAR time, so without this every event in a
# bar shares one stamp and no latency is computable. PYNE_NO_COLOR_LOG=1 forces
# the plain logging.StreamHandler (flushes per emit, log.py:33-35 / :194-208)
# rather than Rich's batching console, and PYTHONUNBUFFERED=1 keeps the pipe hot
# — otherwise the prefix times buffer flushes instead of arrivals.
PREFIXER='import sys
for line in sys.stdin:
    sys.stdout.write("%.3f %s" % (__import__("time").time(), line))'

run_vehicle() {                      # $1 script  $2 log
    PYNE_NO_COLOR_LOG=1 PYTHONUNBUFFERED=1 \
    timeout "$RUN_TIMEOUT" $PYNE run "$1" \
        "dnse_broker:VN30F1M@${TIMEFRAME}" --broker 2>&1 \
        | $PY -u -c "$PREFIXER" | tee "$2" >/dev/null
    return "${PIPESTATUS[0]}"
}

entry_filled() { grep -aq "event FILLED.*leg=entry" "$1"; }

echo "=== [3/4] $FILLS run(s) via $VEHICLE @${TIMEFRAME}m (timeout ${RUN_TIMEOUT}s = ${WINDOW_BARS}+3 bars) ==="
for i in $(seq 1 "$FILLS"); do
    LOG="$LOGDIR/f13_${ARM}_fill${i}_$TS.log"
    echo "--- run $i/$FILLS ($VEHICLE) -> $LOG ---"
    run_vehicle "$SCRIPT" "$LOG"; echo "  pyne exit=$?"

    if [ "$VEHICLE" = l2b ] && ! entry_filled "$LOG"; then
        # The stop never triggered inside the window. Cancel OUR OWN entry by
        # the id OUR log recorded — never a sweep, never an id inferred from
        # the account (the netting book is shared with the operator).
        ENTRY_ID=$(sed 's/\x1b\[[0-9;]*m//g' "$LOG" \
            | grep -aoE "dispatched ENTRY [A-Z]+ .*-> \['[^']+'\]" \
            | tail -1 | grep -oE "\['[^']+'\]" | tr -d "[]'")
        printf '%.3f F13 NO-SAMPLE: stop entry unfilled after %s bars; entry_id=%s\n' \
            "$(date +%s)" "$WINDOW_BARS" "${ENTRY_ID:-UNKNOWN}" >> "$LOG"
        echo "  NO-SAMPLE (stop unfilled). entry id from our log: ${ENTRY_ID:-UNKNOWN}"
        if [ -n "$ENTRY_ID" ]; then
            $PY plugins/dnse/tools/venue.py cancel "$ENTRY_ID"; echo "  cancel exit=$?"
        else
            echo "  !!! could not read our entry id from the log — check"
            echo "      venue.py status BY HAND before the next run. Not guessing an id."
        fi
        if [ "$FALLBACK" -eq 1 ]; then
            FB="$LOGDIR/f13_${ARM}_fill${i}fb_$TS.log"
            echo "  fallback: ONE l2 run so the arm still gets a transport sample -> $FB"
            run_vehicle "$FALLBACK_SCRIPT" "$FB"; echo "  fallback pyne exit=$?"
            printf '%.3f F13 FALLBACK transport-only (l2 in place of run %s)\n' \
                "$(date +%s)" "$i" >> "$FB"
        fi
    fi

    $PY plugins/dnse/tools/venue.py flat >/dev/null 2>&1; F=$?
    [ "$F" -ne 0 ] && { echo "!!! NOT FLAT after run $i (exit $F; 2 = COULD NOT DETERMINE, never 'no') — STOPPING the ladder."; break; }
    echo "  flat after run $i."
done

# -------------------------------------------------------------- report ----
# Grading lives in f13_grade.py — a READ-ONLY tool that touches no venue and is
# pinned against canned logs (plugins/dnse/tests/test_f13_grade.py). It is
# re-runnable on these logs after the session, which this inline block never was.
echo "=== [4/4] grade ($ARM arm) ==="
GRADER=plugins/dnse/testing/live_test/f13_grade.py
LOGS=$(ls -1 "$LOGDIR"/f13_${ARM}_fill*_$TS.log 2>/dev/null)
if [ -z "$LOGS" ]; then
    echo "!!! no run logs produced — nothing to grade (COULD NOT DETERMINE)."
    exit 2
fi
# shellcheck disable=SC2086
$PY "$GRADER" --arm "$ARM" $LOGS
GRADE=$?
echo
echo "grade exit=$GRADE   (0 graded+passed / 1 a gate FAILED / 2 could not determine)"
echo "re-run any time:  $PY $GRADER --arm $ARM $LOGS [--venue-json <file>]"
echo "T(venue fill) needs the venue record: for l2b the entry is a CONDITIONAL,"
echo "so venue.py order <entry id> reports Activated and names the CHILD via"
echo "externalOrderId — the child is what filled (#41). Capture both records."
exit "$GRADE"
