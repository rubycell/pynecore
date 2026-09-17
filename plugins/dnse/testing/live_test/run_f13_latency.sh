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

# F5: a TRAILING value-flag (`--arm` or `--fills` last) used to leave $2 unset,
# so `shift 2` failed, $# never decreased and the loop spun forever — silently,
# until the outer `timeout` killed it with rc=124. Every value flag now proves
# its argument exists first.
need_value() {
    [ $# -ge 2 ] && [ -n "${2:-}" ] || {
        echo "!!! $1 requires a value"; usage; exit 2; }
}
while [ $# -gt 0 ]; do
    case "$1" in
        --arm)         need_value "$@"; ARM="$2"; shift 2 ;;
        --fills)       need_value "$@"; FILLS="$2"; shift 2 ;;
        --vehicle)     need_value "$@"; VEHICLE="$2"; shift 2 ;;
        --window-bars) need_value "$@"; WINDOW_BARS="$2"; shift 2 ;;
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
    # On EVERY exit path including Ctrl-C: say whether the account is clean.
    # An interrupt lands most often DURING a run, which is exactly when a
    # position or a resting entry is most likely to be open — and the operator
    # who just pressed Ctrl-C is owed that answer without having to think of
    # asking for it.
    if [ "${GATES_PASSED:-0}" -eq 1 ]; then
        # TWO agreeing reads, like every other flat check here. This line is
        # read at the moment the operator is deciding what to do — most often
        # straight after Ctrl-C, mid-position — so a stale FLAT (#124-OBS,
        # measured) would tell them exactly the wrong thing exactly then.
        flat_confirmed; local frc=$?
        if [ "$frc" -eq 0 ]; then
            echo "=== account verified FLAT and clean on exit ==="
        else
            echo "!!! ON EXIT THE ACCOUNT IS NOT PROVABLY FLAT (venue.py flat"
            echo "!!! exit $frc; 2 = COULD NOT DETERMINE, never 'no')."
            echo "!!! FLATTEN NOW, operator — run venue.py status."
        fi
    fi
    exit "$rc"
}
trap restore EXIT INT TERM

printf 'arm=%s flag=%s=%s started=%s backup=%s\n' \
    "$ARM" "$FLAG" "$FLAG_VALUE" "$(date +%H:%M:%S)" "$BACKUP" > "$MARKER"

# ---------------------------------------------------------------- gates ----
echo "=== [1/4] session gates ==="
# F2(a): this used to be `token_status.py 2>&1 | grep VERDICT`, which threw the
# exit code away — and worse, when the verdict is bad and stdin is a tty the
# tool PROMPTS (token_status.py:159-163); through the pipe that prompt is eaten
# by grep and the runner blocks on input FOREVER, silently, at this line. So:
# stdin from /dev/null (it cannot prompt), and the verdict is read from BOTH the
# exit status and the VERDICT text — either one failing aborts. Belt and braces
# because a gate is worth more than the two lines it costs.
TOKEN_OUT=$($PY plugins/dnse/tools/token_status.py < /dev/null 2>&1); TOKENRC=$?
echo "$TOKEN_OUT" | grep -E 'VERDICT' || echo "  (no VERDICT line was printed)"
TOKEN_VERDICT=$(echo "$TOKEN_OUT" | grep -E '^VERDICT:' | head -1 | sed 's/^VERDICT:[[:space:]]*//')
if [ "$TOKENRC" -ne 0 ] || ! printf '%s' "$TOKEN_VERDICT" | grep -q '^GOOD'; then
    echo "!!! trading token is not usable (exit $TOKENRC, verdict '${TOKEN_VERDICT:-none}') — ABORTING."
    echo "    mint one: .venv/bin/python plugins/dnse/tools/refresh_token.py --send   (then --otp <code>)"
    exit 2
fi

# F2(b): a HARD session/window gate. L0 cannot serve as one — it SKIPS
# conditional probes outside a session and still exits 0
# (l0_order_semantics.py:271-278, :382) — so `--arm poll` at 14:00 previously
# passed every check and ran 45 minutes straight through the 14:30 ATC, where
# DNSE refuses cancels and fills whatever rests.
PHASE=$($PY -c "
import sys; sys.path.insert(0, 'plugins/dnse/tools')
import venue; print(venue.session_phase())" 2>/dev/null || echo "UNKNOWN")
$PY plugins/dnse/testing/live_test/f13_guard.py window \
    --phase "$PHASE" --run-timeout "$RUN_TIMEOUT"; WINRC=$?
[ "$WINRC" -ne 0 ] && { echo "!!! window gate refused (exit $WINRC) — ABORTING."; exit "$WINRC"; }

$PY plugins/dnse/tools/venue.py flat >/dev/null 2>&1; FLATRC=$?
[ "$FLATRC" -ne 0 ] && { echo "!!! not provably flat (exit $FLATRC; 2 = COULD NOT DETERMINE, never 'no') — ABORTING."; exit "$FLATRC"; }
echo "  FLAT + clean."
$PY plugins/dnse/testing/live_test/level0_venue_semantics/l0_order_semantics.py 2>&1 | tee "$LOGDIR/f13_${ARM}_l0_$TS.log" | tail -2
L0=${PIPESTATUS[0]}; [ "$L0" -ne 0 ] && { echo "!!! L0 FAILED ($L0) — ABORTING."; exit "$L0"; }

# ------------------------------------------------------------- set arm ----
# F-D: ONE `venue.py flat` is one get_position, and a stale-FLAT read is
# MEASURED on this venue (#124-OBS, #122). For the l2/fallback vehicle — no
# bracket, position held for a whole bar — a lagging FLAT would SIGTERM the
# only thing that flattens and then launch the next run over a live position.
# Two agreeing reads, 1.5 s apart, exactly as flatten.py requires before it
# acts on a sign.
flat_confirmed() {
    $PY plugins/dnse/tools/venue.py flat >/dev/null 2>&1 || return 1
    sleep 1.5
    $PY plugins/dnse/tools/venue.py flat >/dev/null 2>&1 || return 1
    return 0
}

# Past this point a run may OPEN a position, so the exit trap owes the operator
# a cleanliness verdict on every path. Before it, a flat check would be noise on
# a refusal that never touched the account.
GATES_PASSED=1

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

entry_filled() { grep -aq "event FILLED.*leg=entry" "$1"; }

# F-C: the window gate must run before EVERY run and every fallback, not once
# before the loop. `--arm poll --fills 2` at 13:00 otherwise reaches run 2 at
# ~14:00 with a 2700 s timeout and trades through the 14:30 ATC — where DNSE
# refuses cancels and fills whatever rests — having passed a gate that was
# true 75 minutes earlier.
window_open() {
    local phase
    phase=$($PY -c "
import sys; sys.path.insert(0, 'plugins/dnse/tools')
import venue; print(venue.session_phase())" 2>/dev/null || echo "UNKNOWN")
    $PY plugins/dnse/testing/live_test/f13_guard.py window \
        --phase "$phase" --run-timeout "$RUN_TIMEOUT"
}

# F6: NO vehicle self-terminates — l2 idles after its `var traded` latch, l2b
# keeps re-entering, and `pyne run --live` never exits — so `timeout` was the
# only terminator and EVERY run cost the full window+3 bars (45 min at the
# default), which is what pushed the poll arm's second run into the 14:30 ATC.
# The runner now ends a run on EVIDENCE: once an entry fill has been seen AND
# the venue reads flat, the vehicle has done its whole job, so SIGTERM it.
# Deliberately venue-truth rather than a log marker: a per-vehicle "flatten
# done" string would have to be re-guessed for every new vehicle, and the venue
# answer is the one that actually matters.
run_vehicle() {                      # $1 script  $2 log
    : > "$2"
    PYNE_NO_COLOR_LOG=1 PYTHONUNBUFFERED=1 \
    timeout "$RUN_TIMEOUT" $PYNE run "$1" \
        "dnse_broker:VN30F1M@${TIMEFRAME}" --broker \
        > >($PY -u -c "$PREFIXER" > "$2") 2>&1 &
    local pid=$! deadline=$(( $(date +%s) + RUN_TIMEOUT )) settled=0
    while kill -0 "$pid" 2>/dev/null && [ "$(date +%s)" -lt "$deadline" ]; do
        sleep 15
        entry_filled "$2" || continue
        if flat_confirmed; then
            # A fill happened AND the account is flat again: the vehicle
            # entered, protected and flattened. Nothing further to observe.
            echo "  run complete (entry fill observed, venue FLAT) — terminating"
            kill -TERM "$pid" 2>/dev/null
            settled=1
            break
        fi
    done
    wait "$pid" 2>/dev/null; local rc=$?
    [ "$settled" -eq 1 ] && rc=0
    return "$rc"
}

echo "=== [3/4] $FILLS run(s) via $VEHICLE @${TIMEFRAME}m (timeout ${RUN_TIMEOUT}s = ${WINDOW_BARS}+3 bars) ==="
for i in $(seq 1 "$FILLS"); do
    LOG="$LOGDIR/f13_${ARM}_fill${i}_$TS.log"
    echo "--- run $i/$FILLS ($VEHICLE) -> $LOG ---"
    if ! window_open; then
        echo "!!! window gate refuses run $i — STOPPING the ladder here."
        echo "    (the gate that passed before run 1 is not evidence for run $i)"
        break
    fi
    run_vehicle "$SCRIPT" "$LOG"; echo "  pyne exit=$?"

    if [ "$VEHICLE" = l2b ] && ! entry_filled "$LOG"; then
        # F1: the stop never triggered. Before ANY fallback can place a market
        # order, every entry this run dispatched must be provably cancelled and
        # the account provably flat — otherwise a stop can still trigger after
        # the fallback filled and the account holds two contracts, one of them
        # unbracketed. The decision lives in f13_guard.py because it is on the
        # order path and bash could not be pinned: the previous version parsed
        # ONE id with `tail -1` (l2b re-issues the entry every flat bar),
        # echoed the cancel's exit code without testing it, and launched the
        # fallback even down the branch that had just refused to guess an
        # unreadable id.
        printf '%.3f F13 NO-SAMPLE: stop entry unfilled after %s bars\n' \
            "$(date +%s)" "$WINDOW_BARS" >> "$LOG"
        echo "  NO-SAMPLE (stop unfilled) — cleaning up before any fallback"
        $PY plugins/dnse/testing/live_test/f13_guard.py cleanup \
            --log "$LOG" --symbol VN30F1M; CLEANRC=$?
        if [ "$CLEANRC" -ne 0 ]; then
            echo "!!! cleanup did NOT confirm (exit $CLEANRC) — no fallback, ladder STOPPED."
            echo "    FLATTEN NOW, operator: this runner cannot, and an entry may be resting."
            break
        fi
        if [ "$FALLBACK" -eq 1 ] && ! window_open; then
            echo "  fallback SKIPPED — the window has closed since this run started."
        elif [ "$FALLBACK" -eq 1 ]; then
            FB="$LOGDIR/f13_${ARM}_fill${i}fb_$TS.log"
            echo "  fallback: ONE l2 run so the arm still gets a transport sample -> $FB"
            run_vehicle "$FALLBACK_SCRIPT" "$FB"; echo "  fallback pyne exit=$?"
            printf '%.3f F13 FALLBACK transport-only (l2 in place of run %s)\n' \
                "$(date +%s)" "$i" >> "$FB"
        fi
    fi

    flat_confirmed; F=$?
    if [ "$F" -ne 0 ]; then
        echo "!!! NOT FLAT after run $i on TWO reads 1.5s apart (2 = COULD NOT DETERMINE, never 'no')"
        echo "    FLATTEN NOW, operator — this runner never flattens; it only stops."
        echo "    venue.py status, then flatten in the app or via flatten_api.py."
        break
    fi
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

# F4: capture the VENUE-CLOCK timestamps. Without these the grader can never
# compute a fill latency and the runner ALWAYS exits 2 — which teaches the
# operator that exit 2 means nothing, and that habit would disarm every other
# guard in this script. `venue.py order --json` also follows an Activated
# conditional to its normal-book CHILD, which is the record that actually
# carries the fill (#41).
VENUE_JSON="$LOGDIR/f13_${ARM}_venue_$TS.json"
# All three id shapes a real l2b run produces (measured): the initial
# dispatch, each CHASE re-place (which logs only `event CREATED … leg=entry`),
# and the normal-book CHILD the conditional activated into — which is the id
# that actually FILLED and therefore the only one carrying a fill timestamp.
ENTRY_IDS=$(for L in $LOGS; do
    sed 's/\x1b\[[0-9;]*m//g' "$L" | {
        grep -aoE "dispatched ENTRY [A-Z]+ .*-> \['[^']+'\]" | grep -oE "\['[^']+'\]" | tr -d "[]'"
    }
    sed 's/\x1b\[[0-9;]*m//g' "$L" | grep -aoE "event CREATED id=[^ ]+ .*leg=entry" | grep -oE "id=[^ ]+" | cut -d= -f2
    sed 's/\x1b\[[0-9;]*m//g' "$L" | grep -aoE "child=[^ ]+" | cut -d= -f2
done | sort -u)
if [ -n "$ENTRY_IDS" ]; then
    # shellcheck disable=SC2086
    $PY plugins/dnse/tools/venue.py order $ENTRY_IDS --json > "$VENUE_JSON" 2>/dev/null
    VJRC=$?
    if [ "$VJRC" -ne 0 ]; then
        echo "  venue record capture incomplete (exit $VJRC) — the grade will"
        echo "  report fill latency as COULD-NOT-DETERMINE rather than guess."
    fi
    echo "  venue records -> $VENUE_JSON"
    VENUE_ARG="--venue-json $VENUE_JSON"
else
    echo "  no entry ids in the logs — no venue records to capture."
    VENUE_ARG=""
fi

# shellcheck disable=SC2086
$PY "$GRADER" --arm "$ARM" $LOGS $VENUE_ARG
GRADE=$?
echo
echo "grade exit=$GRADE   (0 graded+passed / 1 a gate FAILED / 2 could not determine)"
echo "re-run any time:  $PY $GRADER --arm $ARM $LOGS [--venue-json <file>]"
echo "T(venue fill) needs the venue record: for l2b the entry is a CONDITIONAL,"
echo "so venue.py order <entry id> reports Activated and names the CHILD via"
echo "externalOrderId — the child is what filled (#41). Capture both records."
exit "$GRADE"
