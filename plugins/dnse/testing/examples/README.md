# Running DNSE's own SDK examples against the fake venue

```bash
.venv/bin/python plugins/dnse/testing/examples/run_against_fake.py use-cases/market-data.py
```

## Why these are worth running

Every conformance target this fake has had so far was **our reading** of the venue — our pins,
written by the same people who wrote the thing under test. The vendor's examples are an
**external definition of the contract**, written by the people who run the venue. An example that
fails here is evidence about the fake, or about our reading, in a way none of our own tests can
be.

That paid off immediately. Reading them, before running anything, surfaced two wrong payload
shapes; running them surfaced four routes that ignored their own parameters.

## What runs

| example | result |
|---|---|
| `use-cases/market-data.py` | **runs** — security definitions and price bands |
| `use-cases/portfolio-check.py` | **runs** — investor id, accounts, per-asset-class balances |
| `use-cases/order-history.py` | **runs** — history window and positions |
| `reference/orders-email-otp.py` | **runs** — the full lifecycle: place, read back, amend, cancel |
| `reference/orders-smart-otp.py` | **runs** — same lifecycle, smart-OTP token type |
| `use-cases/place-a-trade.py` | **reaches the WebSocket step and stops there** |
| `use-cases/auto-trader.py` | same WebSocket dependency |

**The examples are not modified.** They already read `DNSE_BASE_URL` from the environment, so
pointing them at the fake is configuration. The runner starts the venue, exports the environment
and hands over. Anything that still fails is a real difference between the fake and the venue the
vendor documented, which is the point of running them at all.

They run against the **vendored** SDK, never a pip install: the vendored directory goes on the
path first, so `import dnse` resolves to the pinned copy.

## The WebSocket examples, and why they are not a fake problem

```
ValueError: ssl argument is incompatible with a ws:// URI
```

Measured, by driving the vendor's own example. The vendored connection passes an SSL context
unconditionally, and the pinned websockets version refuses that against a `ws://` address, so no
local plain-WS server can ever be reached by the real client. This is the known gap; the route
off it is the injected client factory added under #160. Nothing about the fake can fix it, and
bending the fake to look fixed would be worse than the gap.

## One shim the runner applies, and why it is not a patch to anything

```
TypeError: HTTPConnection.__init__() got an unexpected keyword argument 'assert_hostname'
```

The vendored SDK builds its pool as `urllib3.PoolManager(..., assert_hostname=False)`. On
urllib3 2.x that keyword reaches the connection class: `HTTPSConnection` accepts it,
`HTTPConnection` does not. **So the SDK as shipped can talk to production over TLS and to nothing
at all over plain HTTP** — every example dies on its first request against any local server.

The runner drops that one keyword for plain-HTTP pools and changes nothing else. It is the same
correction the plugin's own client wrapper already makes, for the same reason: that wrapper
replaces the SDK's pool outright because it also ships `cert_reqs=CERT_NONE`, i.e. unverified
HTTPS, which is unacceptable for a live trading client. The vendored SDK stays pristine and the
examples stay the vendor's own code.

## What the examples found

### Two payload shapes that were wrong, both latent

| endpoint | served | the venue serves |
|---|---|---|
| `/accounts` | `accountNo`, no investor id | `id` plus `investorId`, `dealAccount`, `derivativeAccount` |
| `/price/{symbol}/secdef` | a bare dict, no `basicPrice`, no `securityGroupId` | a **list**, one row per board, with both |

Both were invisible for the same reason the OHLC shape was: the plugin never exercised them. It
resolves an account only when `account_no` is unset, and a fake run always pins it. And
`securityGroupId` is how `classify_market_type` answers **authoritatively** — without it every
classification against the fake fell through to the symbol-prefix guess, which calls any dated
derivative contract a stock. #119/G1 exists because a guess must never scale a price, so a fake
that forces the guess was teaching the wrong lesson in exactly the wrong place.

### Four routes that ignored their own parameters

A route that ignores a parameter is the same defect class as one that serves the wrong shape: it
answers, the answer is well-formed, and it is not an answer to the question. Unlike a wrong shape
it chokes nobody, so it is only ever found by asking two different questions and noticing the
same reply.

| route | ignored | now |
|---|---|---|
| `*/secdef` | `boardId` | answers for the board asked for |
| `*/positions` | `marketType` | a STOCK query is not answered with derivative rows |
| `*/loan-packages` | `marketType`, `symbol` | the package names what it is for |
| `*/orders/history` | `from`, `to` | the window is honoured, and `total` agrees with the rows |

### Four routes that did not exist

`*/balances`, `*/orders/history`, `/registration/send-email-otp` and
`/registration/trading-token`. The fake 404s everything the plugin does not call, which is
deliberate, so each of these was a decision rather than an oversight — and each is served now
because a vendor example needs it.

## Provenance and design choices

**Order history ids are DATE-PREFIXED** (`20260918_101158`), reproduced rather than smoothed.
That prefix is why a cross-day numeric id does not resolve on the cancel endpoint and why the
venue tool falls back to history for previous-day ids. A fake serving bare same-day ids would
make that whole behaviour untestable.

**The OTP mint accepts the published sandbox constant `666666` and nothing else.** No secret
exists anywhere in this path and none is required. Entering a real one-time code is prohibited
for every agent, and a fake that invented a credential flow would teach precisely the habit that
rule exists to prevent.

**DESIGN CHOICE — the balances payload.** Its shape is taken from DNSE's published sample
(`docs/dnse-openapi-documentation/dnse-get-account-balances.md`): nested per asset class, not a
flat set of cash fields, which was the shape assumed before the sample was read. Nothing in this
plugin reads balances, so the *values* are invented and only the *shape* is sourced. What would
settle the values is one read against the real account; until then no test should assert on a
number from here.

**DESIGN CHOICE — one instrument.** The venue replays a single contract, so `market-data.py`
prints the same band for every symbol it asks about. The alternative, fabricating a plausible row
per symbol, would be worse: it would invent prices for instruments this venue does not carry.

## The token cache, and why the runner redirects it

An example that mints a trading token caches it through `token_store`. Its default path is **next
to its own module** — `examples/upstream/.trading_token.json` — not the working directory
(`token_store.py:31-34`), so it lands inside the tree at a fixed path no matter where the runner
is invoked from. The first run here duly wrote one, and it had to be parked.

The contents are this fake's nonsense token. The file name, the `0600` mode and the shape are a
real credential cache, and **the habit of one appearing in the repo is the hazard, not the
bytes.** `token_store` honours `DNSE_TOKEN_CACHE`, so the runner points it at a temporary
directory and the file never exists in the tree at all. That is worth more than the `.gitignore`
entry, which only prevents the commit after the mistake has already been made.

If you drive an example some other way, set `DNSE_TOKEN_CACHE` yourself.

## Why these stay in the suite

Six defects were found in this fake on 2026-09-18. Three came from our own pins and three from
the vendor's examples, and they were not the same kind of defect.

**The three our pins found were shape mismatches that crashed a reader.** A payload the consuming
code could not parse: it raised, or it fell through a guard and answered `False`. Loud enough to
chase once something exercised the path.

**The three the examples found answered cleanly and were simply not answers to the question.** An
account payload keyed on the wrong field. A security definition missing the field that decides
whether a classification is authoritative or a guess. Four routes that accepted a parameter and
never read it. Nothing crashed. Every response was well-formed. They were all wrong.

Our own tests cannot find that class, because **we write both the question and the answer.** When
the same understanding produces the fake and the test of the fake, a shared misreading is
invisible from the inside — the test agrees with the fake precisely because both came from the
same head. The vendor's examples are the only input here written by the people who run the venue,
so they are the only place a misreading of ours can show up as a failure rather than as agreement.

That is the argument for keeping them in the suite rather than treating this as a one-off.

## The upstream copies

`upstream/` holds the examples exactly as published, so a future run can be diffed against them
and so it is always visible that nothing here was edited to make it pass. See `VENDOR_INFO.txt`
for the source and the date they were taken.
