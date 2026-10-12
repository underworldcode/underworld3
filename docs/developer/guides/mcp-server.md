# The transcript MCP server

A local, read-only MCP server lets an AI assistant ask narrow questions
about a run from its transcript instead of reading the file, and instead
of reconstructing the model from source. It is a thin projection of
`uw.Transcript` and the description layer: every tool reads a transcript
and returns the query's own answer as YAML, so what the assistant is told
is what the digest, a notebook and a test read. Nothing in it runs a
model or writes to one.

## Running it

```bash
python -m underworld3.mcp        # speaks MCP on stdio
```

The repository's `.mcp.json` registers it for Claude Code under the name
`underworld`, through `scripts/mcp-server.sh`, which starts it in the
checkout's pixi environment. The server needs the `mcp` package (2.x);
until it is in `pixi.toml`, install it into the environment once:

```bash
pixi run -e <env> python -m pip install mcp
```

## The tools

| tool | answers |
|---|---|
| `uw_transcript_list` | runs under a directory, newest first, with steps, abandoned and backtrack counts |
| `uw_transcript_summary` | one run: scales, counts, patterns, the parts by name |
| `uw_transcript_steps` | rows in sequence order, paged: index, label, dt, wall, operators, outcomes |
| `uw_transcript_patterns` | the run collapsed to its distinct step patterns |
| `uw_transcript_problems` | abandoned steps with what stopped them, backtracks with reasons, failed and capped solves |
| `uw_transcript_events` | events filtered by kind, part, outcome or step |
| `uw_transcript_step` | one step, with `attempt` for a step rewound and retried |
| `uw_transcript_compare` | what differs between two steps |
| `uw_transcript_parts` | the solvers that acted, and when their form or values changed |
| `uw_transcript_part` | what a part was solving at a step: `summary`, `forms` (LaTeX), or `exact` |
| `uw_transcript_key` | the key to the run, Markdown or text |
| `uw_transcript_adjoint_segments` | the run partitioned by adjoint support |
| `uw_describe_render` | any description record in another form |
| `uw_capabilities` | every solver family with its equation templates, terms and conditions, every constitutive model with its parameters, every history scheme |
| `uw_capability` | one family in full, documentation included, as markdown, text or yaml |

`path` may be a transcript file, a run directory, or a `transcripts`
directory, in which case the latest run is read.

## Why it is shaped this way

The answers are compact by default and drill down on request: a summary
names the parts, `uw_transcript_part` with `detail="summary"` gives the
terms and values, `detail="forms"` the residual as LaTeX, `detail="exact"`
the whole record. Large symbolic expressions reach the assistant only when
asked for. The interpretation is the transcript's own: a solve is
`capped` here exactly when the figure marks it amber.

The server reads files, so it answers about runs that have happened, on
this machine, including runs still in progress. Live-object introspection
(`stokes.view()`, `stokes.adjoint_view()`) stays in the session that owns
the objects; the part records carry the same description the live objects
give, so the two agree.
