# reminis

**Your model's weights are just data. Store them in a database.**

`reminis` converts any GGUF model into a SQLite database where every tensor becomes a queryable, versionable, diffable row. Convert back to GGUF when you're done.

```bash
pip install reminis
```

## Why

The ML world treats model weights as opaque files. You save the whole thing, load the whole thing, and if something goes wrong, you retrain from scratch.

Once weights are in a database, you get — for free — everything that 40 years of database engineering has built: queries, rollback, diffs, branching, merging, audit logs, access control, replication.

## Quick Start

```bash
# Convert a GGUF model to SQLite
reminis convert model.gguf

# Convert back to GGUF
reminis export model.db -o model_restored.gguf
```

## Roadmap

- [x] Publish to PyPI
- [ ] GGUF to SQLite converter
- [ ] SQLite to GGUF back-converter
- [ ] Fine-tune tracking with edit logs
- [ ] Surgical rollback of bad training steps
- [ ] Weight diffing between model versions
- [ ] Delta-based model distribution (weight migrations)
- [ ] Model merging via SQL operations
- [ ] Unsloth integration

## License

MIT
