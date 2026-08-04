# skyrl-bird-sql-v1

Native Verifiers v1 environment for the SkyRL-SQL interaction protocol over
the pinned ReViSQL/BIRD-Verified tasks and BIRD SQLite databases.

Set `POSTTRAIN_SKYRL_BIRD_CACHE` to a persistent directory, then prepare and
validate the immutable assets:

    python -m skyrl_bird_sql_v1.assets prepare
    python -m skyrl_bird_sql_v1.assets validate

Preparation downloads roughly 20.7 GB. It is locked, hash-checked,
safe-extracted, atomically published, and idempotent. The runtime needs no SQL
server: every query executes against a read-only local SQLite database.
