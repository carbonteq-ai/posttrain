# External consumer work packages

`cpu_check.yaml` is executed through `posttrain-work` from installed wheels. Its
consumer-owned definition performs a deterministic CPU data partition, records
the run through local Trackio, and reads the result back through Observatory.
