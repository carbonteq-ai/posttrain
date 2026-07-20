# Reusable profiles

Profiles are version-controlled starting points consumed by the `train`, `eval`, and `serve` packages.

```text
profiles/
  models/    loadable foundation or promoted derived model targets
  train/     technique-owned SFT, DPO, and GRPO defaults
  eval/      reusable general-evaluation selections
  serve/     vLLM and SGLang runtime defaults and compatibility
```

Model profiles reference engine-owned configs; they do not copy native settings or list every descendant checkpoint. Every execution captures its fully resolved model profile and engine config.

Profiles are added only through the new model/environment onboarding workflow; the prototype catalog and configuration trees were removed.
