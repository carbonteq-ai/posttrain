# GSM8K

- **Kind:** both (dataset + RL task contract)
- **Source:** `openai/gsm8k` (`main`)
- **Schema:** `question`, `answer` (gold ends with `#### <number>`)

## Candidate Verifiers environment contract

- **Observation:** system + user (question); system asks for `\boxed{}` final answer
- **Action:** assistant completion (chain-of-thought + boxed answer)
- **Reward:** `format_reward` (boxed/####) + `exact_match_reward` (parsed number vs gold)
- **Cheats:** empty/short answers; copying format without solving; unit mismatches

The prototype local reward callbacks were removed. Any future use must be implemented and published as a standalone Verifiers environment package, then consumed by qualified package reference.

## Pairing

- GRPO primary; optional SFT warm-start on other data first

## Evidence

- (none yet)
