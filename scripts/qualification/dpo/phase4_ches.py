"""One GPU container run: the SFT adapter check, then the CHES scores of every candidate pair.

Stops before scoring if the adapter the DPO job downloads is not, byte for byte, the adapter P9 was evaluated with
(p9_adapter_check.py). Scores with that evaluated adapter (ches_rank.py score). Writes into the run's evidence folder.
usage (container, cwd = /evidence/<run>): phase4_ches.py RUN_ADAPTER_DIR P9_ADAPTER_DIR EXPECTED_TREE_DIGEST
                                          CANDIDATES_TOK.jsonl BASE_MODEL_DIR
"""
import sys

import ches_rank
import p9_adapter_check


def main(run_dir, p9_dir, expected_tree, candidates, base_dir):
    if p9_adapter_check.main(run_dir, p9_dir, expected_tree, "p9-adapter-check.json"):
        print("P9 adapter check FAILED; not scoring", flush=True)
        return 2
    return ches_rank.score(candidates, base_dir, p9_dir, "ches-scores")


if __name__ == "__main__":
    sys.exit(main(*sys.argv[1:6]))
