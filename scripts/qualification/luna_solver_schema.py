"""Constrain preparation syntax without asserting semantic correctness."""


def instruction_schema(stage, data):
    stage = stage.removesuffix("_repair")
    text = {"type": "string"}

    def array(items):
        return {"type": "array", "items": items}

    def obj(fields):
        return {"type": "object", "properties": fields, "required": list(fields), "additionalProperties": False}

    def enum(values):
        return {"type": "string", "enum": list(dict.fromkeys(values))}

    if stage == "requirements":
        original_checks = data["manifest"]["checks"]
        checks = [c for c in original_checks if c["role"] in ("goal", "harm")]
        row = obj(
            {
                "check_id": enum(c["check_id"] for c in checks),
                "role": enum(["goal", "harm"]),
                "instruction": text,
                "evidence": array(enum(f"/checks/{i}" for i, c in enumerate(original_checks)
                                       if c["role"] in ("goal", "harm"))),
            }
        )
        return obj({"requirements": {**array(row), "minItems": len(checks), "maxItems": len(checks)}})
    if stage == "workflow":
        actions = data["actions"]
        return obj(
            {
                "actions": array(
                    obj(
                        {
                            "action_id": enum(a["action_id"] for a in actions),
                            "tool": enum(a["call"]["name"] for a in actions),
                            "disposition": enum(["reuse", "failed_attempt", "uncertain"]),
                            "instruction": text,
                            "depends_on": array(enum(a["action_id"] for a in actions)),
                        }
                    )
                )
            }
        )
    if stage == "assembly":
        return obj(
            {
                "steps": array(
                    obj(
                        {
                            "step_id": text,
                            "instruction": text,
                            "depends_on": array(text),
                            "check_ids": array(enum(r["check_id"] for r in data["requirements"]["requirements"])),
                            "action_ids": array(enum(a["action_id"] for a in data["workflow"]["actions"])),
                        }
                    )
                )
            }
        )
    if stage == "review":
        checks = [c for c in data["manifest"]["checks"] if c["role"] in ("goal", "harm")]
        return obj(
            {
                "checks": {
                    **array(
                        obj(
                            {
                                "check_id": enum(c["check_id"] for c in checks),
                                "covered": {"type": "boolean"},
                                "reason": text,
                            }
                        )
                    ),
                    "minItems": len(checks),
                    "maxItems": len(checks),
                },
                "workflow_faithful": {"type": "boolean"},
                "no_private_literals": {"type": "boolean"},
                "approved": {"type": "boolean"},
                "issues": array(text),
            }
        )
    raise ValueError("unknown preparation stage")
