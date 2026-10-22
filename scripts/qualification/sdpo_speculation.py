"""Explicit experimental decoding controls; no implicit runtime fallback."""


def speculative_config(method, num_tokens=2):
    if method == "none":
        return None
    if method == "mtp":
        if num_tokens not in (1, 2):
            raise ValueError("only one/two native MTP tokens selected for qualification")
        return {"method": "mtp", "num_speculative_tokens": num_tokens}
    raise ValueError("unqualified speculative method")


def validate_qualification_request(request, num_tokens, *, prefix_caching,
                                   chunked_prefill, enforce_eager, method="mtp"):
    """Reject receipt reuse across different speculative/runtime operating points."""
    expected = {
        "method": method, "speculative_config": speculative_config(method, num_tokens),
        "concurrency": 40, "context": 65536, "optimizer_updates": 0,
        "prefix_caching": prefix_caching, "chunked_prefill": chunked_prefill,
        "enforce_eager": enforce_eager,
    }
    for key, value in expected.items():
        if key not in request or request[key] != value:
            raise ValueError(f"MTP qualification operating point differs: {key}")


def batch_orders(maximum):
    if not isinstance(maximum, int) or maximum < 1:
        raise ValueError("positive concurrency required")
    orders = [[0]]
    for size in sorted({2, 4, 8, 16, 32, maximum}):
        if size <= maximum:
            orders.append([1, 0, 1, 1] if size == 4 else [0] + [1] * (size - 1))
    orders.append([0])
    return orders
