"""Search document text matching the original WebShop product converter."""


def search_document(product):
    """Keep original text/order/option normalization; omit unused raw product data."""
    options = {}
    for name, entries in (product["customization_options"] or {}).items():
        if entries is None:
            continue
        options[name.lower()] = [
            entry["value"].strip().replace("/", " | ").lower() for entry in entries
        ]
    option_text = ", and ".join(
        f"{name}: {', '.join(values)}" for name, values in options.items()
    )
    bullets = product["small_description"]
    if not isinstance(bullets, list):
        bullets = [bullets]
    if not bullets:
        raise ValueError("Original converter requires a first bullet point")
    text = " ".join(
        [product["name"], product["full_description"], bullets[0], option_text]
    ).lower()
    return dict(id=product["asin"], contents=text)
