def format_price(cents, currency="USD"):
    if currency != "USD":
        raise ValueError(currency)
    sign = "-" if cents < 0 else ""
    d, c = divmod(abs(cents), 100)
    return f"{sign}${d:,}.{c:02d}"
