"""
The three FitFindr tools.

Each one is a standalone function you can call and test on its own, before any
of them are wired into the loop. Build and test them one at a time — three
untested tools joined by a loop is one problem that looks like six, because you
can't tell which layer is lying to you.

    search_listings(description, size, max_price)  → list[dict]
    suggest_outfit(new_item, wardrobe)             → str
    create_fit_card(outfit, new_item)              → str

All three are stubs right now. They run and they do nothing — that's the
starting position and it's deliberate.

⚠️ Before you write any of them, fill in the **Tool Inventory** section of your
README (Milestone 2). Four lines per tool: what it does, each input with its
type, exactly what it returns, and what it returns when it has nothing to give.
That last line is what your loop branches on. "Returns a list" earns nothing —
the description has to say what is *in* the list.
"""

import re

import config  # noqa: F401 — you'll use this in search_listings
from generate import generate
from utils.data_loader import load_listings


# ── Tool 1: search_listings ───────────────────────────────────────────────────

_STOPWORDS = {
    "a", "an", "and", "the", "for", "with", "under", "over", "in", "of"
}

def _keywords(text: str) -> set[str]:
    """Lowercase words worth matching on, stopwords removed."""
    words = re.findall(r"[a-z0-9']+", (text or "").lower())
    return {w for w in words if w not in _STOPWORDS and len(w) > 1}

def _size_tokens(size: str) -> set[str]:
    cleaned = re.sub(r"\([^)]*\)", " ", size or "") # drop parentheticals
    parts = [p.strip().upper() for p in cleaned.split("/")]
    return {p for p in parts if p}

def _size_matches(wanted: str, listing_size: str) -> bool:
    if not wanted:
        return True
    listing_tokens = _size_tokens(listing_size)
    if any(token.startswith("ONE SIZE") for token in listing_tokens):
        return True
    return bool(_size_tokens(wanted) & listing_tokens)


def search_listings(
    description: str,
    size: str | None = None,
    max_price: float | None = None,
) -> list[dict]:
    """
    Search the listings data for items matching a description, and optionally a
    size and a price ceiling.

    This is the tool that doesn't call the model, which makes it the easiest one
    to test and the one to move onto MCP in unit 4.

    Args:
        description: keywords describing what the user wants
                     (e.g. "vintage graphic tee").
        size:        a size string to filter by, or None to skip size filtering.
                     Match case-insensitively — "M" should match "S/M".

                     ⚠️ Read the sizes in the data before you reach for a plain
                     substring test. `"s" in "us 9"` is True, and so is
                     `"l" in "xl"`. A filter that returns shoes when someone
                     asked for a small top reads like a broken search, and it
                     will quietly cost you in unit 4 when you test criterion 1.
                     What counts as a size match is part of your spec — decide
                     it and write it into your Tool Inventory.
        max_price:   maximum price, inclusive, or None to skip price filtering.

    Returns:
        A list of matching listing dicts, best match first.
        **Returns an empty list when nothing matches — an empty list, not None,
        and not an exception.** Your loop branches on this.

    Each listing dict has these fields:
        id, title, description, category, style_tags (list), size,
        condition, price (float), colors (list), brand (str or None), platform

    Note that `brand` is None for most listings. That is deliberate and
    realistic — thrift listings often have no brand. If something you write
    assumes a brand is always there, you will find out in unit 4.

    TODO:
        1. Load every listing with load_listings().
        2. Filter by max_price and by size, when each is provided.
        3. Score what's left by keyword overlap with `description`.
        4. Drop anything scoring zero.
        5. Sort by score, highest first, and return the listing dicts —
           at most config.SEARCH_RESULT_LIMIT of them.

    Test it from a terminal before you move on:
        python -c "from tools import search_listings; print(search_listings('graphic tee', max_price=30))"
    """
    wanted = _keywords(description)
    results = []

    for listing in load_listings():
        # 1. price ceiling (inclusive)
        if max_price is not None and listing["price"] > max_price:
            continue

        # 2. size
        if size and not _size_matches(size, listing.get("size", "")):
            continue

        # 3. score by keyword overlap. A word in the title or style tags is
        #    worth 2 points, a word anywhere else is worth 1. Each search word
        #    counts once, at the best place it appears.
        strong = _keywords(" ".join([
            listing.get("title") or "",
            " ".join(listing.get("style_tags") or []),
        ]))
        weak = _keywords(" ".join([
            listing.get("description") or "",
            listing.get("category") or "",
            " ".join(listing.get("colors") or []),
            listing.get("brand") or "",      # brand is None for most listings
        ]))
        score = 2 * len(wanted & strong) + len((wanted & weak) - strong)

        # 4. drop anything that scored zero
        if score > 0:
            results.append((score, listing))

    # 5. best score first, cheaper first on a tie
    results.sort(key=lambda pair: (-pair[0], pair[1]["price"]))
    return [listing for _, listing in results[: config.SEARCH_RESULT_LIMIT]]


# ── Tool 2: suggest_outfit ────────────────────────────────────────────────────

def suggest_outfit(new_item: dict, wardrobe: dict) -> str:
    """
    Given a thrifted item and the user's wardrobe, suggest one or two outfits.

    This one calls the model, through `generate()`. You don't need to think
    about rate limits — the adapter handles pacing for you.

    Args:
        new_item: a listing dict — the item the user is considering.
        wardrobe: a wardrobe dict with an 'items' key holding a list of items.
                  **It may be empty.** Handle that.

    Returns:
        A non-empty string with outfit suggestions.
        With an empty wardrobe, return general styling advice rather than
        raising or returning "". Unit 4 has you trigger the empty wardrobe on
        purpose, so decide now what it should do.

    TODO:
        1. Check whether wardrobe['items'] is empty.
        2. If it is, ask the model for general styling ideas for this item.
        3. If it isn't, format the wardrobe items into the prompt and ask for
           specific combinations naming pieces the user already owns.
        4. Return the model's response.

    Test it from a terminal before you move on:
        python -c "from tools import suggest_outfit; from utils.data_loader import get_example_wardrobe, load_listings; print(suggest_outfit(load_listings()[0], get_example_wardrobe()))"
    """
    item_text = (
        f"Title: {new_item.get('title')}\n"
        f"Category: {new_item.get('category')}\n"
        f"Colors: {', '.join(new_item.get('colors') or [])}\n"
        f"Style tags: {', '.join(new_item.get('style_tags') or [])}\n"
        f"Description: {new_item.get('description')}"
    )

    owned = (wardrobe or {}).get("items") or []

    if not owned:
        # Empty wardrobe: general advice instead of failing.
        prompt = (
            "I just found this secondhand piece:\n"
            f"{item_text}\n\n"
            "I haven't told you what else I own. Give one or two outfit ideas "
            "built around this piece, using common basics anyone might have "
            "(for example jeans, white sneakers, a plain tee). Keep it short."
        )
    else:
        wardrobe_text = "\n".join(
            f"- {w.get('name')} ({w.get('category')}; "
            f"{', '.join(w.get('colors') or [])})"
            for w in owned
        )
        prompt = (
            "I just found this secondhand piece:\n"
            f"{item_text}\n\n"
            "Here is my wardrobe:\n"
            f"{wardrobe_text}\n\n"
            "Give one or two outfit ideas that combine the new piece with "
            "pieces from my wardrobe. Name the wardrobe pieces exactly as "
            "listed, and do not suggest anything I don't own. Keep it short."
        )

    system = "You are a friendly thrift stylist. Answer in plain text, no headings."
    text = (generate(prompt, system=system) or "").strip()

    # The spec promises a non-empty string.
    return text or "Try pairing it with simple basics, like jeans and clean sneakers."


# ── Tool 3: create_fit_card ───────────────────────────────────────────────────

def create_fit_card(outfit: str, new_item: dict) -> str:
    """
    Write a short caption someone would actually post about the find.

    This calls the model too.

    Args:
        outfit:   the outfit suggestion string from suggest_outfit().
        new_item: the listing dict for the item.

    Returns:
        A two-to-four sentence caption.
        If `outfit` is empty or whitespace, return a descriptive message rather
        than raising.

    The caption should read like a real post rather than a product description,
    mention the item and its price and platform once each, and be specific about
    the vibe.

    It should also come out **differently for different inputs**. If you run
    this three times on the same item and get three word-for-word identical
    strings, it's one of two things, and both are near the top of `config.py`:

        • CACHE_ENABLED — the adapter handed back an answer it already had
        • TEMPERATURE   — at 0.0 the model gives the same words every time

    TODO:
        1. Guard against an empty or whitespace-only `outfit`.
        2. Build a prompt with the item details and the outfit.
        3. Call generate() and return the response.

    Test it from a terminal before you move on:
        python -c "from tools import create_fit_card; from utils.data_loader import load_listings; print(create_fit_card('jeans and white sneakers', load_listings()[0]))"
    """
    if not outfit or not outfit.strip():
        return "No outfit to write a caption for yet. Get an outfit suggestion first."

    price = new_item.get("price")
    price_text = f"${price:.0f}" if price == int(price) else f"${price:.2f}"
    platform = new_item.get("platform") or ""

    prompt = (
        "Write a short caption for a social post about a thrift find.\n\n"
        f"Item: {new_item.get('title')}\n"
        f"Price: {price_text}\n"
        f"Platform: {platform}\n"
        f"Outfit idea: {outfit}\n\n"
        "Rules: two to four sentences. Sound like a real person posting, not a "
        "product description. Mention the item, the price and the platform "
        "once each, and be specific about the vibe."
    )
    system = "You write casual, specific social captions. Plain text only."

    def _has_price_and_platform(text: str) -> bool:
        return bool(re.search(r"\$\s?\d", text)) and platform.lower() in text.lower()

    caption = (generate(prompt, system=system) or "").strip()

    # One retry if the caption left out the price or the platform. The reminder
    # changes the prompt, so the starter's cache does not hand back the same text.
    if not _has_price_and_platform(caption):
        retry_prompt = (
            prompt
            + f"\n\nImportant: your caption MUST include the price {price_text} "
            f"and the platform name {platform}."
        )
        retry = (generate(retry_prompt, system=system) or "").strip()
        if retry:
            caption = retry

    return caption or "Found something great, outfit details to come."


# ── Tool 4 (stretch): compare_prices ──────────────────────────────────────────

def compare_prices(item: dict, results: list[dict]) -> dict:
    """
    Compare the selected item's price with the other search results.

    Args:
        item:    the selected listing dict.
        results: the list of listing dicts search_listings returned.

    Returns:
        A dict with:
            price    (float)        the item's price
            average  (float|None)   average price of the OTHER results
            verdict  (str)          "below average", "about average" or
                                    "above average". About average means within
                                    10% of the average.
        When there are no other results to compare with, average is None and
        the verdict is "no comparison available". It never raises.
    """
    price = item["price"]
    others = [r["price"] for r in results if r.get("id") != item.get("id")]

    if not others:
        return {"price": price, "average": None, "verdict": "no comparison available"}

    average = sum(others) / len(others)
    if price < average * 0.9:
        verdict = "below average"
    elif price > average * 1.1:
        verdict = "above average"
    else:
        verdict = "about average"
    return {"price": price, "average": round(average, 2), "verdict": verdict}

