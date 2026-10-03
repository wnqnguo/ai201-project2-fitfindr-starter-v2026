"""
The FitFindr planning loop.

This is the file that makes FitFindr an agent rather than a script. It decides
which tool to run next based on what the last one returned.

If your loop calls all three tools no matter what comes back, you have a list
of function calls. A loop looks at the last result before it picks the next
step. **That branch is the graded part of this unit.**

Build and test your three tools in `tools.py` first. Then come here.

    python agent.py          runs both example paths below
"""

import json
import os
import re

import config
import trace
from tools import search_listings, suggest_outfit, create_fit_card, compare_prices
from generate import ModelUnavailable


# ── session state ─────────────────────────────────────────────────────────────

def new_session(query: str, wardrobe: dict) -> dict:
    """
    A fresh session for one user interaction.

    The session is the single source of truth for a run. Every tool result goes
    in here, and the next tool reads it back out.

    You could pass values straight from one call to the next. It would work,
    and you would not be able to test it — you can't print a variable you have
    already overwritten. Going through the session is what makes the state
    visible, and unit 4 has you write a criterion about exactly that.

    Add fields if you need them.
    """
    return {
        "query": query,              # what the user typed
        "parsed": {},                # description / size / max_price you pulled out of it
        "search_results": [],        # everything search_listings returned
        "searched": False,           # True once search_listings has run
        "outfit_input_id": None,     # id of the item suggest_outfit received
        "selected_item": None,       # the one you chose — goes into suggest_outfit
        "price_comparison": None,    # what compare_prices returned (stretch)
        "wardrobe": wardrobe,        # the user's wardrobe
        "wardrobe_source": "given",  # "given", or "saved" when loaded from memory (stretch)
        "outfit_suggestion": None,   # what suggest_outfit returned
        "fit_card": None,            # what create_fit_card returned
        "error": None,               # set when the run ended early
    }


# ── query parsing ─────────────────────────────────────────────────────────────

_PRICE_PATTERNS = [
    # "under $30", "below 30", "less than $30", "max $30", "up to 30"
    r"(?:under|below|less than|max|up to|at most)\s*\$?\s*(\d+(?:\.\d+)?)",
    # a bare "$30"
    r"\$\s*(\d+(?:\.\d+)?)",
]
_SIZE_PATTERN = (
    r"\bsize\s+(us\s*\d+(?:\.\d+)?|w\d+|xxs|xs|xxl|xl|s|m|l|\d+(?:\.\d+)?)\b"
)


def parse_query(query: str) -> dict:
    """
    Pull a description, a size and a max_price out of a plain-language query,
    using regular expressions (no model call).

    The price and size phrases are cut out of the description, so the search
    only sees the item words (otherwise "30" would be searched for as a keyword).
    size and max_price are None when the query does not give them.
    """
    text = query or ""
    max_price = None
    size = None

    for pattern in _PRICE_PATTERNS:
        m = re.search(pattern, text, re.IGNORECASE)
        if m:
            max_price = float(m.group(1))
            text = text[:m.start()] + " " + text[m.end():]
            break

    m = re.search(_SIZE_PATTERN, text, re.IGNORECASE)
    if m:
        size = m.group(1).strip().upper()
        if re.fullmatch(r"\d+(?:\.\d+)?", size):   # bare number -> shoe size
            size = "US " + size
        text = text[:m.start()] + " " + text[m.end():]

    description = re.sub(r"\s+", " ", text).strip(" ,.-")
    return {"description": description, "size": size, "max_price": max_price}


# ── style memory (stretch) ────────────────────────────────────────────────────

_SAVED_WARDROBE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                               "data", "saved_wardrobe.json")


def _has_items(wardrobe) -> bool:
    return bool((wardrobe or {}).get("items"))


def _load_saved_wardrobe(path: str = _SAVED_WARDROBE):
    """The wardrobe saved by an earlier run, or None if there isn't one."""
    try:
        with open(path, encoding="utf-8") as f:
            saved = json.load(f)
    except (OSError, ValueError):
        return None
    return saved if _has_items(saved) else None


def _save_wardrobe(wardrobe: dict, path: str = _SAVED_WARDROBE) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(wardrobe, f, indent=2, ensure_ascii=False)


# ── planning loop ─────────────────────────────────────────────────────────────

def run_agent(query: str, wardrobe: dict) -> dict:
    """
    Run the loop once and return the finished session.

    Args:
        query:    what the user asked for, in plain language
                  (e.g. "vintage graphic tee under $30, size M").
        wardrobe: a wardrobe dict — get_example_wardrobe() or
                  get_empty_wardrobe() from utils/data_loader.py.

    Returns:
        The session dict. **Check session["error"] first** — if it isn't None,
        the run ended early and the later fields will still be None.

    ─────────────────────────────────────────────────────────────────────────
    TODO — build this, following the branch rule you wrote in Milestone 2.

      1. Start a session with new_session().

      2. Count the times round the loop, and call trace.check_iterations(count)
         on each one before you go again. It raises when the count passes
         MAX_ITERATIONS in config.py — see trace.py.

      3. Parse the query into a description, a size, and a max_price. Regex,
         string splitting, or asking the model are all fine — say which you
         chose in your README. Put the result in session["parsed"].

      4. Call search_listings() with what you parsed.
         Put the results in session["search_results"].

         ⚠️ THIS IS THE BRANCH. If nothing came back:
              - put a message in session["error"] saying what the user could
                change — "No results" is not that message
              - return the session
              - do NOT call suggest_outfit with nothing

      5. Choose an item — the first result is fine. Put it in
         session["selected_item"].

      6. Call suggest_outfit() with the selected item and the wardrobe.
         Put the result in session["outfit_suggestion"].

      7. Call create_fit_card() with the outfit and the item.
         Put the result in session["fit_card"].

      8. Return the session.

    ─────────────────────────────────────────────────────────────────────────
    IN UNIT 4 you come back and add two things:

      • Trace calls. One per step. `trace.step("search_listings", inputs=...,
        returned=...)` — see trace.py. Your README needs the output.

      • A handler for ModelUnavailable, so a bad key produces a message rather
        than a stack trace. The import is already at the top of this file.
    """
    session = new_session(query, wardrobe)
    count = 0

    # STYLE MEMORY (stretch): no wardrobe given, so use the one saved last time.
    if not _has_items(session["wardrobe"]):
        saved = _load_saved_wardrobe()
        if saved:
            session["wardrobe"] = saved
            session["wardrobe_source"] = "saved"
            print(f"[memory] no wardrobe given, loaded {len(saved['items'])} saved items")

    # Each time round the loop looks at the session and picks the next step
    # from what is already there (or not there yet).
    while True:
        count += 1
        trace.check_iterations(count)

        # Step 1: parse the query.
        if not session["parsed"]:
            session["parsed"] = parse_query(query)

            # SECOND BRANCH (stretch): nothing to search for.
            if not session["parsed"]["description"]:
                session["error"] = (
                    "I couldn't tell what you want to find. Describe the item "
                    "(for example: 'vintage graphic tee, size M, under $30')."
                )
                return session
            continue

        # Step 2: search.
        if not session["searched"]:
            parsed = session["parsed"]
            session["search_results"] = search_listings(
                parsed["description"], parsed["size"], parsed["max_price"]
            )
            session["searched"] = True

            # THE BRANCH: nothing came back, so stop. Do not call the next tool.
            if not session["search_results"]:
                session["error"] = _no_results_message(parsed)
                return session
            continue

        # Step 3: choose the first result.
        if session["selected_item"] is None:
            session["selected_item"] = session["search_results"][0]
            continue

        # Step 3b (stretch): compare the chosen item's price with the other results.
        if session["price_comparison"] is None:
            session["price_comparison"] = compare_prices(
                session["selected_item"], session["search_results"]
            )
            cmp_ = session["price_comparison"]
            print(
                f"[price] ${cmp_['price']:.0f} vs average "
                f"{'n/a' if cmp_['average'] is None else '$%.2f' % cmp_['average']}"
                f" of the other results: {cmp_['verdict']}"
            )
            continue

        # Step 4: outfit, using the item read back out of the session.
        if session["outfit_suggestion"] is None:
            item = session["selected_item"]
            session["outfit_input_id"] = item.get("id")
            print(
                f"[state] session selected_item id={item.get('id')}  |  "
                f"suggest_outfit received id={session['outfit_input_id']}"
            )
            session["outfit_suggestion"] = suggest_outfit(item, session["wardrobe"])
            continue

        # Step 5: fit card, using the outfit and item read back from the session.
        if session["fit_card"] is None:
            session["fit_card"] = create_fit_card(
                session["outfit_suggestion"], session["selected_item"]
            )
            continue

        # Step 6: everything is filled in. Remember the wardrobe for next time.
        if session["wardrobe_source"] == "given" and _has_items(session["wardrobe"]):
            _save_wardrobe(session["wardrobe"])
            print(f"[memory] saved {len(session['wardrobe']['items'])} wardrobe items")
        return session


def _no_results_message(parsed: dict) -> str:
    """Tell the user what they could change, using what they actually asked."""
    asked = f"'{parsed['description']}'"
    if parsed["size"]:
        asked += f", size {parsed['size']}"
    if parsed["max_price"] is not None:
        asked += f", under ${parsed['max_price']:.0f}"

    ideas = ["use fewer or different keywords (for example 'tee' instead of 'graphic tee')"]
    if parsed["size"]:
        ideas.append("try a different size or leave the size out")
    if parsed["max_price"] is not None:
        ideas.append("raise the price limit")
    return f"No listings matched {asked}. You could " + ", or ".join(ideas) + "."


# ── running it directly ───────────────────────────────────────────────────────

def _show(session: dict) -> None:
    if session["error"]:
        print(f"  stopped: {session['error']}")
        print(f"  fit_card is {session['fit_card']!r} — it should still be None here")
        return

    item = session["selected_item"] or {}
    print(f"  found:    {item.get('title')} — ${item.get('price')} on {item.get('platform')}")
    print(f"  outfit:   {session['outfit_suggestion']}")
    print(f"  fit card: {session['fit_card']}")


if __name__ == "__main__":
    from utils.data_loader import get_example_wardrobe

    print("=== A query the data can match ===")
    _show(run_agent(
        query="looking for a vintage graphic tee under $30",
        wardrobe=get_example_wardrobe(),
    ))

    print("\n=== A query it can't ===")
    _show(run_agent(
        query="designer ballgown size XXS under $5",
        wardrobe=get_example_wardrobe(),
    ))

    print(
        "\nThe second one should stop before the fit card. If both paths look "
        "the same,\nthe branch isn't doing anything yet."
    )
