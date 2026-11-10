from models.schema import SearchQuery


def operator_queries(q: SearchQuery) -> list[str]:
    query = q.query.strip()

    act_include_set = [domain for domain in q.include_domains if domain not in q.exclude_domains]
    if q.include_domains and not act_include_set:
        #all includes are in excludes
        return []

    queries = []
    if 1 <= len(act_include_set) <= 3:
        for include in act_include_set:
            queries.append(f"{query} site:{include}")
    elif len(act_include_set) > 3:
        new_q = query + " ("
        for idx, include in enumerate(act_include_set):
            if idx < len(act_include_set) - 1:
                new_q += f"site:{include} OR "
            else:
                new_q += f"site:{include})"
        queries.append(new_q)
    else:
        queries.append(query)
    top_5_excludes = q.exclude_domains[:5]
    exclude_query = "".join(f" -site:{exclude}" for exclude in top_5_excludes)

    for idx, query in enumerate(queries):
        queries[idx] = query + exclude_query

    return queries









