class LocalScoreCache:
    """
    Caches local scores for (node, parent set) pairs to avoid redundant computation.
    Usage:
        cache = LocalScoreCache()
        score = cache.get_local_score(node, parents, scorer)
    """

    def __init__(self):
        self.cache = {}
        self.hits = 0
        self.misses = 0

    def get_local_score(self, node, parents, scorer):
        key = (node, frozenset(parents))
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        self.misses += 1
        score = scorer.local_score(node, list(parents))
        self.cache[key] = score
        return score
