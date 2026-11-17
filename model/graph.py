    def update(self, fid: str, params: dict | None = None, deps=None) -> Feature:
        """Change a feature's parameters. A field set here replaces the old expression, not just its
        value: otherwise `distance: "t"` would silently override the 5.0 the caller just passed."""
        cur = self.get(fid)
        given = dict(params or {})
        merged = {**cur.params, **given}
        exprs = {k: v for k, v in (cur.params.get("exprs") or {}).items() if k not in given}
        if exprs:
            merged["exprs"] = exprs
        else:
            merged.pop("exprs", None)
        new = Feature(cur.id, cur.op, tuple(deps) if deps is not None else cur.deps, merged)
        backup = self._features[fid]
        del self._features[fid]
        try:
            return self._insert(new)
        except GraphError:
            self._features[fid] = backup
            raise
