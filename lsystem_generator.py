"""strawberry L-system: axiom C -> [L]*n, walked by turtle."""

import numpy as np


class StrawberryLSystem:
    def __init__(self,
                 iterations=1,
                 seed=42,
                 leaves_per_crown=8,
                 petiole_length=0.140,
                 petiole_segments=5,
                 leaf_scale=1.0):
        self.iterations       = iterations
        self.rng              = np.random.default_rng(seed)
        self.leaves_per_crown = leaves_per_crown
        self.petiole_length   = petiole_length
        self.petiole_segments = petiole_segments
        self.leaf_scale       = leaf_scale

        self.axiom = "C"
        # one [L] per slot, brackets scope each side branch
        self.rules = {
            "C": "[L]" * self.leaves_per_crown,
            "L": "L",
        }

    def expand(self):
        s = self.axiom
        for _ in range(self.iterations):
            s = "".join(self.rules.get(ch, ch) for ch in s)
        return s

    def generate(self, root_parent="base"):
        s = self.expand()
        elements = []
        self._uid = 0

        # always one crown — emit it up front, then walk for leaves
        crown_name = self._new_id("crown")
        elements.append({
            "type":   "crown",
            "name":   crown_name,
            "parent": root_parent,
            "pos":    [0.0, 0.0, 0.0],
            "scale":  1.0,
        })

        n_leaves_in_string = s.count("L")
        if n_leaves_in_string == 0:
            return elements

        ctx_stack  = [crown_name]
        leaf_index = 0

        for ch in s:
            if ch == "[":
                ctx_stack.append(ctx_stack[-1])
            elif ch == "]":
                ctx_stack.pop()
            elif ch == "L":
                parent_crown = ctx_stack[-1]
                # even spacing around the crown + tiny jitter so it's not too perfect
                azi = (leaf_index + 0.5) * 2.0 * np.pi / n_leaves_in_string
                azi += self.rng.uniform(-0.18, 0.18)
                # high elevation — petioles come off near-vertical
                elev = np.radians(self.rng.uniform(60.0, 80.0))
                d = self._spherical(azi, elev)
                length = (self.petiole_length
                          + self.rng.uniform(-0.014, 0.014))
                elements.append({
                    "type":       "petiole",
                    "name":       self._new_id("pet"),
                    "parent":     parent_crown,
                    "direction":  d.tolist(),
                    "length":     float(length),
                    "segments":   self.petiole_segments,
                    "leaf_scale": float(self.leaf_scale),
                })
                leaf_index += 1
            elif ch == "C":
                # rules never re-emit C, but leaving the branch for later
                pass

        return elements

    def _new_id(self, prefix):
        self._uid += 1
        return f"{prefix}{self._uid}"

    @staticmethod
    def _spherical(azimuth, elevation):
        ce = np.cos(elevation)
        return np.array([np.cos(azimuth) * ce,
                         np.sin(azimuth) * ce,
                         np.sin(elevation)])
