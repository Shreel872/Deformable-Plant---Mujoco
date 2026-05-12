import numpy as np
import xml.etree.ElementTree as ET


class XMLModifier:
    def __init__(self):
        self.root = ET.Element("mujoco", model="deformable_plant")

        option = ET.SubElement(self.root, "option")
        option.set("timestep", "0.0005")
        option.set("integrator", "implicitfast")
        option.set("iterations", "100")
        option.set("ls_iterations", "20")
        option.set("tolerance", "1e-9")
        option.set("gravity", "0 0 -9.81")

        default = ET.SubElement(self.root, "default")
        ET.SubElement(default, "joint",
                      armature="0.05", damping="1.0",
                      limited="false", frictionloss="0.0")

        asset = ET.SubElement(self.root, "asset")
        ET.SubElement(asset, "material", name="stem_base",  rgba="0.22 0.38 0.18 1")
        ET.SubElement(asset, "material", name="stem_mid",   rgba="0.26 0.48 0.20 1")
        ET.SubElement(asset, "material", name="stem_top",   rgba="0.30 0.58 0.22 1")
        ET.SubElement(asset, "material", name="branch",     rgba="0.28 0.52 0.20 1")
        ET.SubElement(asset, "material", name="leaf_dark",  rgba="0.13 0.55 0.18 0.95")
        ET.SubElement(asset, "material", name="leaf_light", rgba="0.22 0.72 0.24 0.90")
        ET.SubElement(asset, "material", name="leaf_yellow",rgba="0.45 0.68 0.15 0.90")
        ET.SubElement(asset, "material", name="pot_body",   rgba="0.52 0.30 0.18 1")
        ET.SubElement(asset, "material", name="pot_rim",    rgba="0.60 0.36 0.20 1")
        ET.SubElement(asset, "material", name="soil",       rgba="0.22 0.14 0.08 1")
        ET.SubElement(asset, "material", name="probe_mat",  rgba="0.90 0.15 0.15 1")

        self.worldbody = ET.SubElement(self.root, "worldbody")

        # Disable shadows in XML — no API version issues
        visual = ET.SubElement(self.root, "visual")
        ET.SubElement(visual, "quality", shadowsize="0")
        ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5",
                      diffuse="0.6 0.6 0.6", specular="0.05 0.05 0.05")

        ET.SubElement(self.worldbody, "light", pos="0.3 -0.5 1.5",
                      dir="-0.3 0.5 -1", diffuse="0.7 0.7 0.65", specular="0.1 0.1 0.1")
        ET.SubElement(self.worldbody, "light", pos="-0.5 0.3 1.0",
                      dir="0.5 -0.3 -1", diffuse="0.3 0.3 0.35")
        ET.SubElement(self.worldbody, "geom", type="plane", size="5 5 0.1",
                      rgba="0.78 0.72 0.62 1", contype="1", conaffinity="1")

        self.contact  = ET.SubElement(self.root, "contact")
        self.equality = ET.SubElement(self.root, "equality")

        self._create_pot()
        self._create_main_stem(segments=12, height=0.38, base_radius=0.012, taper=0.88)
        self._create_probe()

    def _pg(self):
        return dict(contype="1", conaffinity="1", condim="3",
                    friction="0.7 0.3 0.3", solimp="0.9 0.95 0.001", solref="0.01 1")

    def _create_pot(self):
        ET.SubElement(self.worldbody, "geom", type="cylinder",
                      pos="0 0 0.025", size="0.072 0.025",
                      material="pot_body", contype="0", conaffinity="0")
        ET.SubElement(self.worldbody, "geom", type="cylinder",
                      pos="0 0 0.048", size="0.060 0.004",
                      material="pot_body", contype="0", conaffinity="0")
        ET.SubElement(self.worldbody, "geom", type="cylinder",
                      pos="0 0 0.053", size="0.065 0.003",
                      material="pot_rim", contype="0", conaffinity="0")
        ET.SubElement(self.worldbody, "geom", type="cylinder",
                      pos="0 0 0.056", size="0.055 0.004",
                      material="soil", contype="0", conaffinity="0")
        for angle in np.linspace(0, 2*np.pi, 7, endpoint=False):
            r = 0.03 + np.random.uniform(-0.008, 0.008)
            sx = r * np.cos(angle); sy = r * np.sin(angle)
            ET.SubElement(self.worldbody, "geom", type="sphere",
                          pos=f"{sx:.3f} {sy:.3f} 0.062", size="0.006",
                          rgba="0.28 0.20 0.12 1", contype="0", conaffinity="0")

        self.base = ET.SubElement(self.worldbody, "body", name="base", pos="0 0 0.063")
        ET.SubElement(self.base, "geom", type="sphere", size="0.006",
                      material="stem_base", contype="0", conaffinity="0")

    def _create_main_stem(self, segments=7, height=0.30,
                          base_radius=0.013, taper=0.84):
        seg_h   = height / segments
        current = self.base

        # Trunk (1-4): very stiff — must support full canopy weight
        # Mid (5-8): moderate — visible bending under ball contact
        # Tip (9-12): soft — whips freely
        stiffness = [40.0, 30.0, 22.0, 16.0, 10.0, 6.0, 3.5, 2.0, 1.0, 0.5, 0.2, 0.08]
        damping   = [12.0, 10.0,  8.0,  6.5,  5.0, 4.0, 3.0, 2.5, 2.0, 1.5, 1.0, 0.7]
        armature  = [0.25, 0.22,  0.18, 0.15, 0.12, 0.10, 0.08, 0.06, 0.05, 0.04, 0.03, 0.02]
        colours   = ["stem_base"] * 3 + ["stem_mid"] * 5 + ["stem_top"] * 4

        for i in range(1, segments + 1):
            pz   = 0.020 if i == 1 else seg_h
            body = ET.SubElement(current, "body",
                                 name=f"stem{i}", pos=f"0 0 {pz:.4f}")
            ET.SubElement(body, "joint", type="ball",
                          stiffness=str(stiffness[i-1]),
                          damping=str(damping[i-1]),
                          armature=str(armature[i-1]))
            r   = base_radius * (taper ** (i - 1))
            ET.SubElement(body, "geom", type="capsule",
                          fromto=f"0 0 0  0 0 {seg_h:.4f}",
                          size=f"{r:.5f}", material=colours[i-1], **self._pg())
            current = body

        self.main_stem  = current
        self.n_segments = segments

    def _create_probe(self):
        probe = ET.SubElement(self.worldbody, "body",
                              name="probe", pos="0.22 0.0 0.20")
        ET.SubElement(probe, "freejoint", name="probe_free")
        ET.SubElement(probe, "geom", type="sphere", size="0.025",
                      material="probe_mat", name="probe_geom",
                      contype="1", conaffinity="1", condim="3",
                      friction="0.7 0.3 0.3",
                      solimp="0.9 0.95 0.001", solref="0.01 1",
                      mass="2.0")

    def add_branch(self, name, parent, pos, direction,
                   length=0.05, radius=0.003, stiffness=40, damping=2):
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        pos = np.array(pos, dtype=float)

        branch = ET.Element("body", name=name,
                            pos=" ".join(f"{v:.5f}" for v in pos))
        ET.SubElement(branch, "joint", type="ball",
                      stiffness="1.5", damping="2.0", armature="0.04")

        # 3-segment branch: root -> mid -> tip, each softer
        seg_len = length / 3.0

        # Segment 1 — root
        end1 = direction * seg_len
        ET.SubElement(branch, "geom", type="capsule",
                      fromto="0 0 0  " + " ".join(f"{v:.5f}" for v in end1),
                      size=f"{max(radius, 0.003):.5f}",
                      material="branch", **self._pg())

        # ===================== CHANGED (leaves not only at tip) =====================
        # Add some leaves along segment 1, attached near random point on that segment
        # Comment these out if you want tip-only leaves again.
        attach1 = direction * np.random.uniform(0.0, seg_len)
        self._add_leaf_cluster(branch, attach1, direction, n=np.random.randint(1, 4))
        # ===================== END CHANGED =====================

        seg2 = ET.SubElement(branch, "body", name=f"{name}_s2",
                             pos=" ".join(f"{v:.5f}" for v in end1))
        ET.SubElement(seg2, "joint", type="ball",
                      stiffness="0.8", damping="1.4", armature="0.03")
        end2 = direction * seg_len
        ET.SubElement(seg2, "geom", type="capsule",
                      fromto="0 0 0  " + " ".join(f"{v:.5f}" for v in end2),
                      size=f"{max(radius*0.85, 0.0025):.5f}",
                      material="branch", **self._pg())

        # ===================== CHANGED (leaves not only at tip) =====================
        # Add some leaves along segment 2 as well
        # Comment these out if you want tip-only leaves again.
        attach2 = direction * np.random.uniform(0.0, seg_len)
        self._add_leaf_cluster(seg2, attach2, direction, n=np.random.randint(1, 4))
        # ===================== END CHANGED =====================

        # Segment 3 — tip
        seg3 = ET.SubElement(seg2, "body", name=f"{name}_s3",
                             pos=" ".join(f"{v:.5f}" for v in end2))
        ET.SubElement(seg3, "joint", type="ball",
                      stiffness="0.35", damping="0.9", armature="0.02")
        end3 = direction * seg_len
        ET.SubElement(seg3, "geom", type="capsule",
                      fromto="0 0 0  " + " ".join(f"{v:.5f}" for v in end3),
                      size=f"{max(radius*0.7, 0.002):.5f}",
                      material="branch", **self._pg())

        # ===================== CHANGED (leaf attach not always at very end) =====================
        # Previously: attach=end3 (so everything clumped at tip).
        # Now: attach somewhere along segment 3 (still "tip region", but not always at the end).
        # Comment these 2 lines and restore attach=end3 if you want the old behaviour.
        attach3 = direction * np.random.uniform(0.0, seg_len)
        self._add_leaf_cluster(seg3, attach3, direction, n=np.random.randint(2, 8))
        # ===================== END CHANGED =====================

       parent_body = self.find_body(parent)
        if parent_body is not None:
            parent_body.append(branch)

    def _add_leaf_cluster(self, body, attach, direction, n=4):
        direction = np.array(direction, dtype=float)
        up   = np.array([0.0, 0.0, 1.0])
        perp = np.cross(direction, up)
        if np.linalg.norm(perp) < 1e-6:
            perp = np.array([1.0, 0.0, 0.0])
        perp /= np.linalg.norm(perp)

        leaf_mats = ["leaf_dark", "leaf_light", "leaf_yellow"]

        for i in range(n):
            # Spread leaves around the branch + vary elevation
            angle   = (i / n) * 2 * np.pi + np.random.uniform(-0.2, 0.2)
            cos_a, sin_a = np.cos(angle), np.sin(angle)
            right = perp * cos_a + np.cross(direction, perp) * sin_a
            right /= np.linalg.norm(right)

            # Mix upward tilt into spread so leaves fan out naturally
            fan_dir = right * 0.7 + up * 0.3
            fan_dir /= np.linalg.norm(fan_dir)

            spread = 0.013 + np.random.uniform(-0.003, 0.005)

            # If leaves still look too clustered, uncomment this line and comment out the next one.
            # lpos   = attach + fan_dir * spread + direction * np.random.uniform(-0.01, 0.02)
            lpos   = attach + fan_dir * spread + direction * np.random.uniform(0, 0.008)

            leaf = ET.SubElement(body, "body",
                                 name=f"leaf_{body.attrib['name']}_{i}",
                                 pos=" ".join(f"{v:.5f}" for v in lpos))
            ET.SubElement(leaf, "joint", type="ball",
                          stiffness="0.5", damping="0.8", armature="0.015")

            mat  = leaf_mats[i % len(leaf_mats)]
            quat = self._dir_to_quat(fan_dir)
            w = 0.021 + np.random.uniform(-0.004, 0.005)
            h = 0.011 + np.random.uniform(-0.002, 0.003)
            ET.SubElement(leaf, "geom", type="ellipsoid",
                          size=f"{w:.4f} {h:.4f} 0.0013",
                          quat=quat, material=mat, **self._pg())

    def _dir_to_quat(self, d):
        d = d / np.linalg.norm(d)
        z  = np.array([0.0, 0.0, 1.0])
        ax = np.cross(z, d)
        if np.linalg.norm(ax) < 1e-6:
            return "1 0 0 0"
        ax  /= np.linalg.norm(ax)
        ang  = np.arccos(np.clip(np.dot(z, d), -1.0, 1.0))
        w    = np.cos(ang / 2)
        xyz  = ax * np.sin(ang / 2)
        return f"{w:.6f} {xyz[0]:.6f} {xyz[1]:.6f} {xyz[2]:.6f}"

    def find_body(self, body_name):
        def _search(elem):
            if elem.tag == "body" and elem.get("name") == body_name:
                return elem
            for child in elem:
                r = _search(child)
                if r is not None:
                    return r
        return _search(self.root)

    def save(self, filename="deformable_plant.xml"):
        ET.SubElement(self.equality, "weld", body1="world", body2="base",
                      solimp="0.999 0.9999 0.00001", solref="0.001 1")
        tree = ET.ElementTree(self.root)
        try:
            ET.indent(tree, space="  ")
        except AttributeError:
            pass
        tree.write(filename, encoding="utf-8", xml_declaration=True)
        print(f"Saved -> {filename}")