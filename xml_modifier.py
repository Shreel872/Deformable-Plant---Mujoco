import numpy as np
import xml.etree.ElementTree as ET


class XMLModifier:
    def __init__(self):
        self.root = ET.Element("mujoco", model="deformable_plant")

        # solver tuned down — the plant has tons of joints, default settings tank fps
        option = ET.SubElement(self.root, "option")
        option.set("timestep", "0.001")
        option.set("integrator", "implicitfast")
        option.set("iterations", "30")
        option.set("ls_iterations", "8")
        option.set("tolerance", "1e-7")
        option.set("jacobian", "sparse")
        option.set("cone", "pyramidal")
        option.set("gravity", "0 0 -9.81")

        self._leaf_id = 0

        default = ET.SubElement(self.root, "default")
        ET.SubElement(default, "joint",
                      armature="0.05", damping="1.0",
                      limited="false", frictionloss="0.0")

        self.asset = ET.SubElement(self.root, "asset")
        asset = self.asset
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

        # one shared OBJ for all blades, scaled per-leaflet
        self._leaf_mesh_path = "leaf_mesh.obj"
        self._generate_leaf_mesh_file(self._leaf_mesh_path)
        self._registered_meshes = set()

        self.worldbody = ET.SubElement(self.root, "worldbody")

        # no shadows — faster + cleaner look
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

        # tip registry so stolon daughters attach at the tip, not the base
        self._element_tip = {}

        # main.py reads this to track displacement
        self.displacement_body = None

        self._create_pot()
        self._create_probe()

    def _generate_leaf_mesh_file(self, path):
        # scalloped disc, unit half-extent. one file, many scales.
        import os
        if os.path.exists(path):
            return
        N      = 36
        LOBES  = 12
        AMP    = 0.045
        verts  = [(0.0, 0.0, +1.0), (0.0, 0.0, -1.0)]
        for i in range(N):
            theta = 2.0 * np.pi * i / N
            r = 1.0 + AMP * np.cos(LOBES * theta)
            x, y = r * np.cos(theta), r * np.sin(theta)
            verts.append((x, y, +1.0))
            verts.append((x, y, -1.0))
        faces = []
        for i in range(N):
            rt   = 2 * i + 3
            rb   = 2 * i + 4
            rt_n = 2 * ((i + 1) % N) + 3
            rb_n = 2 * ((i + 1) % N) + 4
            faces.append((1,  rt, rt_n))
            faces.append((2,  rb_n, rb))
            faces.append((rt, rb, rb_n))
            faces.append((rt, rb_n, rt_n))
        with open(path, "w") as f:
            f.write("# strawberry leaflet blade\n")
            for v in verts:
                f.write(f"v {v[0]:.5f} {v[1]:.5f} {v[2]:.5f}\n")
            for face in faces:
                f.write(f"f {face[0]} {face[1]} {face[2]}\n")

    def _register_blade_mesh(self, name, scale_xyz):
        if name in self._registered_meshes:
            return
        self._registered_meshes.add(name)
        ET.SubElement(self.asset, "mesh",
                      name=name,
                      file=self._leaf_mesh_path,
                      scale=f"{scale_xyz[0]:.5f} "
                            f"{scale_xyz[1]:.5f} "
                            f"{scale_xyz[2]:.5f}")

    def _pg(self):
        return dict(contype="1", conaffinity="1", condim="3",
                    friction="0.7 0.3 0.3", solimp="0.9 0.95 0.001", solref="0.01 1")

    def _structural(self):
        # collide with probe (bit 8) but not with each other — kills the
        # crown-cluster spam without letting the ball phase through stems
        return dict(contype="8", conaffinity="0", condim="3",
                    friction="0.7 0.3 0.3",
                    solimp="0.9 0.95 0.001", solref="0.01 1")

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

    def _create_probe(self):
        probe = ET.SubElement(self.worldbody, "body",
                              name="probe", pos="0.22 0.0 0.20")
        ET.SubElement(probe, "freejoint", name="probe_free")
        # bits 1+4+8 = floor, cloth, plant. others have conaffinity=0
        # so they only ever collide with the probe.
        ET.SubElement(probe, "geom", type="sphere", size="0.025",
                      material="probe_mat", name="probe_geom",
                      contype="1", conaffinity="13", condim="3",
                      friction="0.7 0.3 0.3",
                      solimp="0.9 0.95 0.001", solref="0.01 1",
                      mass="2.0")
        # site at probe origin → torque taken about TCP, like a real ATI sensor
        ET.SubElement(probe, "site", name="probe_ft", pos="0 0 0",
                      size="0.003", rgba="0 0 0 0")

    def _resolve_parent(self, parent_name):
        if parent_name in self._element_tip:
            return self._element_tip[parent_name]
        return self.find_body(parent_name)

    def process_elements(self, elements):
        for el in elements:
            t = el["type"]
            if   t == "crown":   self.add_crown(el)
            elif t == "petiole": self.add_petiole(el)
            else:
                raise ValueError(f"Unknown element type: {t}")

    def add_crown(self, el):
        # rigid stub. no joint — mother is welded to world, daughters ride the stolon.
        parent = self._resolve_parent(el["parent"])
        if parent is None:
            raise RuntimeError(f"Crown {el['name']!r}: parent "
                               f"{el['parent']!r} not found")

        scale = el.get("scale", 1.0)
        # pos = parent tip offset + own offset (lets l-system scatter crowns in pot)
        tip_offset = self._element_tip.get(
            el["parent"] + "__tip_offset", np.zeros(3))
        tip_offset = np.asarray(tip_offset, dtype=float)
        own_pos    = np.asarray(el.get("pos", [0.0, 0.0, 0.0]),
                                dtype=float)
        pos = tip_offset + own_pos
        crown = ET.SubElement(parent, "body", name=el["name"],
                              pos=" ".join(f"{v:.5f}" for v in pos))

        h_stub = 0.012 * scale
        r_stub = 0.010 * scale
        ET.SubElement(crown, "geom", type="cylinder",
                      pos=f"0 0 {h_stub/2:.5f}",
                      size=f"{r_stub:.5f} {h_stub/2:.5f}",
                      material="stem_base", **self._structural())
        # remember stub height so children sit on top, not inside
        self._element_tip[el["name"] + "__top_z"] = h_stub

    def add_petiole(self, el):
        parent = self._resolve_parent(el["parent"])
        if parent is None:
            raise RuntimeError(f"Petiole {el['name']!r}: parent "
                               f"{el['parent']!r} not found")

        direction = np.array(el["direction"], dtype=float)
        direction /= np.linalg.norm(direction)
        length    = float(el["length"])
        n_seg     = int(el.get("segments", 3))
        seg_len   = length / n_seg
        leaf_scale = el.get("leaf_scale", 1.0)

        crown_top = self._element_tip.get(el["parent"] + "__top_z", 0.0)

        # stiff base, soft tip — outer third should visibly droop under the leaf
        STIFF_PROFILE = [110.0, 55.0, 24.0, 10.0, 4.0, 1.6, 0.9, 0.55]
        DAMP_PROFILE  = [  5.5,  3.2, 1.9, 1.1, 0.6, 0.35, 0.22, 0.16]
        ARM_PROFILE   = [0.22, 0.15, 0.10, 0.065, 0.040, 0.025, 0.018, 0.013]
        # resample so 3-seg and 8-seg petioles cover the same stiffness range
        idx = np.linspace(0, len(STIFF_PROFILE) - 1, n_seg)
        stiffness = [float(np.interp(i, np.arange(len(STIFF_PROFILE)),
                                     STIFF_PROFILE)) for i in idx]
        damping   = [float(np.interp(i, np.arange(len(DAMP_PROFILE)),
                                     DAMP_PROFILE))  for i in idx]
        armature  = [float(np.interp(i, np.arange(len(ARM_PROFILE)),
                                     ARM_PROFILE))   for i in idx]

        # taper 3mm -> 1.4mm
        r_base, r_tip = 0.0030, 0.0014
        radii = [r_base + (r_tip - r_base) * (s / max(n_seg - 1, 1))
                 for s in range(n_seg)]

        end = direction * seg_len

        # pre-bent rest pose so it droops at startup instead of standing upright
        horiz = np.array([direction[0], direction[1], 0.0])
        if np.linalg.norm(horiz) > 1e-6:
            horiz /= np.linalg.norm(horiz)
            pitch_axis = np.array([-horiz[1], horiz[0], 0.0])
        else:
            pitch_axis = np.array([0.0, 1.0, 0.0])
        bend_start   = max(2, n_seg // 2)
        bend_per_seg = np.radians(16.0)
        h            = bend_per_seg * 0.5
        bend_w       = np.cos(h)
        bend_xyz     = pitch_axis * np.sin(h)
        bend_quat    = (f"{bend_w:.6f} {bend_xyz[0]:.6f} "
                        f"{bend_xyz[1]:.6f} {bend_xyz[2]:.6f}")

        cur = ET.SubElement(parent, "body", name=el["name"],
                            pos=f"0 0 {crown_top:.5f}")
        ET.SubElement(cur, "joint", type="ball",
                      stiffness=f"{stiffness[0]:.4f}",
                      damping=f"{damping[0]:.4f}",
                      armature=f"{armature[0]:.4f}")
        ET.SubElement(cur, "geom", type="capsule",
                      fromto="0 0 0  " + " ".join(f"{v:.5f}" for v in end),
                      size=f"{radii[0]:.5f}",
                      material="branch", **self._structural())

        for s in range(1, n_seg):
            seg_attrs = {"name": f"{el['name']}_s{s}",
                         "pos":  " ".join(f"{v:.5f}" for v in end)}
            if s >= bend_start:
                seg_attrs["quat"] = bend_quat
            seg = ET.SubElement(cur, "body", **seg_attrs)
            ET.SubElement(seg, "joint", type="ball",
                          stiffness=f"{stiffness[s]:.4f}",
                          damping=f"{damping[s]:.4f}",
                          armature=f"{armature[s]:.4f}")
            ET.SubElement(seg, "geom", type="capsule",
                          fromto="0 0 0  " + " ".join(f"{v:.5f}" for v in end),
                          size=f"{radii[s]:.5f}",
                          material="branch", **self._structural())
            cur = seg

        self._add_leaf_cluster(cur, attach=end, direction=direction,
                               n=1, scale=leaf_scale)

        # first petiole built = displacement reference body
        if self.displacement_body is None:
            self.displacement_body = f"{el['name']}_s{n_seg-1}" if n_seg > 1 \
                                     else el["name"]

    def _add_leaf_cluster(self, body, attach, direction, n=1, scale=1.0):
        # trifoliate: 3 leaflets 120° apart. each leaflet is 4 chained ball-jointed
        # segments along the midrib so the leaf can curl locally instead of as a slab.
        direction = np.array(direction, dtype=float)
        direction /= np.linalg.norm(direction)
        up = np.array([0.0, 0.0, 1.0])

        def rot_around_up(d, ang):
            ca, sa = np.cos(ang), np.sin(ang)
            return np.array([d[0]*ca - d[1]*sa,
                             d[0]*sa + d[1]*ca,
                             d[2]])

        outward = np.array([direction[0], direction[1], 0.0])
        if np.linalg.norm(outward) < 1e-3:
            outward = np.array([1.0, 0.0, 0.0])
        outward /= np.linalg.norm(outward)

        SPREAD = np.radians(120)
        leaflet_dirs = [
            ("c", outward),
            ("l", rot_around_up(outward,  SPREAD)),
            ("r", rot_around_up(outward, -SPREAD)),
        ]

        self._leaf_id += 1
        cluster_id = self._leaf_id

        # leaflets sit flat, not canted to petiole tilt
        TILT_UP = np.radians(0.0)

        # circular leaflets, center+side at 100%/92%
        LEAF_LENGTH = 0.110 * scale
        LEAF_WIDTH  = 0.110 * scale
        SIDE_RATIO  = 0.92
        BLADE_THICK = 0.00065 * scale
        MIDRIB_R    = 0.00085 * scale

        # root = 2 hinges (pitch + roll, no yaw) so the leaf can't spin around the petiole axis
        ROOT_PITCH_STIFF, ROOT_PITCH_DAMP = 0.95, 0.40
        ROOT_ROLL_STIFF,  ROOT_ROLL_DAMP  = 0.65, 0.32
        ROOT_HINGE_ARM = 0.012
        SEG_STIFF = [None, 0.16, 0.080, 0.038]
        SEG_DAMP  = [None, 0.090, 0.055, 0.032]
        SEG_ARM   = [None, 0.0032, 0.0022, 0.0016]
        BLADE_MASS = 0.0075 * scale * scale

        mat_names = ["leaf_dark", "leaf_light", "leaf_yellow"]

        # blades overlap heavily along x so they read as one continuous ovate leaf
        SEG_BODY_X    = [0.00, 0.22, 0.50, 0.78]
        BLADE_LOCAL_X = [0.30, 0.25, 0.20, 0.10]
        BLADE_HALF_L  = [0.30, 0.28, 0.26, 0.13]
        BLADE_HALF_W  = [0.44, 0.50, 0.50, 0.20]  # follows a disc cross-section
        MIDRIB_LEN    = [0.27, 0.27, 0.27, 0.20]
        MIDRIB_R_MUL  = [1.00, 0.92, 0.82, 0.72]

        def _add_blade(parent, idx, mat, mesh_name, half_w_mul=1.0):
            ET.SubElement(parent, "geom", type="capsule",
                          fromto=f"0 0 0  {L*MIDRIB_LEN[idx]:.5f} 0 0",
                          size=f"{MIDRIB_R*MIDRIB_R_MUL[idx]:.5f}",
                          material="branch",
                          contype="0", conaffinity="0")
            self._register_blade_mesh(
                mesh_name,
                (L * BLADE_HALF_L[idx],
                 W * BLADE_HALF_W[idx] * half_w_mul,
                 BLADE_THICK))
            ET.SubElement(parent, "geom", type="mesh",
                          mesh=mesh_name,
                          pos=f"{L*BLADE_LOCAL_X[idx]:.5f} 0 0",
                          material=mat,
                          mass=f"{BLADE_MASS:.5f}",
                          contype="1", conaffinity="1", condim="3",
                          friction="0.7 0.3 0.3",
                          solimp="0.9 0.95 0.001", solref="0.01 1")

        for k, (tag, ldir) in enumerate(leaflet_dirs):
            ldir = ldir / np.linalg.norm(ldir)

            size_mul = 1.0 if tag == "c" else SIDE_RATIO
            L = LEAF_LENGTH * size_mul
            W = LEAF_WIDTH  * size_mul

            horiz = np.array([ldir[0], ldir[1], 0.0])
            if np.linalg.norm(horiz) < 1e-6:
                horiz = np.array([1.0, 0.0, 0.0])
            horiz /= np.linalg.norm(horiz)
            body_x = horiz * np.cos(TILT_UP) + up * np.sin(TILT_UP)
            body_x /= np.linalg.norm(body_x)
            body_z = up - np.dot(up, body_x) * body_x
            body_z /= np.linalg.norm(body_z)
            body_y = np.cross(body_z, body_x)

            leaf_name = f"leaflet_{body.attrib['name']}_{cluster_id}_{tag}"
            mat_name  = mat_names[k % 3]

            leaf_root = ET.SubElement(
                body, "body", name=leaf_name,
                pos=" ".join(f"{v:.5f}" for v in attach),
                quat=self._quat_from_axes(body_x, body_y, body_z))
            # pitch + roll, no yaw → leaf droops and curls but doesn't spin
            ET.SubElement(leaf_root, "joint", type="hinge",
                          axis="0 1 0",
                          stiffness=f"{ROOT_PITCH_STIFF:.4f}",
                          damping=f"{ROOT_PITCH_DAMP:.4f}",
                          armature=f"{ROOT_HINGE_ARM:.4f}")
            ET.SubElement(leaf_root, "joint", type="hinge",
                          axis="1 0 0",
                          stiffness=f"{ROOT_ROLL_STIFF:.4f}",
                          damping=f"{ROOT_ROLL_DAMP:.4f}",
                          armature=f"{ROOT_HINGE_ARM:.4f}")
            mesh_base = f"lf_{cluster_id}_{tag}"
            _add_blade(leaf_root, 0, mat_name, f"{mesh_base}_s0")

            # lateral veins only on the root seg — rest of the leaf is too narrow
            LAT_VEIN_R = MIDRIB_R * 0.42
            vein_specs = [
                (L*0.06, np.radians(58), W*0.40),
                (L*0.18, np.radians(50), W*0.42),
            ]
            for ax, ang, vl in vein_specs:
                for sign in (+1, -1):
                    ex = ax + vl * np.cos(ang)
                    ey = sign * vl * np.sin(ang)
                    ET.SubElement(leaf_root, "geom", type="capsule",
                                  fromto=f"{ax:.5f} 0 0  "
                                         f"{ex:.5f} {ey:.5f} 0",
                                  size=f"{LAT_VEIN_R:.5f}",
                                  material="branch",
                                  contype="0", conaffinity="0")

            cur_parent = leaf_root
            for s in range(1, 4):
                seg = ET.SubElement(cur_parent, "body",
                                    name=f"{leaf_name}_s{s}",
                                    pos=f"{L*SEG_BODY_X[s]-L*SEG_BODY_X[s-1]:.5f} 0 0")
                ET.SubElement(seg, "joint", type="ball",
                              stiffness=f"{SEG_STIFF[s]:.4f}",
                              damping=f"{SEG_DAMP[s]:.4f}",
                              armature=f"{SEG_ARM[s]:.4f}")
                _add_blade(seg, s, mat_name, f"{mesh_base}_s{s}")

                # veins on the mid segs too, skip the tip
                if s in (1, 2):
                    ax = L * 0.10
                    ang = np.radians(48)
                    vl = W * (0.38 if s == 1 else 0.30)
                    for sign in (+1, -1):
                        ex = ax + vl * np.cos(ang)
                        ey = sign * vl * np.sin(ang)
                        ET.SubElement(seg, "geom", type="capsule",
                                      fromto=f"{ax:.5f} 0 0  "
                                             f"{ex:.5f} {ey:.5f} 0",
                                      size=f"{LAT_VEIN_R*0.85:.5f}",
                                      material="branch",
                                      contype="0", conaffinity="0")

                cur_parent = seg

    def _quat_from_axes(self, x_axis, y_axis, z_axis):
        # standard shoemake rotation-matrix to quat
        R = np.column_stack([x_axis, y_axis, z_axis])
        tr = R[0, 0] + R[1, 1] + R[2, 2]
        if tr > 0.0:
            s = np.sqrt(tr + 1.0) * 2.0
            w = 0.25 * s
            x = (R[2, 1] - R[1, 2]) / s
            y = (R[0, 2] - R[2, 0]) / s
            z = (R[1, 0] - R[0, 1]) / s
        elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
            w = (R[2, 1] - R[1, 2]) / s
            x = 0.25 * s
            y = (R[0, 1] + R[1, 0]) / s
            z = (R[0, 2] + R[2, 0]) / s
        elif R[1, 1] > R[2, 2]:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
            w = (R[0, 2] - R[2, 0]) / s
            x = (R[0, 1] + R[1, 0]) / s
            y = 0.25 * s
            z = (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
            w = (R[1, 0] - R[0, 1]) / s
            x = (R[0, 2] + R[2, 0]) / s
            y = (R[1, 2] + R[2, 1]) / s
            z = 0.25 * s
        return f"{w:.6f} {x:.6f} {y:.6f} {z:.6f}"

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
        # weld base to world so the mother crown is rigidly grounded
        ET.SubElement(self.equality, "weld", body1="world", body2="base",
                      solimp="0.999 0.9999 0.00001", solref="0.001 1")
        tree = ET.ElementTree(self.root)
        try:
            ET.indent(tree, space="  ")
        except AttributeError:
            pass
        tree.write(filename, encoding="utf-8", xml_declaration=True)
        print(f"Saved -> {filename}")
