"""
main.py - Deformable Plant with In-Scene Force Visualisation

Two windows:
  1. MuJoCo viewer  - 3D sim with live force arrow + label at contact point
  2. Pygame window  - two separate real-time strip charts:
                       top panel    -> Contact Force (N) vs time
                       bottom panel -> Stem Tip Displacement (m) vs time

CONTROLS (global via pynput):
  Arrow UP / DOWN     -> move ball toward / away from plant
  Arrow LEFT / RIGHT  -> move ball left / right
  [ / ]               -> move ball up / down
  backslash           -> reset ball to start position

Requires: pip install pynput pygame
"""

from lsystem_generator import LSystemPlant
from xml_modifier import XMLModifier
import mujoco
import mujoco.viewer
import numpy as np
import time
import threading
import pygame

BALL_SPEED = 1
HISTORY    = 1500

# Pygame colours
BG    = (12,  12,  20)
WHITE = (220, 220, 220)
GREY  = (60,  60,  80)
DGREY = (28,  28,  42)
RED   = (220, 60,  60)
BLUE  = (80,  160, 230)
GOLD  = (220, 180, 60)

shared = {
    "force":   [],
    "disp":    [],
    "lock":    threading.Lock(),
    "running": True,
}

keys_held = set()


# ── Keyboard ──────────────────────────────────────────────────────────────────
def _start_keyboard():
    try:
        from pynput import keyboard as kb
        def on_press(k):   keys_held.add(k)
        def on_release(k): keys_held.discard(k)
        l = kb.Listener(on_press=on_press, on_release=on_release)
        l.daemon = True
        l.start()
        return True, kb
    except Exception as e:
        print(f"pynput unavailable: {e}")
        return False, None


def get_all_contacts(model, data):

    gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "probe_geom")
    contacts = []
    for i in range(data.ncon):
        c = data.contact[i]
        if c.geom1 == gid or c.geom2 == gid:
            cf = np.zeros(6, dtype=np.float64)
            mujoco.mj_contactForce(model, data, i, cf)
            f_mag = abs(cf[0])
            if f_mag > 0.001:
                pos    = np.array(c.pos)
                normal = np.array(c.frame[:3])
                if c.geom2 == gid:
                    normal = -normal
                contacts.append((pos, normal, f_mag))
    return contacts


def probe_total_force(model, data):
    return sum(f for _, _, f in get_all_contacts(model, data))


def stem_tip_pos(model, data):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "stem12")
    return data.xpos[bid].copy()


def get_probe_indices(model):
    probe_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "probe")
    for jid in range(model.njnt):
        if (model.jnt_bodyid[jid] == probe_bid and
                model.jnt_type[jid] == mujoco.mjtJoint.mjJNT_FREE):
            return model.jnt_qposadr[jid], model.jnt_dofadr[jid]
    raise RuntimeError("No freejoint on probe body")


def reset_probe(data, qaddr, daddr, pos):
    for i, v in enumerate(pos):
        data.qpos[qaddr+i] = v
    data.qpos[qaddr+3] = 1.0
    data.qpos[qaddr+4] = 0.0
    data.qpos[qaddr+5] = 0.0
    data.qpos[qaddr+6] = 0.0
    data.qvel[daddr:daddr+6] = 0.0


def draw_force_overlay(viewer, contacts, total_force):
    with viewer.lock():
        viewer.user_scn.ngeom = 0

        for pos, normal, f_mag in contacts:
            arrow_len = min(f_mag * 0.005, 0.15)
            if arrow_len < 0.005:
                continue

            tip = pos + normal * arrow_len

            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                g = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g,
                    mujoco.mjtGeom.mjGEOM_ARROW,
                    np.zeros(3), np.zeros(3), np.zeros(9),
                    np.array([1.0, 0.15, 0.15, 0.95], dtype=np.float32)
                )
                mujoco.mjv_makeConnector(
                    g, mujoco.mjtGeom.mjGEOM_ARROW, 0.007,
                    pos[0], pos[1], pos[2],
                    tip[0], tip[1], tip[2]
                )
                g.label = f"{f_mag:.2f}N"
                viewer.user_scn.ngeom += 1

            if viewer.user_scn.ngeom < viewer.user_scn.maxgeom:
                g2 = viewer.user_scn.geoms[viewer.user_scn.ngeom]
                mujoco.mjv_initGeom(
                    g2,
                    mujoco.mjtGeom.mjGEOM_SPHERE,
                    np.array([0.009, 0.009, 0.009]),
                    pos,
                    np.eye(3).flatten(),
                    np.array([1.0, 0.95, 0.1, 1.0], dtype=np.float32)
                )
                viewer.user_scn.ngeom += 1


def pygame_thread():
    pygame.init()
    W, H = 620, 520
    screen = pygame.display.set_mode((W, H))
    pygame.display.set_caption("Force & Displacement - Live")
    font   = pygame.font.SysFont("monospace", 13)
    font_b = pygame.font.SysFont("monospace", 14, bold=True)
    clock  = pygame.time.Clock()

    PAD_L  = 72   # left padding for y-axis labels
    PAD_R  = 20
    PAD_T  = 36   # top padding per panel (title)
    PAD_B  = 30   # bottom padding for controls hint
    GAP    = 26   # gap between the two panels

    panel_h = (H - PAD_T * 2 - PAD_B - GAP) // 2

    force_rect = pygame.Rect(PAD_L, PAD_T,
                             W - PAD_L - PAD_R, panel_h)
    disp_rect  = pygame.Rect(PAD_L, PAD_T + panel_h + GAP,
                             W - PAD_L - PAD_R, panel_h)

    def draw_panel(rect, values, color, title, ylabel, is_bottom=False):
        pygame.draw.rect(screen, DGREY, rect)
        pygame.draw.rect(screen, GREY,  rect, 1)

        # Title above the panel
        t_surf = font_b.render(title, True, WHITE)
        screen.blit(t_surf, (rect.left, rect.top - PAD_T + 4))

        if not values:
            msg = font.render("waiting for data...", True, GREY)
            screen.blit(msg, (rect.centerx - msg.get_width()//2,
                               rect.centery - msg.get_height()//2))
            return

        lo = min(values)
        hi = max(values)
        if hi - lo < 1e-9:
            hi = lo + max(abs(lo) * 0.1, 1e-3)

        # Horizontal grid lines + y-axis tick labels
        for frac in [0.0, 0.25, 0.5, 0.75, 1.0]:
            gy = rect.bottom - int(frac * rect.height)
            pygame.draw.line(screen, GREY, (rect.left, gy), (rect.right, gy), 1)
            val = lo + frac * (hi - lo)
            lbl = font.render(f"{val:.4f}", True, GREY)
            screen.blit(lbl, (rect.left - 60, gy - 7))

        # Vertical grid lines
        for frac in [0.25, 0.5, 0.75]:
            gx = rect.left + int(frac * rect.width)
            pygame.draw.line(screen, GREY, (gx, rect.top), (gx, rect.bottom), 1)

        # Build point list
        n   = len(values)
        pts = []
        for i, v in enumerate(values):
            x = rect.left + int(i / max(n - 1, 1) * rect.width)
            y = rect.bottom - int((v - lo) / (hi - lo) * rect.height)
            pts.append((x, max(rect.top, min(rect.bottom, y))))

        # Filled area under curve
        if len(pts) >= 2:
            fill = [pts[0]] + pts + [(pts[-1][0], rect.bottom), (pts[0][0], rect.bottom)]
            surf = pygame.Surface((rect.width, rect.height), pygame.SRCALPHA)
            local = [(p[0] - rect.left, p[1] - rect.top) for p in fill]
            pygame.draw.polygon(surf, (*color, 38), local)
            screen.blit(surf, (rect.left, rect.top))
            pygame.draw.lines(screen, color, False, pts, 2)

        # Current value dot
        if pts:
            pygame.draw.circle(screen, WHITE, pts[-1], 5)
            pygame.draw.circle(screen, color, pts[-1], 3)

        # Live readout (inside top-right corner)
        readout = font_b.render(f"now: {values[-1]:.4f}", True, GOLD)
        screen.blit(readout, (rect.right - readout.get_width() - 6,
                               rect.top + 4))

        # Y-axis label (rotated 90 deg)
        yl = pygame.transform.rotate(font.render(ylabel, True, GREY), 90)
        screen.blit(yl, (rect.left - 68, rect.centery - yl.get_height()//2))

        # X-axis label only on the bottom panel
        if is_bottom:
            xl = font.render("← older samples                 newer →", True, GREY)
            screen.blit(xl, (rect.centerx - xl.get_width()//2, rect.bottom + 4))

    while shared["running"]:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                shared["running"] = False

        screen.fill(BG)

        with shared["lock"]:
            force_vals = list(shared["force"][-HISTORY:])
            disp_vals  = list(shared["disp"][-HISTORY:])

        draw_panel(force_rect, force_vals, RED,
                   "Contact Force (N)", "Force (N)")
        draw_panel(disp_rect,  disp_vals,  BLUE,
                   "Stem Tip Displacement (m)", "Displ. (m)", is_bottom=True)

        # Controls hint
        hint = font.render(
            "Arrows=move ball   [/]=up/dn   backslash=reset", True, GREY)
        screen.blit(hint, (PAD_L, H - 18))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()


def main():
    print("Building plant ...")
    xml_mod  = XMLModifier()
    system   = LSystemPlant(iterations=2, angle=np.pi / 7)
    branches = system.generate_branches(max_branches=16,
                                         n_stem_segments=12,
                                         stem_height=0.38)
    print(f"   {len(branches)} branches.")
    for b in branches:
        xml_mod.add_branch(
            name=b["name"], parent=b["parent"],
            pos=b["pos"],   direction=b["dir"],
            length=b["length"], radius=b["radius"],
            stiffness=b["stiffness"], damping=b["damping"],
        )
    model_path = "deformable_plant.xml"
    xml_mod.save(model_path)

    model = mujoco.MjModel.from_xml_path(model_path)
    data  = mujoco.MjData(model)
    qaddr, daddr = get_probe_indices(model)

    START_POS = np.array([0.22, 0.0, 0.20])
    reset_probe(data, qaddr, daddr, START_POS)

    print("Settling — ramping gravity to prevent explosion ...")
    model.opt.gravity[2] = 0.0
    for _ in range(int(0.3 / model.opt.timestep)):
        data.qvel[daddr:daddr+6] = 0.0
        mujoco.mj_step(model, data)
    steps = int(1.5 / model.opt.timestep)
    for i in range(steps):
        model.opt.gravity[2] = -9.81 * (i / steps)
        data.qvel[daddr:daddr+6] = 0.0
        mujoco.mj_step(model, data)
    model.opt.gravity[2] = -9.81
    for _ in range(int(0.5 / model.opt.timestep)):
        data.qvel[daddr:daddr+6] = 0.0
        mujoco.mj_step(model, data)

    # rest position captured AFTER full gravity settling
    rest = stem_tip_pos(model, data)
    print(f"   Rest pos: {rest}")

    kb_active, kb = _start_keyboard()

    t = threading.Thread(target=pygame_thread, daemon=True)
    t.start()

    print("\nArrow keys move ball | [ / ] = up/down | backslash = reset")
    print("RED ARROW in viewer = contact force direction & magnitude\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        viewer.cam.azimuth   = 150
        viewer.cam.elevation = -20
        viewer.cam.distance  = 1.0
        viewer.cam.lookat[:] = [0.05, 0, 0.20]

        while viewer.is_running() and shared["running"]:
            t0 = time.time()

            vx, vy, vz = 0.0, 0.0, 0.0
            if kb_active:
                if kb.Key.up    in keys_held: vx = -BALL_SPEED
                if kb.Key.down  in keys_held: vx =  BALL_SPEED
                if kb.Key.left  in keys_held: vy =  BALL_SPEED
                if kb.Key.right in keys_held: vy = -BALL_SPEED
                for k in list(keys_held):
                    try:
                        if k.char == '[':    vz =  BALL_SPEED
                        elif k.char == ']':  vz = -BALL_SPEED
                        elif k.char == '\\':
                            reset_probe(data, qaddr, daddr, START_POS)
                            mujoco.mj_forward(model, data)
                    except AttributeError:
                        pass

            data.qvel[daddr+0] = vx
            data.qvel[daddr+1] = vy
            data.qvel[daddr+2] = vz
            data.qvel[daddr+3] = 0.0
            data.qvel[daddr+4] = 0.0
            data.qvel[daddr+5] = 0.0
            if vz == 0.0:
                data.qvel[daddr+2] = 0.0

            mujoco.mj_step(model, data)

            contacts    = get_all_contacts(model, data)
            total_force = sum(f for _, _, f in contacts)

            draw_force_overlay(viewer, contacts, total_force)

            # displacement = Euclidean distance of stem12 from its rest position
            tip  = stem_tip_pos(model, data)
            disp = float(np.linalg.norm(tip - rest))

            with shared["lock"]:
                shared["force"].append(total_force)
                shared["disp"].append(disp)
                if len(shared["force"]) > 8000:
                    shared["force"] = shared["force"][-8000:]
                    shared["disp"]  = shared["disp"][-8000:]

            viewer.sync()
            time.sleep(max(0.0, 1/60 - (time.time() - t0)))

    shared["running"] = False


if __name__ == "__main__":
    main()
    