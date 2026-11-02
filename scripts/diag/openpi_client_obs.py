"""openpi LIBERO client-side check (runs in examples/libero/.venv, Python 3.8). Read-only diagnostics.

    python openpi_client_obs.py OUT.npz [--port P]

1) Builds the first real observation exactly like examples/libero/main.py (libero_spatial task 0, init state 0,
   10 dummy steps, 180-degree image rotation, resize_with_pad 224, state = eef_pos + axis-angle + gripper_qpos),
   prints its shapes/ranges/finiteness and saves it to OUT.npz.
2) With --port: sends it to a running policy server through openpi_client (the same path the eval uses) and prints
   the returned action chunk's stats, then steps the env with the first 5 actions and reports ctrl finiteness.
"""
import math
import sys

import numpy as np


def quat2axisangle(quat):  # same as examples/libero/main.py
    quat = np.array(quat, dtype=np.float64)
    quat[3] = min(1.0, max(-1.0, quat[3]))
    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


def stats(name, a):
    a = np.asarray(a, dtype=np.float64)
    fin = bool(np.all(np.isfinite(a)))
    print("%s shape=%s finite=%d min=%.4g max=%.4g absmax=%.4g" % (
        name, a.shape, fin, np.nanmin(a) if a.size else 0, np.nanmax(a) if a.size else 0,
        np.nanmax(np.abs(a)) if a.size else 0))
    return fin


def main(argv):
    out = argv[1]
    port = int(argv[argv.index("--port") + 1]) if "--port" in argv else None
    import pathlib
    import openpi_client
    from openpi_client import image_tools
    from libero.libero import benchmark, get_libero_path
    from libero.libero.envs import OffScreenRenderEnv
    import libero.libero as ll
    print("CLIENT python=%s numpy=%s openpi_client=%s libero=%s" % (
        sys.version.split()[0], np.__version__, getattr(openpi_client, "__version__", "?"), ll.__file__))
    suite = benchmark.get_benchmark_dict()["libero_spatial"]()
    task = suite.get_task(0)
    bddl = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    print("CLIENT bddl=%s" % bddl)
    env = OffScreenRenderEnv(bddl_file_name=str(bddl), camera_heights=256, camera_widths=256)
    env.seed(7)
    env.reset()
    obs = env.set_init_state(suite.get_task_init_states(0)[0])
    for _ in range(10):
        obs, _, _, _ = env.step([0.0] * 6 + [-1.0])
    img = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["agentview_image"][::-1, ::-1]), 224, 224))
    wri = image_tools.convert_to_uint8(image_tools.resize_with_pad(np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1]), 224, 224))
    state = np.concatenate((obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"]))
    ok = stats("CLIENT obs/state", state)
    print("CLIENT obs/state values=%s" % np.array2string(state, precision=4))
    print("CLIENT obs/image shape=%s dtype=%s mean=%.1f std=%.1f" % (img.shape, img.dtype, img.mean(), img.std()))
    print("CLIENT obs/wrist shape=%s dtype=%s mean=%.1f std=%.1f" % (wri.shape, wri.dtype, wri.mean(), wri.std()))
    print("OBS_FINITE=%d" % int(ok and state.shape == (8,)))
    np.savez(out, image=img, wrist_image=wri, state=state, prompt=np.array(task.language))
    print("CLIENT prompt=%r saved=%s" % (task.language, out))
    if port is None:
        return 0
    from openpi_client import websocket_client_policy
    client = websocket_client_policy.WebsocketClientPolicy("localhost", port)
    print("CLIENT server metadata=%s" % (client.get_server_metadata(),))
    el = {"observation/image": img, "observation/wrist_image": wri, "observation/state": state, "prompt": str(task.language)}
    act = np.asarray(client.infer(el)["actions"])
    fin = stats("ROUNDTRIP actions", act)
    print("ROUNDTRIP first action=%s" % np.array2string(act[0], precision=4))
    print("ROUNDTRIP_ACTIONS_FINITE=%d" % int(fin))
    for a in act[:5]:
        env.step(a.tolist())
    ctrl = np.asarray(env.env.sim.data.ctrl)
    print("ENV_CTRL_FINITE=%d ctrl_absmax=%.4g" % (int(np.all(np.isfinite(ctrl))), np.nanmax(np.abs(ctrl))))
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
