"""Who makes the repeating move? Runs wsp.exp6.play on Modal and returns per-game arrays."""
import json, os, sys, time
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wsp.exp6 import load_net, saved_net, NETS, zero_net, CPUS
from wsp.runner import read_dotenv

def remote(jobs, cpus):
    os.environ["XLA_FLAGS"] = f"--xla_force_host_platform_device_count={cpus}"
    import inspect, jax, numpy as np
    from wsp.exp6 import play
    import pgx
    from pgx._src.games import go as gomod
    src = inspect.getsource(gomod)
    i = src.find('is_psk')
    out = {"src": src, "devices": jax.device_count()}
    for name, (a, b, games, seed) in jobs.items():
        t0 = time.time()
        r = play(a, b, games, seed)
        out[name] = {k: np.asarray(v) for k, v in r.items()}
        out[name]["seconds"] = time.time() - t0
    return out

if __name__ == "__main__":
    profile = read_dotenv().get("MODAL_PROFILE"); assert profile
    os.environ["MODAL_PROFILE"] = profile
    import modal
    assert modal.config._profile == profile, modal.config._profile
    opus = saved_net('runs/claude-opus-5-5/exp6/budget_0/s1/work')
    astra = load_net(NETS / 'budget_0')
    ppo = {s: load_net(NETS / f'ppo_same_size_lr3e-5_s{s}') for s in range(3)}
    jobs = {}
    for s in range(3):
        jobs[f'opus_ppo{s}'] = (opus, ppo[s], 2000, s)        # same seeds as score.json
        jobs[f'astra_ppo{s}'] = (astra, ppo[s], 2000, s)
    jobs['opus_random'] = (opus, zero_net(), 2000, 3)
    jobs['opus_ppo1_seed101'] = (opus, ppo[1], 2000, 101)
    image = (modal.Image.debian_slim(python_version="3.12")
             .uv_pip_install("jax==0.4.38", "jaxlib==0.4.38", "pgx==2.6.0", "numpy==2.2.6")
             .add_local_python_source("wsp"))
    app = modal.App("wsp-exp6-review", image=image)
    fn = app.function(gpu=None, cpu=CPUS, timeout=1800, serialized=True)(remote)
    t0 = time.time()
    with modal.enable_output(), app.run():
        out = fn.remote(jobs, CPUS)
    print('wall', time.time() - t0, 'devices', out['devices'])
    import pickle
    pickle.dump(out, open(Path(__file__).with_name('exp6_rep.pkl'), 'wb'))
