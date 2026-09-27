import os
import numpy as np

from libero.libero import benchmark
from libero.libero.envs import OffScreenRenderEnv
from libero.libero import get_libero_path


# -----------------------------
# Config
# -----------------------------
TASK_SUITE = "libero_10"
TASK_ID = 0
INIT_STATE_ID = 0

# 小さい値で試す
TEST_MAGNITUDE = 0.2


# -----------------------------
# Build LIBERO environment
# -----------------------------
benchmark_dict = benchmark.get_benchmark_dict()
task_suite = benchmark_dict[TASK_SUITE]()

task = task_suite.get_task(TASK_ID)

bddl_file = os.path.join(
    get_libero_path("bddl_files"),
    task.problem_folder,
    task.bddl_file,
)

print("Task:", task.name)
print("Instruction:", task.language)
print("BDDL:", bddl_file)

env = OffScreenRenderEnv(
    bddl_file_name=bddl_file,
    camera_heights=128,
    camera_widths=128,
)

env.seed(0)

obs = env.reset()

init_states = task_suite.get_task_init_states(TASK_ID)
obs = env.set_init_state(init_states[INIT_STATE_ID])


# 少し settle
for _ in range(10):
    obs, reward, done, info = env.step(
        np.zeros(7, dtype=np.float32)
    )


def run_axis_test(name, action):
    global obs

    before = obs["robot0_eef_pos"].copy()

    print()
    print("=" * 60)
    print("TEST:", name)
    print("action:", action)
    print("before:", before)

    obs, reward, done, info = env.step(
        np.asarray(action, dtype=np.float32)
    )

    after = obs["robot0_eef_pos"].copy()
    delta = after - before

    print("after :", after)
    print("delta :", delta)

    return delta


# -----------------------------
# +X
# -----------------------------
run_axis_test(
    "+X",
    [
        TEST_MAGNITUDE, 0.0, 0.0,
        0.0, 0.0, 0.0,
        -1.0,  # gripper open
    ],
)

env.close()