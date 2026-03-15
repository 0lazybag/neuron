import moderngl
import numpy as np
import time
from multiprocessing import shared_memory
import pynvml
import secrets
import sys
import os

# --- НАСТРОЙКИ ФАЙЛА ---
BRAIN_FILE = "memory"

# --- ПРИНУДИТЕЛЬНАЯ РАЗБЛОКИРОВКА ТЕРМИНАЛА ---
sys.stdout.reconfigure(line_buffering=True)
os.environ["PYTHONUNBUFFERED"] = "1"

# --- ИНИЦИАЛИЗАЦИЯ NVML ---
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

def get_gpu_clock():
    return pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)

# --- ГЕНЕРАЦИЯ ИМЕН ---
NAME_INPUT = secrets.token_hex(8)
NAME_RESPONSE = secrets.token_hex(8)
NAME_STORAGE = secrets.token_hex(8)

# --- ГРАФИКА ---
ctx = moderngl.create_standalone_context()

compute_shader_source = """
#version 430
layout(local_size_x = 8) in;
layout(std430, binding = 0) buffer InputBuffer { uint input_val; };
layout(std430, binding = 1) buffer StateBuffer { uint last_val; };
layout(std430, binding = 2) buffer OutputBuffer { uint output_val; };
layout(std430, binding = 3) buffer WeightsBuffer { float weights[8]; };

uniform float delta_freq;
uniform float learning_rate;

void main() {
    uint i = gl_LocalInvocationID.x;
    uint current_bit = (input_val >> i) & 1u;
    float activation = current_bit * weights[i];
    uint out_bit = activation > 0.5 ? 1u : 0u;

    if (out_bit == 1u) {
        atomicOr(output_val, (1u << i));
    } else {
        atomicAnd(output_val, ~(1u << i));
    }

    weights[i] += delta_freq * learning_rate * (float(current_bit) - activation);
    weights[i] = clamp(weights[i], 0.0, 1.0);
    if (i == 0) last_val = input_val;
}
"""

compute_shader = ctx.compute_shader(compute_shader_source)

# --- ЛОГИКА ЗАГРУЗКИ МОЗГА ---
if os.path.exists(BRAIN_FILE):
  #  print(f"// Обнаружен сохраненный мозг. Загрузка весов...", flush=True)
    initial_weights = np.fromfile(BRAIN_FILE, dtype='f4')
    # Если файл поврежден или пуст, создаем заново
    if initial_weights.size != 8:
        initial_weights = np.random.rand(8).astype('f4')
else:
  #  print("// Мозг не найден. Создание новых весов...", flush=True)
    initial_weights = np.random.rand(8).astype('f4')

in_vram = ctx.buffer(reserve=4)
state_vram = ctx.buffer(data=np.array([999], dtype='u4').tobytes())
out_vram = ctx.buffer(data=np.array([0], dtype='u4').tobytes())
weights_vram = ctx.buffer(data=initial_weights.tobytes())

try:
    shm_in = shared_memory.SharedMemory(name=NAME_INPUT, create=True, size=1)
    shm_st = shared_memory.SharedMemory(name=NAME_STORAGE, create=True, size=1)
    shm_res = shared_memory.SharedMemory(name=NAME_RESPONSE, create=True, size=1)

    print(f"{NAME_INPUT}", flush=True)
    print(f"{NAME_RESPONSE}", flush=True)
    
    with open("shm_keys.tmp", "w") as f:
        f.write(f"{NAME_INPUT}\n{NAME_RESPONSE}")

    last_val_cpu = shm_in.buf[0]
    last_clock = get_gpu_clock()

    while True:
        current_val_cpu = shm_in.buf[0]

        # Если получили 0 — сохраняем мозг
        if current_val_cpu == 0:
            current_weights = np.frombuffer(weights_vram.read(), dtype='f4')
            current_weights.tofile(BRAIN_FILE)
            # Чтобы не писать на диск 100 раз в секунду, пока висит 0:
            time.sleep(0.1) 

        if current_val_cpu != last_val_cpu:
            current_clock = get_gpu_clock()
            delta = float(current_clock - last_clock)

            in_vram.write(np.array([current_val_cpu], dtype='u4').tobytes())
            out_vram.write(np.array([0], dtype='u4').tobytes()) 

            in_vram.bind_to_storage_buffer(0)
            state_vram.bind_to_storage_buffer(1)
            out_vram.bind_to_storage_buffer(2)
            weights_vram.bind_to_storage_buffer(3)

            compute_shader['delta_freq'].value = delta
            compute_shader['learning_rate'].value = 0.005
            compute_shader.run(group_x=1)

            res_bits = np.frombuffer(out_vram.read(), dtype='u4')[0]

            shm_st.buf[0] = current_val_cpu
            shm_res.buf[0] = int(res_bits)

            last_val_cpu = current_val_cpu
            last_clock = current_clock

        time.sleep(0.01)

except Exception as e:
    print(f"// Ошибка: {e}", flush=True)
finally:
    pynvml.nvmlShutdown()
    if os.path.exists("shm_keys.tmp"):
        os.remove("shm_keys.tmp")
    for s in [shm_in, shm_st, shm_res]:
        try:
            s.close()
            s.unlink()
        except:
            pass