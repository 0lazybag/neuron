from multiprocessing import shared_memory
import time
import pynvml

# --- НАСТРОЙКИ NVML (Управление ГП) ---
pynvml.nvmlInit()
gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)

МИН_ЧАСТОТА = 210
МАКС_ЧАСТОТА = 3105
МНОЖИТЕЛЬ = 1.1  # Шаг изменения частоты

def изменить_частоту(увеличить=True):
    try:
        # Получаем текущую частоту
        текущая = pynvml.nvmlDeviceGetClockInfo(gpu_handle, pynvml.NVML_CLOCK_GRAPHICS)
        
        if увеличить:
            новая = int(текущая * МНОЖИТЕЛЬ)
            status = "УВЕЛИЧЕНИЕ"
        else:
            новая = int(текущая / МНОЖИТЕЛЬ)
            status = "СНИЖЕНИЕ"
        
        # Проверка границ
        новая = max(МИН_ЧАСТОТА, min(новая, МАКС_ЧАСТОТА))
        
        # Фиксация частоты (нужны права админа)
        pynvml.nvmlDeviceSetGpuLockedClocks(gpu_handle, новая, новая)
        print(f"[*] ГП: {текущая}MHz -> {новая}MHz ({status})")
        
    except pynvml.NVMLError as err:
        print(f"[!] Ошибка NVML: {err} (Запустите от Админа)")

# --- НАСТРОЙКА SHARED MEMORY ---
NAME_INPUT = "shm_input"
NAME_RESPONSE = "shm_response"

try:
    # Подключаемся к памяти сервера
    shm_in = shared_memory.SharedMemory(name=NAME_INPUT)
    shm_res = shared_memory.SharedMemory(name=NAME_RESPONSE)
    
    print("[*] КЛИЕНТ: Связь с ОЗУ и ГП установлена.")
    print(f"[*] Лимиты: {МИН_ЧАСТОТА}-{МАКС_ЧАСТОТА} MHz")

    while True:
        msg = input("\nВведите данные (0-255) или 'exit': ")
        if msg.lower() == 'exit': 
            break
            
        try:
            val_to_send = int(msg)
            if 0 <= val_to_send <= 255:
                # 1. Запоминаем старый ответ
                old_response = shm_res.buf[0]
                
                # 2. Пишем в память "вход"
                shm_in.buf[0] = val_to_send
                print(f"[>] Отправлено в ОЗУ: {val_to_send}. Ждем расчет...")
                
                # 3. Ждем, пока сервер обновит "ответ" (цикл мониторинга)
                timeout = 0
                while shm_res.buf[0] == old_response and timeout < 30:
                    time.sleep(0.05)
                    timeout += 1
                
                res_val = shm_res.buf[0]
                print(f"[<] Получен ответ из ОЗУ: {res_val}")

                # 4. ЛОГИКА УПРАВЛЕНИЯ ЧАСТОТОЙ
                # Если ответ шейдера совпал с тем, что мы ввели — повышаем частоту
                if res_val == val_to_send:
                    print("[+] СОВПАДЕНИЕ! Нейросеть в ОЗУ подтвердила данные.")
                    изменить_частоту(увеличить=True)
                else:
                    print("[-] ОШИБКА! Данные в адресах не совпали.")
                    изменить_частоту(увеличить=False)
                    
            else:
                print("[-] Введите число от 0 до 255")
        except ValueError:
            print("[-] Нужна цифра")

except FileNotFoundError:
    print("[!] Ошибка: Сервер (GPU_Server) не найден в ОЗУ!")
finally:
    # Сброс настроек ГП и закрытие памяти
    try:
        pynvml.nvmlDeviceResetGpuLockedClocks(gpu_handle)
        print("[*] Частоты ГП сброшены в авто-режим.")
    except:
        pass
    
    if 'shm_in' in locals(): shm_in.close()
    if 'shm_res' in locals(): shm_res.close()
    pynvml.nvmlShutdown()