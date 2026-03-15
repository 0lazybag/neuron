from multiprocessing import shared_memory
import time
import pynvml
import subprocess
import sys

# --- НАСТРОЙКИ ---
exe_path = r"D:\neiron\neiron.exe"
МИН_ЧАСТОТА = 210
МАКС_ЧАСТОТА = 3105
МНОЖИТЕЛЬ = 1.1

def run_neiron():
    """Запускает EXE и считывает два динамических адреса из терминала"""
    try:
        процесс = subprocess.Popen(
            [exe_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding='utf-8', # Если будут ошибки в тексте, попробуй 'cp866'
            bufsize=1
        )

        # Читаем первые две строки — это должны быть имена адресов памяти
        адрес_входа = процесс.stdout.readline().strip()
        адрес_выхода = процесс.stdout.readline().strip()

        if адрес_входа and адрес_выхода:
            print(f"[*] Стримы получены: Вход={адрес_входа}, Выход={адрес_выхода}")
            return адрес_входа, адрес_выхода, процесс
        else:
            print("[!] EXE не предоставил адреса памяти.")
            return None, None, None

    except Exception as ошибка:
        print(f"[!] Ошибка запуска EXE: {ошибка}")
        return None, None, None

def изменить_частоту(хэндл_гп, увеличить=True):
    """Управляет частотой графического ядра"""
    try:
        текущая = pynvml.nvmlDeviceGetClockInfo(хэндл_гп, pynvml.NVML_CLOCK_GRAPHICS)
        
        if увеличить:
            новая = int(текущая * МНОЖИТЕЛЬ)
            статус = "РАЗГОН"
        else:
            новая = int(текущая / МНОЖИТЕЛЬ)
            статус = "СБРОС"
        
        # Ограничиваем частоту рамками железа
        новая = max(МИН_ЧАСТОТА, min(новая, МАКС_ЧАСТОТА))
        
        # Установка частоты (требует прав админа)
        pynvml.nvmlDeviceSetGpuLockedClocks(хэндл_гп, новая, новая)
        print(f"[*] ГП: {текущая}MHz -> {новая}MHz ({статус})")
        
    except pynvml.NVMLError as err:
        print(f"[!] Ошибка NVML (нужен админ): {err}")

# --- ОСНОВНАЯ ЛОГИКА ---

# 1. Инициализация NVML
try:
    pynvml.nvmlInit()
    gpu_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
except Exception as e:
    print(f"[!] Не удалось инициализировать NVML: {e}")
    sys.exit()

# 2. Запуск нейронки и получение адресов
адрес_отправки, адрес_чтения, neiron_proc = run_neiron()

if not адрес_отправки or not адрес_чтения:
    print("[!] Работа невозможна без адресов. Выход.")
    pynvml.nvmlShutdown()
    sys.exit()

# 3. Подключение к SharedMemory и основной цикл
try:
    shm_in = shared_memory.SharedMemory(name=адрес_отправки)
    shm_res = shared_memory.SharedMemory(name=адрес_чтения)
    
    print("\n[*] Связь с ОЗУ установлена. Система готова.")
    print(f"[*] Диапазон: {МИН_ЧАСТОТА}-{МАКС_ЧАСТОТА} MHz")

    while True:
        команда = input("\nВвод (0-255) или 'exit': ").strip()
        
        if команда.lower() == 'exit':
            break
            
        try:
            значение = int(команда)
            if 0 <= значение <= 255:
                # Запоминаем текущий ответ для отслеживания изменений
                старый_ответ = shm_res.buf[0]
                
                # Пишем данные для EXE
                shm_in.buf[0] = значение
                print(f"[>] Отправлено: {значение}. Ждем ответ...")
                
                # Ожидание реакции (таймаут 1.5 сек)
                таймер = 0
                while shm_res.buf[0] == старый_ответ and таймер < 30:
                    time.sleep(0.05)
                    таймер += 1
                
                ответ_exe = shm_res.buf[0]
                print(f"[<] Ответ нейронки: {ответ_exe}")

                # Логика управления частотой
                if ответ_exe == значение:
                    print("[+] Совпадение! Повышаем частоту.")
                    изменить_частоту(gpu_handle, увеличить=True)
                else:
                    print("[-] Неверно. Снижаем частоту.")
                    изменить_частоту(gpu_handle, увеличить=False)
            else:
                print("[-] Введи число от 0 до 255.")
        except ValueError:
            print("[-] Нужна цифра.")

except Exception as e:
    print(f"[!] Критическая ошибка: {e}")

finally:
    # Очистка ресурсов при выходе
    print("\n[*] Завершение работы...")
    
    try:
        pynvml.nvmlDeviceResetGpuLockedClocks(gpu_handle)
        print("[*] Частоты ГП возвращены в авто-режим.")
    except:
        pass
    
    if 'shm_in' in locals(): shm_in.close()
    if 'shm_res' in locals(): shm_res.close()
    
    if neiron_proc:
        neiron_proc.terminate() # Закрываем EXE
        print("[*] EXE процесс завершен.")
        
    pynvml.nvmlShutdown()
    print("[*] NVML выключен. Пока.")