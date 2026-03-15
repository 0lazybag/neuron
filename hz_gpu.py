# -*- coding: utf-8 -*-
import tkinter as tk
import pynvml
import sys


class МониторГП:
    def __init__(self):
        # Инициализация NVML
        try:
            pynvml.nvmlInit()
            self.хэндл = pynvml.nvmlDeviceGetHandleByIndex(0)
        except pynvml.NVMLError as ошибка:
            print(f"Ошибка инициализации: {ошибка}")
            sys.exit()

        # Создание окна
        self.root = tk.Tk()
        self.root.title("GPU")
        
        # Размер 100x100 и положение (например, в углу экрана)
        self.root.geometry("100x100+100+100")
        
        # Окно всегда поверх остальных
        self.root.attributes("-topmost", True)
        
        # Настройка текста
        self.label_title = tk.Label(self.root, text="GPU MHz", font=("Arial", 10, "bold"))
        self.label_title.pack(pady=5)
        
        self.label_freq = tk.Label(self.root, text="0", font=("Arial", 14))
        self.label_freq.pack(pady=10)

        # Запуск цикла обновления
        self.обновить_данные()
        self.root.protocol("WM_DELETE_WINDOW", self.закрыть)
        self.root.mainloop()

    def обновить_данные(self):
        try:
            # Получаем частоту
            частота = pynvml.nvmlDeviceGetClockInfo(self.хэндл, pynvml.NVML_CLOCK_GRAPHICS)
            self.label_freq.config(text=f"{частота}")
        except pynvml.NVMLError:
            self.label_freq.config(text="Ошибка")
        
        # Обновление каждые 40мс (25 FPS)
        self.root.after(40, self.обновить_данные)

    def закрыть(self):
        pynvml.nvmlShutdown()
        self.root.destroy()

if __name__ == "__main__":
    МониторГП()