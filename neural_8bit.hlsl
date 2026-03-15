#version 430

layout(local_size_x = 8) in; // Запускаем 8 потоков (по одному на выходной бит)

// Буферы данных
layout(std430, binding = 0) buffer InputBuffer  { uint input_val; };
layout(std430, binding = 1) buffer StateBuffer  { uint last_val; };
layout(std430, binding = 2) buffer OutputBuffer { uint output_val; };

// Буферы весов (Матрицы)
layout(std430, binding = 3) buffer WeightsA { float wA[8 * 32]; }; // Вход -> Скрытый
layout(std430, binding = 4) buffer WeightsB { float wB[32 * 8]; }; // Скрытый -> Выход

uniform float delta_freq;
uniform float learning_rate;

// Функция активации (чтобы нейроны "ожили")
float relu(float x) {
    return max(0.0, x);
}

void main() {
    uint idx = gl_LocalInvocationID.x; // ID текущего бита (0-7)

    // 1. Проверка на изменение входных данных (только первым потоком)
    if (idx == 0) {
        if (input_val == last_val) return; 
    }
    barrier(); // Ждем, пока все потоки проверят условие

    // --- СЛОЙ 1: Вход (8) -> Скрытый (32) ---
    float hidden[32];
    for (int j = 0; j < 32; j++) {
        float sum = 0.0;
        for (int i = 0; i < 8; i++) {
            float bit = float((input_val >> i) & 1u);
            sum += bit * wA[i * 32 + j];
        }
        hidden[j] = relu(sum);
    }

    // --- СЛОЙ 2: Скрытый (32) -> Выходной бит (текущий idx) ---
    float final_sum = 0.0;
    for (int j = 0; j < 32; j++) {
        final_sum += hidden[j] * wB[j * 8 + idx];
    }

    // Активация выходного бита
    uint out_bit = (final_sum > 0.5) ? 1u : 0u;

    // --- СБОРКА РЕЗУЛЬТАТА ---
    if (out_bit == 1u) {
        atomicOr(output_val, (1u << idx));
    } else {
        atomicAnd(output_val, ~(1u << idx));
    }

    // --- ОБУЧЕНИЕ (Backpropagation "на коленке") ---
    // Если частота ГП выросла (delta_freq > 0), двигаем веса к активации
    float grad = delta_freq * learning_rate;
    
    for (int j = 0; j < 32; j++) {
        // Правим веса второго слоя
        wB[j * 8 + idx] += grad * hidden[j];
        
        // Правим веса первого слоя (упрощенно)
        for (int i = 0; i < 8; i++) {
            float bit = float((input_val >> i) & 1u);
            wA[i * 32 + j] += grad * bit * 0.1; // Меньший шаг для глубокого слоя
        }
    }

    // Обновляем состояние (только один поток)
    if (idx == 0) last_val = input_val;
}