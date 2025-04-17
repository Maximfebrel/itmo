import numpy as np


def im2col(X, HF, WF, stride, pad):
    # паддинг
    X_padded = np.pad(X, ((0,0), (0,0), (pad, pad), (pad, pad)), mode='constant')
    # получаем индексы для матрицы
    i, j, d = get_indices(X.shape, HF, WF, stride, pad)
    # создаем матрицу столбцов
    cols = X_padded[:, d, i, j]
    cols = np.concatenate(cols, axis=-1)
    return cols


def col2im(dX_col, X_shape, HF, WF, stride, pad):
    # размер входного изображения
    N, D, H, W = X_shape
    # паддинг
    H_padded, W_padded = H + 2 * pad, W + 2 * pad
    X_padded = np.zeros((N, D, H_padded, W_padded))

    # получаем индексы для матрицы
    i, j, d = get_indices(X_shape, HF, WF, stride, pad)
    # разбиваем dX_col на N строк
    dX_col_reshaped = np.array(np.hsplit(dX_col, N))
    # добавляем к исходной матрице расчитанную, возвращая ее к исходному состоянию
    np.add.at(X_padded, (slice(None), d, i, j), dX_col_reshaped)
    if pad == 0:
        return X_padded
    elif type(pad) is int:
        return X_padded[pad:-pad, pad:-pad, :, :]


def get_indices(X_shape, HF, WF, stride, pad):
    # размер входного изображения
    m, n_C, n_H, n_W = X_shape

    # размер выходного изображения
    out_h = int((n_H + 2 * pad - HF) / stride) + 1
    out_w = int((n_W + 2 * pad - WF) / stride) + 1

    # индексы для первого канала
    level1 = np.repeat(np.arange(HF), WF)
    # дублируем для остальных каналов
    level1 = np.tile(level1, n_C)
    # вектор с горизонтальными индексами
    everyLevels = stride * np.repeat(np.arange(out_h), out_w)
    # создаем матрицу индексов для каждого канала
    i = level1.reshape(-1, 1) + everyLevels.reshape(1, -1)

    # индексы для первого канала
    slide1 = np.tile(np.arange(WF), HF)
    # дублируем для остальных каналов
    slide1 = np.tile(slide1, n_C)
    # вектор с горизонтальными индексами
    everySlides = stride * np.tile(np.arange(out_w), out_h)
    # создаем матрицу индексов для каждого канала
    j = slide1.reshape(-1, 1) + everySlides.reshape(1, -1)

    # создаем матрицу индексов для ограничения границ каждого канала
    d = np.repeat(np.arange(n_C), HF * WF).reshape(-1, 1)

    return i, j, d

