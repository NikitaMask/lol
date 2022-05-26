from PIL import Image, ImageDraw
from numpy import asarray
import random
import numpy as np
from sympy import *
import math
from tkinter import Tk,Button,filedialog as fd
from tkinter import *
#g sys для исправления  ошибок
matrix = np.array([[1,0,0,0,0,0,0,0,1,0,1,0,0,1,1,1,1,0,0],
 [0 ,1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
 [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
 [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
 [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
 [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
 [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
 [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]])

#функциии для открытия картинок только в пнг джпг
name = ""
def callback():
    ftypes = [("Картинки","*.jpg"),("Картинки2", "*.png")]
    name = fd.askopenfilename(filetypes = ftypes)
    return name

def call():
    global name
    name = callback()

window = Tk()

#вывод картинки в оригинальном виде(использовалась библеотека пил)
def orig():
    global name
    img = Image.open(name)
    numpydata = asarray(img)

    draw = ImageDraw.Draw(img)
    width = img.size[0]
    height = img.size[1]
    # Первая часть(вывод обычной картинки)
    for i in range(height):
        for j in range(width):
            draw.point((j, i), (numpydata[i][j][0], numpydata[i][j][1], numpydata[i][j][2]))

    img.show()
#вывод зашумленной картинки
def shum():
    global name
    img = Image.open(name)
    numpydata = asarray(img)

    draw = ImageDraw.Draw(img)
    width = img.size[0]
    height = img.size[1]
    #функция для добавления двух ошибок
    def oshibki(a):
        spisok = range(0,8)
        ind = random.sample(spisok, 2)
        for i in range(len(ind)):
            if int(a[ind[i]]) == 0:
                a[ind[i]] = "1"
            else:
                a[ind[i]] = '0'
        return a

    # Вторая часть(Вывод зашумленной)берет каждое рджби переносится в двоичное вносится ошибка переносится в десятичное
    #и выводится картинка
    for i in range(height):
        for j in range(width):
            lol = []
            for l in range(3):
                a = bin(numpydata[i][j][l])[2:].zfill(8)
                c = ''.join(oshibki(list(a)))
                lol.append(int(c, 2))
            draw.point((j, i), (lol[0], lol[1], lol[2]))

    img.show()
#кодирование декодирование зашумление
def kodidecodi():
    img = Image.open(name)
    numpydata = asarray(img)

    draw = ImageDraw.Draw(img)
    width = img.size[0]
    height = img.size[1]
    global matrix
    # УДАЛЕНИЕ ИЗ МАТРИЦЫ СТОЛБЦОВ ДЛЯ ЕДИНИЧНОЙ
    def delite(matrix):
        pomoh = []
        stolb = 0
        stolb88 = 0
        ed = np.eye(len(matrix))
        for i in range(len(matrix)):
            stolb = ed[:, i]
            for j in range(len(matrix[0])):
                stolb88 = matrix[:, j]
                if (stolb == stolb88).all():
                    pomoh.append(j)
                    stolb = 0
        matrix = np.delete(matrix, np.s_[pomoh], axis=1)
        return matrix
#для того чтобы заменить в матрице все на 0 и 1
    def spisok(kod):
        for i in range(len(kod)):
            if kod[i] % 2 == 0:
                kod[i] = 0
            if kod[i] > 1 and kod[i] % 2 != 0:
                kod[i] = 1
        return kod
#добавление ошибок для другого числа
    def oshibki2(kod1):
        spisok = range(0,19)
        ind = random.sample(spisok, 2)
        for i in range(len(ind)):
            if int(kod[ind[i]]) == 0:
                kod[ind[i]] = 1
            else:
                kod[ind[i]] = 0
        return kod
#декодирование
#передаем вектор ошибки
#каждый вектор ошибки умножается на S штрих, в матрице заменяются на 0 и 1
    def decoder(a1, matrix2_t, matrica, matrix_t_S, matrix_c, ogoSpis2):
        # Вектор
        vectrospis = []
        vectrospisOsn = []
        vectrospis.append(a1)
        for i in vectrospis:
            qw = list(i)
            qw = list(map(int, qw))
            vectrospisOsn.append(qw)
       # print("Вектор ошибки\n",vectrospisOsn)
        s_ = np.dot(vectrospisOsn, matrix2_t)
       # print("SSSS\n",s_)
        for i in range(len(s_)):
            for j in range(len(s_[i])):
                if s_[i][j] % 2 == 0:
                    s_[i][j] = 0
                if s_[i][j] % 2 != 0:
                    s_[i][j] = 1
#сравниваем наше S штрих с обычной S и находим е
        hello = 0
        for i in range(len(matrica)):
            if (s_[0] == matrix_t_S[i]).all():
                hello = matrica[i]
        hello = np.array([hello])

        # Находим c': складываем векторы
        Cspis = []
        for i in range(len(vectrospisOsn)):
            for j in range(len(vectrospisOsn[i])):
                Cspis.append((vectrospisOsn[i][j] + hello[i][j]))
        c_ = np.array([Cspis])
        for i in range(len(c_)):
            for j in range(len(c_[i])):
                if c_[i][j] % 2 == 0:
                    c_[i][j] = 0
                if c_[i][j] % 2 != 0:
                    c_[i][j] = 1
#сравниваем С штрих с обычной С и находим информационное слово i
        iii = 0
        for i in range(len(matrix_c)):
            if (c_[0] == matrix_c[i]).all():
                iii = ogoSpis2[i]
        iiM = np.array(int(''.join(map(str, iii)), 2))
        return iiM

    # Третья часть

    # Gsys
    matrix3 = np.array([[1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 1, 1, 1, 0, 0],
                        [0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 0, 0, 1, 0, 1],
                        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 0],
                        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0, 0],
                        [0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1],
                        [0, 0, 0, 0, 0, 1, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 0, 1],
                        [0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 0, 1, 1, 1, 0, 0, 0, 0],
                        [0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 0, 1, 1, 0]])

    matrix = delite(matrix)
    # Hsys
    matrix2 = matrix.transpose()
    ed = np.eye(len(matrix2))
    matrix2 = np.append(matrix2, ed, axis=1)
    # print("Матрица Hsys")
    # vivodMAT(matrix2)

    # i нахождение
    vay = []
    for i in range(2 ** (len(matrix3))):
        s = bin(i)[2:]
        vay.append(s.zfill(len(matrix3)))
    # print(ogoSpis)
# список с i поэлементно
    vay2 = []
    for i in vay:
        vot = list(i)
        vot = list(map(int, vot))
        vay2.append(vot)
    # print(ogoSpis2)
# находим размерность матрицы и минимальное количество единиц
    wthSpis = []
    matrix_c = np.dot(vay2, matrix3)
    for i in range(len(matrix_c)):
        wth = 0
        for j in range(len(matrix_c[i])):
            if matrix_c[i][j] == 1:
                wth = wth + 1
            if matrix_c[i][j] % 2 == 0:
                matrix_c[i][j] = 0
            if matrix_c[i][j] > 1 & matrix_c[i][j] % 2 != 0:
                matrix_c[i][j] = 1
        wthSpis.append(wth)

    print("n =", len(matrix3[0]), " ", "k =", len(matrix3), " ", "d =", min(wthSpis[1:]))
    dmin = min(wthSpis[1:])
    print("")

    # Решение уровенения( с прошлой лабы это сколько матрица исправляет ошибок)
    x = Symbol('x')
    t = solve(2 * x + 1 - dmin, x)
    t = math.floor(t[0])
    ro = solve(x + 1 - dmin, x)
    print("t =", t, " ", "p =", ro[0])
    print("")
    if t == 0:
        print("Нечего исправлять")
        exit()

    print("ЛИДЕРНОЕ ДЕКОДИРОВАНИЕ")
    print("")
#H sys транспонированное
    matrix2_t = np.transpose(matrix2)
    print("Hsys-T")
    print(matrix2_t)
    print("")
#матрица из элементов по 19 значений со всевозможным расположением одной 1 и двух
    print("e")
    Spis12 = []
    for i in range(2 ** (19)):
        s = bin(i)[2:]
        Spis12.append(s.zfill(19))

    Spis22 = []
    for i in range(len(Spis12)):
        if Spis12[i].count("1") == 1:
            Spis22.append(Spis12[i])
        if Spis12[i].count("1") == 2:
            Spis22.append(Spis12[i])

    Spis23 = []
    for i in Spis22:
        vot = list(i)
        vot = list(map(int, vot))
        Spis23.append(vot)
    matrica = np.array(Spis23)
#матрица по тетрадке
    matrix_t_S = np.dot(matrica, matrix2_t)

    for i in range(len(matrix_t_S)):
        for j in range(len(matrix_t_S[i])):
            if matrix_t_S[i][j] % 2 == 0:
                matrix_t_S[i][j] = 0
            if matrix_t_S[i][j] > 1 and matrix_t_S[i][j] % 2 != 0:
                matrix_t_S[i][j] = 1
    print("S")

    print("")
    #перевод в двоичную каждого рджби , кодирование, добавление ошибок, декодирование с исправлением ошибок и вывод картинки
    slovaosh = []
    kodslova = []
    slova = []
    for i in range(height):
        for j in range(width):
            norm = []
            for l in range(3):
                a = bin(numpydata[i][j][l])[2:].zfill(8)
                slova.append(a)
                a = list(map(int, a))
                kod = np.dot(a, matrix3)
                kod1 = spisok(kod).tolist()
                kodslova.append(kod1)
                a1 = oshibki2(kod1).tolist()
                slovaosh.append(oshibki2(kod1).tolist())
                norm.append((decoder(a1, matrix2_t, matrica, matrix_t_S, matrix_c, vay2)).tolist())
            draw.point((j, i), (norm[0], norm[1], norm[2]))
    img.show()

    print("Информационные слова:\n",slova)
    print("Закодированные информационные слова:\n",kodslova)
    print("Закодированные информационные слова с ошибкой:\n",slovaosh)




btn1 = Button(text = "ORIGINAL",command = orig, bg="white")
btn1.pack()
btn2 = Button(text = "SHUM", command = shum, bg="blue")
btn2.pack()
btn3 = Button(text = "KODER AND DECODER", command = kodidecodi, bg="red")
btn3.pack()
btn4 = Button(text = "FILE",command = call, bg="green")
btn4.pack()
window.title("G")
window.mainloop()