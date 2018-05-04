#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  : mystic
# @date    : 2018/5/1 19:23
from tkinter import Tk, Label, Entry, Button, IntVar, Checkbutton

import matplotlib
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from numpy import mat, arange

import tree_regression

matplotlib.use('TkAgg')


def re_draw(tolerance_s, tolerance_n):
    re_draw.fig.clf()  # clear the figure
    re_draw.a = re_draw.fig.add_subplot(111)
    y_hat = None
    if chk_btn_var.get():
        if tolerance_n < 2:
            tolerance_n = 2
            my_tree = tree_regression.create_tree(re_draw.raw_data, tree_regression.model_leaf,
                                                  tree_regression.model_err, (tolerance_s, tolerance_n))
            y_hat = tree_regression.create_forecast(my_tree, re_draw.test_data, tree_regression.model_tree_evaluation)
    else:
        my_tree = tree_regression.create_tree(re_draw.raw_data, ops=(tolerance_s, tolerance_n))
        y_hat = tree_regression.create_forecast(my_tree, re_draw.test_data)
    re_draw.a.scatter(re_draw.raw_data[:, 0], re_draw.raw_data[:, 1], s=5)
    re_draw.a.plot(re_draw.test_data, y_hat, linewidth=2.0)


def draw_new_tree():
    pass


root = Tk()

re_draw.fig = Figure(figsize=(5, 4), dpi=100)  # create canvas
re_draw.canvas = FigureCanvasTkAgg(re_draw.fig, master=root)
re_draw.canvas.show()
re_draw.canvas.get_tk_widget().grid(row=0, columnspan=3)

Label(root, text='Plot Place Holder').grid(row=0, columnspan=3)

Label(root, text='tolerance_n').grid(row=1, column=0)
tol_n_entry = Entry(root)
tol_n_entry.grid(row=1, column=1)
tol_n_entry.insert(0, '10')

Label(root, text='tolerance_s').grid(row=2, column=0)
tol_s_entry = Entry(root)
tol_s_entry.grid(row=2, column=1)
tol_s_entry.insert(0, '1.0')

Button(root, text='ReDraw', command=draw_new_tree).grid(row=1, column=2, rowspan=3)

chk_btn_var = IntVar()
chk_btn = Checkbutton(root, text='Model Tree', variable=chk_btn_var)
chk_btn.grid(row=3, column=0, columnspan=2)

re_draw.raw_data = mat(tree_regression.load_data_set('resource/sine.txt'))
re_draw.test_data = arange(min(re_draw.raw_data[:, 0]), max(re_draw.raw_data[:, 0]), 0.01)

re_draw(1.0, 10)

root.mainloop()

if __name__ == '__main__':
    print('++++++++++++++++main++++++++++++++++')
