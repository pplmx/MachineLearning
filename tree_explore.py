#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @author  : mystic
# @date    : 2018/5/1 19:23
from tkinter import Tk, Label, Entry, Button, IntVar, Checkbutton

from numpy import mat, arange

import tree_regression


def tkinter_in_use():
    root = Tk()

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

    global raw_data
    global test_data
    raw_data = mat(tree_regression.load_data_set('resource/sine.txt'))
    test_data = arange(min(raw_data[:, 0]), max(raw_data[:, 0]), 0.01)

    re_draw(1.0, 10)

    root.mainloop()


def re_draw(tolerance_s, tolerance_n):
    pass


def draw_new_tree():
    pass


if __name__ == '__main__':
    tkinter_in_use()
