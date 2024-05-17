#!/usr/bin/env python
# -*- coding: utf-8 -*-
# Created by PyCharm
# @author  : mystic
# @date    : 2/22/2018 14:49

import numpy as np
import pyqtgraph as pg

x = np.random.random(50)
y = np.random.random(8)


def pg_plot():
    app = pg.QtGui.QApplication([])
    pg.plot(x, title="=============console============= plot() x数组")
    pg.plot(y, title="=============console============= plot() y数组")
    app.exec_()


def pg_plot_add():
    plot = pg.plot(title="=============console============= plot()")
    plot.plot(x)
    plot.plot(y, pen="r")
    pg.QtGui.QGuiApplication.exec_()


def pg_windows_add_plot():
    win = pg.GraphicsWindow(title="=============console=============")
    plot = win.addPlot(title="窗口绘制图形")
    plot.plot(x)
    plot2 = win.addPlot(title="绘制子图")
    plot2.plot(y)
    pg.QtGui.QApplication.exec_()


def pg_graphics_layout_plot():
    app = pg.QtGui.QApplication([])
    view = pg.GraphicsView()
    layout = pg.GraphicsLayout()
    view.setCentralItem(layout)
    view.show()
    p1 = layout.addPlot(title="=============console============= - 通过图形层绘制图形")
    p1.plot(x)
    app.exec_()


def pg_plot_widget_plot():
    app = pg.QtGui.QApplication([])
    pw = pg.PlotWidget(title="=============console============= - 通过图形部件绘制图形")
    pw.plot(x)
    pw.show()
    app.exec_()


def pg_plot_item_plot():
    app = pg.QtGui.QApplication([])
    xx = pg.GraphicsView()
    plot = pg.PlotItem(title="=============console============= - 通过图形部件绘制图形")
    plot.plot(x)
    xx.setCentralWidget(plot)
    xx.show()
    app.exec_()


if __name__ == "__main__":
    pg_plot_item_plot()
