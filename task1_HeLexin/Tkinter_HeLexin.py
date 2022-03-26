import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.filedialog

def hit_me1():
  global on_hit1
  if on_hit1==False:
    on_hit=True
    var1.set("model3.obj+Вариант 5")
  else:
    on_hit1=False
    var1.set(" ")

def hit_me2():
  global on_hit2
  if on_hit2==False:
    on_hit2=True
    var2.set("coefficients.csv")
  else:
    on_hit2=False
    var2.set(" ")

def hit_me3():
  global on_hit3
  if on_hit3==False:
    on_hit3=True
    var3.set("Results.csv")
  else:
    on_hit3=False
    var3.set(" ")

def hit_me4():
  global on_hit4
  if on_hit4==False:
    on_hit4=True
    var4.set("34.17016314")
  else:
    on_hit4=False
    var4.set(" ")

def hit_me5():
  global on_hit5
  if on_hit5==False:
    on_hit5=True
    var5.set("-8.72051142")
  else:
    on_hit5=False
    var5.set(" ")

window=tk.Tk()
window.title("Расчет и визуализация теплового режима космического аппарата")
window.geometry("640x480")

var1=tk.StringVar()
l1=tk.Label(window,textvariable=var1,bg="white",font=("Arial",12),width=20,height=2)
l1.place(x=300,y=20)

var2=tk.StringVar()
l2=tk.Label(window,textvariable=var2,bg="white",font=("Arial",12),width=20,height=2)
l2.place(x=300,y=120)

var3=tk.StringVar()
l3=tk.Label(window,textvariable=var3,bg="white",font=("Arial",12),width=20,height=2)
l3.place(x=300,y=220)

var4=tk.StringVar()
l4=tk.Label(window,textvariable=var4,bg="white",font=("Arial",12),width=20,height=2)
l4.place(x=300,y=320)

var5=tk.StringVar()
l5=tk.Label(window,textvariable=var5,bg="white",font=("Arial",12),width=20,height=2)
l5.place(x=300,y=420)

on_hit1=False
b1 = tk.Button(window,text="Модель+Вариант", width=15, height=2,command=hit_me1)
b1.place(x = 100,y = 20)

on_hit2=False
b2=tk.Button(window,text="Коэффициенты",width=15,height=2,command=hit_me2)
b2.place(x = 100,y = 120)

on_hit3=False
b3=tk.Button(window,text="Результаты",width=15,height=2,command=hit_me3)
b3.place(x = 100,y = 220)

on_hit4=False
b4=tk.Button(window,text="T_max",width=15,height=2,command=hit_me4)
b4.place(x = 100,y = 320)

on_hit5=False
b5=tk.Button(window,text="T_min",width=15,height=2,command=hit_me5)
b5.place(x = 100,y = 420)

window.mainloop()

