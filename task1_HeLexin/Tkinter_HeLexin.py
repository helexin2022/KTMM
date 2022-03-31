import tkinter as tk
from tkinter import *
from tkinter import ttk
import tkinter.filedialog
import task1_2

def hit_me1():
  var1.set("model3.obj+Вариант 5")

def hit_me2():
   var2.set("coefficients.csv")

def hit_me3():
  var3.set("Results.csv")


def M():
  M = task1_2.Max
  var4.set(M)

def m():
  m = task1_2.Min
  var5.set(m)

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


b1 = tk.Button(window,text="Модель+Вариант", width=15, height=2,command=hit_me1)
b1.place(x = 100,y = 20)

b2=tk.Button(window,text="Коэффициенты",width=15,height=2,command=hit_me2)
b2.place(x = 100,y = 120)

b3=tk.Button(window,text="Результаты",width=15,height=2,command=hit_me3)
b3.place(x = 100,y = 220)

b4=tk.Button(window,text="T_max",width=15,height=2,command=lambda:M())
b4.place(x = 100,y = 320)

b5=tk.Button(window,text="T_min",width=15,height=2,command=lambda:m())
b5.place(x = 100,y = 420)

window.mainloop()


