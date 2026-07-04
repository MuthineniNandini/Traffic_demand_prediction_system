# from tkinter import*
# from PIL import Image,ImageTk
# window=Tk()
# window.title("first")
# img=Image.open(r"C:\Users\nandi\OneDrive\Desktop\tkinter\TKINTER\134093276330436966.jpg")
# img = img.resize((300, 300)) 
# pho=ImageTk.PhotoImage(img)
# window.iconphoto(True,pho)
# #DAY2
# lab=Label(window,text="welcome",
#                 compound="top",
#                 font=('Arial',40,'bold'),
#                 fg="pink",
                
#                 bg='blue',
#                 relief=RAISED,
#                 bd=10,
#                 padx=20,
#                 pady=20,
#                 image=pho
#                 )
# lab.pack(padx=10,
#         pady=10)


# window.mainloop()
# CONCEPT 3----BUTTONS
# from tkinter import *
# from PIL import Image,ImageTk
# wid=Tk()
# c=0
# def click():
#         global c
#         c+=1
#         lab.config(text=c)
# img=Image.open(r"C:\Users\nandi\OneDrive\Desktop\tkinter\TKINTER\134093276330436966.jpg")
# img=img.resize((100,100))
# pho=ImageTk.PhotoImage(img)
# but=Button(wid,text="hit me",
#                 command=click,
                # font=("Arial",10,"bold"),
                # fg="pink",
                # bg='blue',
                # relief=RAISED,
                # bd=10,
                # padx=20,
                # pady=20,
                # image=pho,
                # compound="bottom",
                # activebackground="red",
                # activeforeground="green")
# but.config(state=DISABLED)
# lab=Label(wid,text=c)
# lab.pack()
# but.pack()
# wid.mainloop()
# TEXT WIDGET 
# from tkinter import*
# # from PIL import Image,ImageTk
# wid=Tk()
# def submit():
#     txt=entry.get()
#     print("here yours"+txt)
# def delete():
#     entry.delete(0,END)
# def backspace():
#     entry.delete(len(entry.get())-1,END)
# sub=Button(wid,text="submit",command=submit)
# dele=Button(wid,text="del",command=delete)
# backspace=Button(wid,text="backspace",command=backspace)
# entry=Entry(wid,font=("ink free",50),
#                 width=20,
#                 show="." )
# entry.insert(0,"kjhgfds")
# sub.pack()
# dele.pack()
# backspace.pack()
# entry.pack()
# wid.mainloop()
# CHECKBOX------------
# from tkinter import *
# from PIL import Image,ImageTk

# wid=Tk()
# def display():
#     if (x.get()==1):
#         print("ON")
#     else:
#         print("OFF")
# x=IntVar()
# img=Image.open(r"C:\Users\nandi\OneDrive\Desktop\tkinter\TKINTER\134093276330436966.jpg")
# img=img.resize((100,100))
# pho=ImageTk.PhotoImage(img)
# cb=Checkbutton(wid,text="check",
#                 variable=x,
#                 onvalue=1,
#                 offvalue=0,
#                 command=display,
#                 font=("Arial",10,"bold"),
#                 fg="pink",
#                 bg='blue',
#                 relief=RAISED,
#                 bd=10,
#                 padx=20,
#                 pady=20,
                
#                 activebackground="red",
#                 activeforeground="green")
# cb.config(image=pho,compound="bottom",)
# cb.config(anchor='e')#to align different widgets in same line and direction 
# cb.pack()
# wid.mainloop()

# RADIOBUTTON--------------
from tkinter import *
from PIL import Image,ImageTk
def display():
    l=['pizza','burger']
    print(l[x.get()])
food=['pizza','burger']
wid=Tk()
x=IntVar()
img1=Image.open(r"C:\Users\nandi\OneDrive\Desktop\tkinter\TKINTER\pizza.png")
img1=img1.resize((200,200))
pizzaI=ImageTk.PhotoImage(img1)

img=Image.open(r"C:\Users\nandi\OneDrive\Desktop\tkinter\TKINTER\burger.png")
img=img.resize((200,200))
burgerI=ImageTk.PhotoImage(img)
photos=[pizzaI,burgerI]
for i in range(len(food)):
    rb=Radiobutton(wid,text=food[i],variable=x,value=i,
                    width=100,
                    height=100,
                    padx=10,
                    image=photos[i],
                    compound='left',
                    indicatoron=0,
                    command=display)
    rb.pack(anchor='w')
wid.mainloop()