import tkinter as tk

def toggle_fs(dummy=None):
    state = False if root.attributes('-fullscreen') else True
    root.attributes('-fullscreen', state)
    if not state:
        root.geometry('300x300+100+100')

root = tk.Tk()
root.attributes('-fullscreen', True) # make main window full-screen

canvas1 = tk.Canvas(root, bg='white', highlightthickness=0)
canvas1.pack(fill=tk.BOTH, expand=True) # configure canvas to occupy the whole main window


# with sklearn
# Intercept_result = ('Intercept: ', lm.intercept_)
# label_Intercept = tk.Label(root, text=Intercept_result, justify = 'center')
# canvas1.create_window(260, 220, window=label_Intercept)

# # with sklearn
# Coefficients_result  = ('Coefficients: ', lm.coef_)
# label_Coefficients = tk.Label(root, text=Coefficients_result, justify = 'center')
# canvas1.create_window(260, 240, window=label_Coefficients)
# index=['timestamp','temperature.gpu','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]' ,'memory.used [MiB]']
# timestamp label and input box
label1 = tk.Label(root, text='Timestamp: ', font='Helvetica 10 bold')
canvas1.create_window(300, 80, window=label1)

ts = tk.Entry (root) # create 1st entry box
canvas1.create_window(415, 80, window=ts)

# Temperature label and input box
label2 = tk.Label(root, text=' temperature.gpu: ', font='Helvetica 10 bold')
canvas1.create_window(610, 80, window=label2)

temp = tk.Entry (root) # create 2nd entry box
canvas1.create_window(730, 80, window=temp)

# Utilization.gpu label and input box
label3 = tk.Label(root, text=' utilization.gpu [%]: ', font='Helvetica 10 bold')
canvas1.create_window(978, 80, window=label3)

util_gpu = tk.Entry (root) # create 3rd entry box
canvas1.create_window(1100, 80, window=util_gpu)


# timestamp label and input box
label4 = tk.Label(root, text='utilization.memory [%]: ', font='Helvetica 10 bold')
canvas1.create_window(280, 140, window=label4)

util_mem = tk.Entry (root) # create 4th entry box
canvas1.create_window(415, 140, window=util_mem)

# Temperature label and input box
label5 = tk.Label(root, text=' clocks.current.sm [MHz]: ', font='Helvetica 10 bold')
canvas1.create_window(590, 140, window=label5)

sm_freq = tk.Entry (root) # create 5th entry box
canvas1.create_window(730, 140, window=sm_freq)

# Utilization.gpu label and input box
label6 = tk.Label(root, text=' memory.used [MiB]: ', font='Helvetica 10 bold')
canvas1.create_window(965, 140, window=label6)

mem_used = tk.Entry (root) # create 6th entry box
canvas1.create_window(1100, 140, window=mem_used)

def values(): 
    global New_Time_Stamp #our 1st input variable
    New_Time_Stamp = float(ts.get()) 
    
    global New_Temperature #our 2nd input variable
    New_Temperature = float(temp.get()) 
    
    global New_Utilization_GPU #our 3rd input variable
    New_Utilization_GPU = float(util_gpu.get())

    global New_Utilization_Mem #our 4th input variable
    New_Utilization_Mem = float(util_mem.get())

    global New_SM_Freq #our 5th input variable
    New_SM_Freq = float(sm_freq.get())

    global New_Mem_Used #our 5th input variable
    New_Mem_Used = float(mem_used.get()) 

    test_df = pd.DataFrame({
    "Features":[New_Time_Stamp,New_Temperature,New_Utilization_GPU,New_Utilization_Mem,New_SM_Freq,New_Mem_Used]
    }, 
    index=['timestamp','temperature.gpu','utilization.gpu [%]','utilization.memory [%]','clocks.current.sm [MHz]' ,'memory.used [MiB]']
    )
    
    x_test_df = sc_x.fit_transform(test_df)
    Prediction_result  = ('Predicted Power (W): ', lm.predict(x_test_df.reshape(1,-1)))
    label_Prediction = tk.Label(root, text= Prediction_result, font='Verdana 10 bold',bg='orange')
    canvas1.create_window(770, 200, window=label_Prediction)
    
button1 = tk.Button (root, text='Predict Power (W)',command=values, bg='deepskyblue') # button to call the 'values' command above 
canvas1.create_window(570, 200, window=button1)
 
 
'''
#plot 1st scatter 
figure3 = plt.Figure(figsize=(5,4), dpi=100)
ax3 = figure3.add_subplot(111)
ax3.scatter(df['Interest_Rate'].astype(float),df['Stock_Index_Price'].astype(float), color = 'r')
scatter3 = FigureCanvasTkAgg(figure3, root) 
scatter3.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH)
ax3.legend(['Stock_Index_Price']) 
ax3.set_xlabel('Interest Rate')
ax3.set_title('Interest Rate Vs. Stock Index Price')

#plot 2nd scatter 
figure4 = plt.Figure(figsize=(5,4), dpi=100)
ax4 = figure4.add_subplot(111)
ax4.scatter(df['Unemployment_Rate'].astype(float),df['Stock_Index_Price'].astype(float), color = 'g')
scatter4 = FigureCanvasTkAgg(figure4, root) 
scatter4.get_tk_widget().pack(side=tk.RIGHT, fill=tk.BOTH)
ax4.legend(['Stock_Index_Price']) 
ax4.set_xlabel('Unemployment_Rate')
ax4.set_title('Unemployment_Rate Vs. Stock Index Price')
'''
root.bind('<Escape>', toggle_fs)

root.mainloop()