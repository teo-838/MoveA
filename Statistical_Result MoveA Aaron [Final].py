# -*- coding: utf-8 -*-
"""
Created on Sat Sep 25 23:18:04 2021

@author: user
"""
#Import library
import itertools
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import csv
import pandas as pd 
import seaborn as sns
from statannot import add_stat_annotation
from scipy.stats import iqr
import scipy.stats as stat
from statistics import *
       
#Read and write csv file
def read_csv(path):
  rows = []
  with open(path) as csvfile:
    file_reader = csv.reader(csvfile)
    for row in file_reader:
      rows.append(list(row))
  return rows
def writecsv(path,r):
    f = open(path, "w",newline='')
    writer = csv.writer(f)
    writer.writerows(r)
    f.close()

#Retrieve analysis file (File 3/ File 4) and convert the data into correct format (Speed data: system(), Displacement data: system_displacement())
def manual(lst):
    Treatment=[]
    for field in lst:
        data=read_csv(field+'.csv')#retrieve
        Track=list(map(lambda x: float(x[13]),list(filter(lambda x: x[13] != 'N/A' and x[13] != 'Track Velocity (µm/sec)' ,data[1:]))))
        Treatment += Track
    return Treatment

def manual_displacement(lst):
    Treatment=[]
    for field in lst:
        data=read_csv(field+'.csv')#retrieve
        Track=list(map(lambda x: float(x[15]),list(filter(lambda x: x[15] != 'N/A' and x[15] != 'Displacement (µm)' ,data[1:]))))
        Treatment += Track
    return Treatment

def system(lst):
    Treatment=[]
    for field in lst:
        print(field)
        data=read_csv(field+'.csv')#retrieve
        Track=list(map(lambda x: float(x[2]),data[1:len(data)-2]))
        Treatment += Track
    return Treatment


def system_displacement(lst):
    Treatment=[]
    for field in lst:
        data=read_csv(field+'.csv')#retrieve
        Track=list(map(lambda x: float(x[1]),data[1:len(data)-2]))
        Treatment += Track
    return Treatment

#Self-defined Unequal variance t-test calculation function
def unequal_var_t(data1, data2, mu_interest, conf_lvl):
    mean1, mean2, n1, n2 = np.mean(data1), np.mean(data2), len(data1), len(data2)
    var1, var2 = 0, 0
    for i in range(n1):
        var1 += (data1[i]-mean1)**2
    for i in range(n2):
        var2 += (data2[i]-mean2)**2
    var1, var2 = var1/(n1-1), var2/(n2-1)
    a, b = var1/n1, var2/n2
    c, d = (a**2)/(n1-1), (b**2)/(n2-1)
    df = round(((a+b)**2)/ (c+d))
    t = ((mean1-mean2)-mu_interest)/np.sqrt(a+b)
    lvl = 1-(conf_lvl/2)
    tcrit = stats.t.ppf(lvl, df)
    p = (1 - stats.t.cdf(abs(t),df))*2
    return [mean1, mean2, var1, var2, df, t, tcrit, p,n1,n2]

def fig8():
    Treatment_color=sns.color_palette('tab10')[0]#Color for treatment boxplot
    Control_color=sns.color_palette('tab10')[1]#Color for control boxplot
    O65_color=sns.color_palette('tab10')[2]#Color for O65 boxplot
    #Plot statistical comparison between different treatment and control pairs
    fig8,ax = plt.subplots(figsize=(1,10))
    sns.set(style="whitegrid")
    x = "Group"
    y = "Speed (µm/s)"
    order = Grplabel
    if boolen:
        my_pal = {"S4":Treatment_color, "O65": O65_color, "Control": Control_color}
        sns.boxplot(data=df, x=x, y=y, order=order,palette=my_pal) #Boxplot using DataFrame
    else: 
        sns.boxplot(data=df, x=x, y=y, order=order) #Boxplot using DataFrame
    add_stat_annotation(ax, data=df, x=x, y=y, order=order,
                        box_pairs=permutation,
                        test='t-test_welch', text_format='star', loc='inside', verbose=2)#Annotate the Welch's t-test results (Unequal variance t-test) for each pair of comparison
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    figManager = plt.get_current_fig_manager()
    figManager.window.showMaximized()
    ax.yaxis.set_ticks(np.arange(0, t_lim[1], 0.5))
    plt.ylim(t_lim[0],t_lim[1]) 
    ax.grid(False)
    plt.show()
    fig8.savefig(TaskNo+' t-test_welch.tiff')#Fig8

def fig9():
    #Plot speed of each treatment and control groups
    Treatment_color=sns.color_palette('tab10')[0]#Color for treatment spectrum
    Control_color=sns.color_palette('tab10')[1]#Color for control spectrum
    Treatment_line_color=sns.color_palette('pastel')[0]#Color for treatment median and average vertical line
    Control_line_color=sns.color_palette('pastel')[1]#Color for control median and average vertical line
    O65_color=sns.color_palette('tab10')[2]#Color for O65 spectrum
    O65_line_color=sns.color_palette('pastel')[2]#Color for O65 median and average vertical line
    if boolen:
        Color=[Treatment_color,O65_color,Control_color]
        Colorv=[Treatment_line_color,O65_line_color,Control_line_color]
    else: 
        Color=[Treatment_color,Control_color]
        Colorv=[Treatment_line_color,Control_line_color]
    Main=[Group_speed]
    Labellst=Grplabel
    Data=['System']
    for k in range(len(Data)):
        fig9=plt.figure(figsize=(FigSizeW, FigSizeH))
        for jj in range(len(Labellst)):
            Label=Data[k]+' '+Labellst[jj]
            mean=np.mean(Main[k][jj])
            median=np.median(Main[k][jj])
            #ax=sns.distplot(Main[k][jj],label=Label,axlabel='Speed (µm/s)',color=Color[jj],hist=False)
            ax=sns.kdeplot(Main[k][jj],label=Label,color=Color[jj])
            ax.axvline(mean, color=Colorv[jj], linestyle='--', label="Mean",lw=4)
            ax.axvline(median, color=Colorv[jj], linestyle=':', label="Median",lw=4)
            ax.set(xlabel='Speed (µm/s)', ylabel='Density')
            ax.xaxis.set_ticks(np.arange(0, x_lim[1], 0.1))
            ax.yaxis.set_ticks(np.arange(0, x_lim[3], 1))
        #plt.legend();
        plt.xlim(x_lim[0],x_lim[1]) 
        plt.ylim(x_lim[2],x_lim[3]) 
        ax.grid(False)
        #plt.title('Speed Distribution')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        fig9.savefig(TaskNo+' Speed.tiff') #Fig9

def fig10():
    #Plot speed of each treatment and control groups
    Treatment_color=sns.color_palette('tab10')[0]#Color for treatment spectrum
    Control_color=sns.color_palette('tab10')[1]#Color for control spectrum
    Treatment_line_color=sns.color_palette('pastel')[0]#Color for treatment median and average vertical line
    Control_line_color=sns.color_palette('pastel')[1]#Color for control median and average vertical line
    O65_color=sns.color_palette('tab10')[2]#Color for O65 spectrum
    O65_line_color=sns.color_palette('pastel')[2]#Color for O65 median and average vertical line
    
    if boolen:
        Color=[Treatment_color,O65_color,Control_color]
        Colorv=[Treatment_line_color,O65_line_color,Control_line_color]
    else: 
        Color=[Treatment_color,Control_color]
        Colorv=[Treatment_line_color,Control_line_color]
        
    Main=[Group_displacement]
    Labellst=Grplabel
    Data=['System']
    for k in range(len(Data)):
        fig10=plt.figure(figsize=(FigSizeW, FigSizeH))
        for jj in range(len(Labellst)):
                Label=Data[k]+' '+Labellst[jj]
                mean=np.mean(Main[k][jj])
                median=np.median(Main[k][jj])
                df= pd.DataFrame(np.array(Main[k][jj]))
                ax=sns.kdeplot(Main[k][jj],label=Label,color=Color[jj])
                ax.axvline(mean, color=Colorv[jj], linestyle='--', label="Mean",lw=4)
                ax.axvline(median, color=Colorv[jj], linestyle=':', label="Median",lw=4)
                ax.set(xlabel='Displacement (µm)', ylabel='Density')
                ax.xaxis.set_ticks(np.arange(0, d_lim[1], 10))
                ax.yaxis.set_ticks(np.arange(0, d_lim[3], 0.005))
        #plt.legend();
        plt.xlim(d_lim[0],d_lim[1]) 
        plt.ylim(d_lim[2],d_lim[3]) 
        ax.grid(False)
        #plt.title('Movement Distribution')
        figManager = plt.get_current_fig_manager()
        figManager.window.showMaximized()
        fig10.savefig(TaskNo+' Displacement.tiff') #Fig10
    
def statresult():
    #Calculate the Spectrum Peak (Using gaussian_kde to calculate)
    #Speed
    Peak=[]
    for i in Group_speed:
        density=stats.gaussian_kde(i)
        xs = i
        ys = density(xs)
        index = np.argmax(ys)
        max_y = ys[index]
        max_x = xs[index]
        Peak.append([max_y,max_x])
    #Displacement
    Peak_displacement=[]
    for i in Group_displacement:
        density=stats.gaussian_kde(i)
        xs = i
        ys = density(xs)
        index = np.argmax(ys)
        max_y = ys[index]
        max_x = xs[index]
        Peak_displacement.append([max_y,max_x])
    
    #Generate statistical analysis result file (with statistic details)
    #A. Compute the Mean, Median, Variance, Stdev, Number of data, Spectrum Peak, 25 percentile, 75 percentile, interquartile range for each treatment and control groups
    #A.Speed
    res=[['Statistic Detail']]
    res.append(['Info']+Grplabel)
    functions=[mean,median,variance,stdev,len]
    functionsname=['Mean (speed µm/s)','Median (speed µm/s)','Variance','Stdev','Number of data']
    for i in range(len(functions)): #Compute the Mean, Median, Variance, Stdev, Number of data for each treatment and control groups
        res.append([functionsname[i]]+list(map(lambda x: functions[i](x),Group_speed)))
    
    res.append(['Spectrum Peak (speed µm/s)']+list(map(lambda x: x[1],Peak)))
    res.append(['Spectrum Peak (density)']+list(map(lambda x: x[0],Peak)))
    res.append(['25 percentile (speed µm/s)']+list(map(lambda x: np.percentile(x,25),Group_speed)))
    res.append(['75 percentile (speed µm/s)']+list(map(lambda x: np.percentile(x,75),Group_speed)))
    res.append(['IQR (speed µm/s)']+list(map(lambda x: iqr(x,axis=0),Group_speed)))
    
    #B. Calculate Welch's t-test statistic details (Mean, Variance, Degree of Freedom, T_statistic, T-critical, P-value, Number of data)
    res.append([])
    res.append(['Statistics Test (Welch t-test)'])
    
    keys=list(itertools.combinations(list(dict_grp.keys()), 2))
    res.append(['Comparison','TreatmentMean (speed µm/s)','ControlMean (speed µm/s)','TreatmentVariance','ControlVariance','Degree of Freedom','T_statistic','T_critical','P-Value','Treatment Number','Control Number']) 
    for u,v in keys:
        statistic, pvalue = stat.ttest_ind(dict_grp[u],dict_grp[v],equal_var=False)
        statistic_res = unequal_var_t(dict_grp[u],dict_grp[v], 0, 0.05)
        print(round(statistic_res[-5],4)==round(statistic,4))
        statistic_res[-3] = pvalue
        statistic_res[-5] = statistic
        res.append([u+' (Treatment) vs. '+v+' (Control)']+statistic_res)
    
    #A.Displacement
    res.append([])
    res.append(['Displacement'])
    res.append(['Info']+Grplabel)
    functions=[mean,median,variance,stdev,len]
    functionsname=['Mean (displacement µm)','Median (displacement µm)','Variance','Stdev','Number of data']
    for i in range(len(functions)):
        res.append([functionsname[i]]+list(map(lambda x: functions[i](x),Group_displacement)))
        
    res.append(['Spectrum Peak (displacement µm)']+list(map(lambda x: x[1],Peak_displacement)))
    res.append(['Spectrum Peak (density)']+list(map(lambda x: x[0],Peak_displacement)))
    res.append(['25 percentile (displacement µm)']+list(map(lambda x: np.percentile(x,25),Group_displacement)))
    res.append(['75 percentile (displacement µm)']+list(map(lambda x: np.percentile(x,75),Group_displacement)))
    res.append(['IQR (displacement µm)']+list(map(lambda x: iqr(x,axis=0),Group_displacement)))
    
    
    #Store statistical analysis result (both A. and B.) into .csv file
    writecsv(TaskNo+' StatDetail.csv',res) #File5
    
### Execute Genetic Dataset

###ONLY NEED CHANGE###
TaskNo='18_4_22 Genetic Result'
Grplabel=['SKD1','Control']
Lst_of_Fields=[['SKD1_1 result','SKD1_2 result','SKD1_4 result','SKD1_6 result','SKD1_7 result','SKD1_8 result'],['Control_4 result','Control_5 result','Control_6 result','Control_7 result']]

boolen = False #True for biological dataset, false for else, #True for biological dataset, false for else, boolen = True is 3 comparison, boolen = False is 2 comparison
###TILL HERE###

#Outputsetting
FigSizeW = 10 #In inches
FigSizeH = 10 #In inches
x_lim = [0,1.2,0,9.0]
d_lim = [0,250,0,0.05]
t_lim = [0,2.5]

#Creating all possible comparison to perform unequal variance t-test
permutation=list(itertools.combinations(Grplabel, 2))
p_new=[]
for i in range(len(permutation)):
    p_new.append(permutation[i])  
permutation=p_new

#Generate a list include the speed and displacement from all groups (Treatments and controls)
Group_speed=[]
Group_displacement=[]
for group in Lst_of_Fields:
    Group_speed.append(system(group)) #Retrieve the data from every treatment and control groups and append into the main list (Group_speed for speed, Group_displacement for displacement)
    Group_displacement.append(system_displacement(group))

#Prepare for statistic csv file: dict_grp
#Prepare for generate Pandas Dataframe: Lstofgroup and Valueofgroup
dict_grp={}
Lstofgroup=[]
Valueofgroup=[]
for i in range(len(Group_speed)):
    Lstofgroup += [Grplabel[i]]*len(Group_speed[i]) #The list of each track label (of treatments and controls group),
    Valueofgroup.extend(Group_speed[i]) #The average speed of each track in the Lstofgroup,
    dict_grp[Grplabel[i]]=Group_speed[i] #dict_grp: {<name of treatment/control group> : list of the tracks' average speed }

#Generate Pandas DataFrame using Lstofgroup and Valueofgroup.   
df = pd.DataFrame(list(zip(Lstofgroup,Valueofgroup)), columns =['Group','Speed (µm/s)']) #Generate a dataframe with two column: Group and Speed
print(df)#For coder visualize the DataFrame ONLY

fig8()
fig9()
fig10()
statresult()

###############################YOU DON't NEED

'''
### Execute Chemical SIT Dataset
TaskNo='18_4_2022 Chemical Sitaxentan Result'
Grplabel=['Sitaxentan','Control'] #Note: len(Grplabel)=len(Lst_of_Fields)
Lst_of_Fields=[['Sitaxentan1a result','Sitaxentan1b result','Sitaxentan1c result','Sitaxentan1d result','Sitaxentan2b result','Sitaxentan2c result','Sitaxentan2d result'],['SitaxentanControl1a result','SitaxentanControl1b result','SitaxentanControl1c result','SitaxentanControl1d result']]

boolen = False #True for biological dataset, false for else

#Outputsetting
FigSizeW = 10 #In inches
FigSizeH = 10 #In inches
x_lim = [0,1.2,0,9.0]
d_lim = [0,250,0,0.05]
t_lim = [0,2.5]

#Creating all possible comparison to perform unequal variance t-test
permutation=list(itertools.combinations(Grplabel, 2))
p_new=[]
for i in range(len(permutation)):
    p_new.append(permutation[i])  
permutation=p_new

#Generate a list include the speed and displacement from all groups (Treatments and controls)
Group_speed=[]
Group_displacement=[]
for group in Lst_of_Fields:
    Group_speed.append(system(group)) #Retrieve the data from every treatment and control groups and append into the main list (Group_speed for speed, Group_displacement for displacement)
    Group_displacement.append(system_displacement(group))

#Prepare for statistic csv file: dict_grp
#Prepare for generate Pandas Dataframe: Lstofgroup and Valueofgroup
dict_grp={}
Lstofgroup=[]
Valueofgroup=[]
for i in range(len(Group_speed)):
    Lstofgroup += [Grplabel[i]]*len(Group_speed[i]) #The list of each track label (of treatments and controls group),
    Valueofgroup.extend(Group_speed[i]) #The average speed of each track in the Lstofgroup,
    dict_grp[Grplabel[i]]=Group_speed[i] #dict_grp: {<name of treatment/control group> : list of the tracks' average speed }

#Generate Pandas DataFrame using Lstofgroup and Valueofgroup.   
df = pd.DataFrame(list(zip(Lstofgroup,Valueofgroup)), columns =['Group','Speed (µm/s)']) #Generate a dataframe with two column: Group and Speed
print(df)#For coder visualize the DataFrame ONLY

fig8()
fig9()
fig10()
statresult()



### Execute Chemical TGZ Dataset

TaskNo='18_4_2022 Chemical Troglitazone Result' 
Grplabel=['Troglitazone','Control'] #Note: len(Grplabel)=len(Systemlst)
Lst_of_Fields=[['Trog1a result','Trog1b result','Trog1c result'],['TrogControl1a result','TrogControl1b result','TrogControl1c result','TrogControl1d result','TrogControl1e result','TrogControl1f result']]

boolen = False #True for biological dataset, false for else

#Outputsetting
FigSizeW = 10 #In inches
FigSizeH = 10 #In inches
x_lim = [0,1.2,0,9.0]
d_lim = [0,250,0,0.05]
t_lim = [0,2.5]

#Creating all possible comparison to perform unequal variance t-test
permutation=list(itertools.combinations(Grplabel, 2))
p_new=[]
for i in range(len(permutation)):
    p_new.append(permutation[i])  
permutation=p_new

#Generate a list include the speed and displacement from all groups (Treatments and controls)
Group_speed=[]
Group_displacement=[]
for group in Lst_of_Fields:
    Group_speed.append(system(group)) #Retrieve the data from every treatment and control groups and append into the main list (Group_speed for speed, Group_displacement for displacement)
    Group_displacement.append(system_displacement(group))

#Prepare for statistic csv file: dict_grp
#Prepare for generate Pandas Dataframe: Lstofgroup and Valueofgroup
dict_grp={}
Lstofgroup=[]
Valueofgroup=[]
for i in range(len(Group_speed)):
    Lstofgroup += [Grplabel[i]]*len(Group_speed[i]) #The list of each track label (of treatments and controls group),
    Valueofgroup.extend(Group_speed[i]) #The average speed of each track in the Lstofgroup,
    dict_grp[Grplabel[i]]=Group_speed[i] #dict_grp: {<name of treatment/control group> : list of the tracks' average speed }

#Generate Pandas DataFrame using Lstofgroup and Valueofgroup.   
df = pd.DataFrame(list(zip(Lstofgroup,Valueofgroup)), columns =['Group','Speed (µm/s)']) #Generate a dataframe with two column: Group and Speed
print(df)#For coder visualize the DataFrame ONLY

fig8()
fig9()
fig10()
statresult()




### Execute Biological Dataset

TaskNo='18_4_2022 Biological Result'
Grplabel=['S4','O65','Control'] #Note: len(Grplabel)=len(Systemlst)
Lst_of_Fields=[['S4_2 result','S4 result'],['O65_2 result','O65 result'],['Control_2 result','Control_3 result']]

boolen = True #True for biological dataset, false for else, boolen = True is 3 comparison, boolen = False is 2 comparison

#Outputsetting
FigSizeW = 10 #In inches
FigSizeH = 10 #In inches
x_lim = [0,1.2,0,9.0]
d_lim = [0,250,0,0.05]
t_lim = [0,2.5]

#Creating all possible comparison to perform unequal variance t-test
permutation=list(itertools.combinations(Grplabel, 2))
p_new=[]
for i in range(len(permutation)):
    p_new.append(permutation[i])  
permutation=p_new

#Generate a list include the speed and displacement from all groups (Treatments and controls)
Group_speed=[]
Group_displacement=[]
for group in Lst_of_Fields:
    Group_speed.append(system(group)) #Retrieve the data from every treatment and control groups and append into the main list (Group_speed for speed, Group_displacement for displacement)
    Group_displacement.append(system_displacement(group))

#Prepare for statistic csv file: dict_grp
#Prepare for generate Pandas Dataframe: Lstofgroup and Valueofgroup
dict_grp={}
Lstofgroup=[]
Valueofgroup=[]
for i in range(len(Group_speed)):
    Lstofgroup += [Grplabel[i]]*len(Group_speed[i]) #The list of each track label (of treatments and controls group),
    Valueofgroup.extend(Group_speed[i]) #The average speed of each track in the Lstofgroup,
    dict_grp[Grplabel[i]]=Group_speed[i] #dict_grp: {<name of treatment/control group> : list of the tracks' average speed }

#Generate Pandas DataFrame using Lstofgroup and Valueofgroup.   
df = pd.DataFrame(list(zip(Lstofgroup,Valueofgroup)), columns =['Group','Speed (µm/s)']) #Generate a dataframe with two column: Group and Speed
print(df)#For coder visualize the DataFrame ONLY

fig8()
fig9()
fig10()
statresult()

