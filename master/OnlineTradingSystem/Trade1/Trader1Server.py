from h5py import Dataset, Group, File
import h5py
import numpy as np
import pandas as pd
from select import *
# 导入 socket 模块
import socket
import numpy as np
import pandas as pd
from socket import *
from select import *
from signal import *
import array, binascii, ctypes, struct, asyncio, time, nest_asyncio
import asyncio
from functools import wraps, partial


def clean_stk_txt(path):
    for stk_code in range(1, 10):
        filename = path + "stk_" + str(stk_code) + '_.txt'
        file = open(filename, 'w').close()
    print("all the stk_txt clean")

def read_hook(path='/data/100x10x10/'):
    key = "hook"
    hook = WithdrawH5Dataset(key, path + key + '.h5')
    print(hook.shape)
    return hook

def WithdrawH5Dataset(key, path):
    with h5py.File(path, "r") as f:
        for k in f.keys():
            if isinstance(f[k], Dataset):
                Datasets = f[k][:]
                rkey = f[k].name
                if k == key:
                    # print('key is now return',key)
                    return Datasets
            else:
                print(f[k].name)
        # print('key is not return',key,'instead we find',rkey)
        return Datasets

def Initial_hooks():
    hook=read_hook()
    key_words=["self_order_id", "target_stk_code", "target_trade_idx", "arg"]
    HookList=[pd.DataFrame([],columns=key_words) for i in range(10)]
    index=0
    for i in range(10):
        hook_item=hook[i]
        for j in range(hook.shape[1]):
            HookList[index].loc[hook_item[j][0]]=hook_item[j]
        HookList[index] = HookList[index].sort_values(by = ["self_order_id"],ascending = True)
        HookList[index] = HookList[index].reset_index(drop=True)
        index+=1
    return HookList

def check_hooks(stk_code,HookList,HookList_upper_bound):
    enum=int(stk_code)-1
    hook_idx=HookList[int(stk_code)-1][HookList[int(stk_code)-1].self_order_id==HookList_upper_bound[enum]+1].index.tolist()[0]
    target_stk_code=HookList[int(stk_code)-1].loc[hook_idx].target_stk_code
    target_trade_idx=HookList[int(stk_code)-1].loc[hook_idx].target_trade_idx
    path="stk_"+str(target_stk_code-1)+"_.txt"
    key_words=["bid_id", "ask_id", "price", "volume"]
    stk_trades = pd.read_csv(path,names=key_words,header=None,sep=' ') # 获hook数据
    #print(len(stk_trades))
    #print(HookList_upper_bound[int(stk_code)-1]+1)
    if int(target_trade_idx)-1<=len(stk_trades):
        stk_trades.index=range(1,len(stk_trades)+1)
        #print("HookList[int(stk_code)-1].iloc[HookList_upper_bound[int(stk_code)-1]].arg",HookList[int(stk_code)-1].loc[hook_idx].arg)
        #print("stk_trades[int(target_trade_idx)-1].volume",stk_trades.loc[int(target_trade_idx)-1].volume)
        if stk_trades.loc[int(target_trade_idx)-1].volume<=HookList[int(stk_code)-1].loc[hook_idx].arg:
            return True,True
        else:
            return True,False
    else:
        return False,False

def hookDataCheck(data):
    print(data)
    stk_code,target_trade_idx,hook_idx=data.split(" ")
    path="stk_"+str(stk_code)+"_.txt"
    key_words=["bid_id", "ask_id", "price", "volume"]
    stk_trades = pd.read_csv(path,names=key_words,header=None,sep=' ') # 获hook数据
    if int(target_trade_idx)-1<=len(stk_trades):
        stk_trades.index=range(1,len(stk_trades)+1)
        if stk_trades.loc[int(target_trade_idx)-1].volume<=HookList[int(stk_code)-1].loc[hook_idx].arg:
            return True,True
        else:
            return True,False
    else:
        return False,False


def Trader():
    host = gethostname()  # ip-10-216-68-189#"10.216.68.190" #socket.gethostname()
    print(host)
    port = 31622
    s = socket()
    s.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)  # 创建套接字作为需要被关注的IO
    s.bind((host, port))
    s.listen(10)
    ep = epoll()  # 创建epoll()对象
    fdmap = {s.fileno(): s}  # 创建查找字典，{文件描述符fileno:io_obj},将s监听套接字维护进入系统的监控字典中
    ep.register(s, EPOLLIN | EPOLLERR)  # 监控s,读IO或异常IO
# 循环监控IO的发生(不断会有请求传进来）
    data_list = []
    hook_list = []
    Traders = []
    signal(SIGPIPE, SIG_IGN)
    while True:
        events = ep.poll()
        print(events)
        for fd, event in events:
            if fd == s.fileno():
                c, addr = fdmap[fd].accept()
                print('Connect from', addr)
                ep.register(c, EPOLLIN)
                fdmap[c.fileno()] = c
            elif event & EPOLLIN:
                if "T" in data:
                    server_data = clientsocket.recv(1024).decode()
                    receive_data(server_data)
                    fdmap[fd].send("yes".encode("utf-8"))
                else:
                    data = fdmap[fd].recv(1024).decode()
                    Done,Send=hookDataCheck(data, hook_list)
                    print(Done,Send)
                    Send_pack=str(Done)+" "+str(Send)
                    fdmap[fd].send(Send_pack.encode("utf-8"))
                if not data or data == "over":
                    print("客户端退出")
                    ep.unregister(fd)
                    fdmap[fd].close()
                    del fdmap[fd]
                    continue


global HookListCheck,HookList
HookList=Initial_hooks()
path="/data/team-16/"
clean_stk_txt(path)
Trader()
    