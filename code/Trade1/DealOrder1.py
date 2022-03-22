from h5py import Dataset, Group, File
import h5py
import numpy as np
import pandas as pd
from logger import logger_config
from socket import *
from select import *
# 导入 socket 模块
import socket
import numpy as np
import pandas as pd
from socket import *
from select import *
from signal import *
import array, binascii, ctypes, struct, socket, asyncio, time, nest_asyncio
import asyncio
from functools import wraps, partial

# 发送hook
def create_logger(path):
    logger = logger_config(log_path='log.txt', logging_name='CreateOrderLog')
    IndexList = np.zeros((100, 4))
    # print(IndexList.shape)
    for stk_code in range(1, 11):
        logger = logger_config(log_path=path + "stk_" + str(stk_code) + '_.txt', logging_name='CreateOrderLog')


def WriteTraders(stk_code, Trader,path='/data/team-16'):
    filename = path + "stk_" + stk_code + '_.txt'
    with open(filename, 'a+') as f:  # 如果filename不存在会自动创建， 'w'表示写数据，写之前会清空文件中的原有数据！
        f.write(Trader + "\n")
    f.close()


def clean_stk_txt(path):
    for stk_code in range(1, 10):
        filename = path + "stk_" + str(stk_code) + '_.txt'
        file = open(filename, 'w').close()
    print("all the stk_txt clean")


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


def IsValidOrder(CurrentPrice, LastPrice):
    if CurrentPrice > (1 + .1) * LastPrice or CurrentPrice < (1 - .1) * LastPrice:
        return False
    else:
        return True


def FilteringInvalidOrdersCurrent(prev_price, price):
    price = float(price)
    prev_price = float(prev_price)
    if prev_price * 1.1 < price or prev_price * 0.9 > price:
        return True
    else:
        return False


def send_hooks(host, port, hook):
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (host, port)
    clientsocket.connect(server_address)
    hooks, index = [], 0
    for stk_code in range(1, 11):
        hooks += [' '.join([str(stk_code - 1)] + [str(i) for i in hook[stk_code - 1, j, :]]) for j in
                  range(hook.shape[1])]
        # print(len(hooks))
    res = max(hooks, key=len, default='')
    tail = 1024 // (max(len(res), 1) + 1)
    clientsocket.sendall("Hooks_begin".encode("utf-8"))
    while index <= len(hooks) - tail:
        data = 'H'.join(hooks[index:index + tail]).encode('utf-8')
        opreator = "H".encode('utf-8')
        if not data:
            break
        clientsocket.sendall(data)
        clientsocket.sendall(opreator)
        index += tail
        # print(index)
    data = 'H'.join(hooks[index:]).encode('utf-8')
    clientsocket.sendall(data)
    opreator = "Hooks_over".encode('utf-8')
    clientsocket.sendall(opreator)
    # server_data= clientsocket.recv(1024).decode()
    # print(server_data)
    # if server_data=="Hook_over":
    socket.close(1)
    return


def read_file(path='/data/100x10x10/'):
    key_words = ["order_id", "direction", "type", "price", "volume"]
    data = {}
    for key in key_words:
        f1 = WithdrawH5Dataset(key, path + key + str(1) + '.h5')
        data[key] =f1
        print(key, data[key].shape)
    key = 'prev_data'
    path1 = path + 'price' + str(1) + '.h5'
    with h5py.File(path1, "r") as f:
        for k in f.keys():
            if k == key:
                f1 = f[k]
    data['prev_price'] = f1
    # print(data['prev_price'].shape)
    key = "hook"
    hook = WithdrawH5Dataset(key, path + key + '.h5')
    print("hook shape is",hook.shape)
    return data, hook



def Initial_hooks(hook):
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

def add_one_VOrderList(stk_code,VOrderList,ord_id,data_all):
    coordinates=np.argwhere(data_all["order_id"]==int(ord_id)+1)
    enum=stk_code-1
    find=0
    for item in coordinates:
        if item[0]%10==enum:
            coordinate=item
            find=1
            break
    #print(data["order_id"].shape)
        #print(data["order_id"][coordinate])
    if not find:
        return VOrderList
    prev_price=data_all["prev_price"][coordinate[0]][coordinate[1]][coordinate[2]]
    price=data_all["price"][coordinate[0]][coordinate[1]][coordinate[2]]
    Invalid=FilteringInvalidOrdersCurrent(prev_price,price)
    if not Invalid:
        order_id=data_all["order_id"][coordinate[0]][coordinate[1]][coordinate[2]]
        direction=data_all["direction"][coordinate[0]][coordinate[1]][coordinate[2]]
        type=data_all["type"][coordinate[0]][coordinate[1]][coordinate[2]]
        volume=data_all["volume"][coordinate[0]][coordinate[1]][coordinate[2]]
        stk_code=coordinate[0]%10+1
#            OrderList[stk_code-1][order_id-1]=
            #ValidOrderList[stk_code-1].loc[order_id-1]=[str(direction),str(type),price,str(volume)]
        VOrderList[enum].append(' '.join([str(stk_code-1)+str(order_id),str(direction)+str(type)+str(price),str(volume)]))
    return VOrderList


def add_VOrderList(stk_code,VOrderList,data_all,HookList_lower_bound,HookList_upper_bound):
    enum=stk_code-1
    for ord_id in range(HookList_lower_bound[enum],HookList_upper_bound[enum]):
        VOrderList=add_one_VOrderList(stk_code,VOrderList,ord_id,data_all)
    return VOrderList


def check_hooks(stk_code,HookList,HookList_upper_bound):
    enum=int(stk_code)-1
    hook_idx=HookList[int(stk_code)-1][HookList[int(stk_code)-1].self_order_id==HookList_upper_bound[enum]+1].index.tolist()[0]
    #print(hook_idx)
    target_stk_code=HookList[int(stk_code)-1].loc[hook_idx].target_stk_code
    target_trade_idx=HookList[int(stk_code)-1].loc[hook_idx].target_trade_idx
    if target_stk_code>5:
        Done,Send_sp_hook=checkHooksFromTrader(target_stk_code,target_trade_idx,hook_idx)
        return Done,Send_sp_hook
    #print("target_stk_code",target_stk_code)
    #print("target_trade_idx",target_trade_idx)
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



def ConnectTrader(send_pack,Tradeip="10.216.68.192",Tradeport=31622):
    global Trader_list
    host = Tradeip
    server_address=(host,Tradeport)
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(server_address)
    signal(SIGPIPE, SIG_IGN)
    clientsocket.sendall(send_pack.encode("utf-8"))
    recv_data = clientsocket.recv(1024).decode()
    Done,Send_sp_hook=recv_data.split(" ")
    socket().close(1)
    return Done,Send_sp_hook


def checkHooksFromTrader(stk_code,order_id,hook_idx,Exchangeport2=31622):
    send_pack=str(stk_code)+" "+str(order_id)+" "+str(hook_idx)
    Done,Send_sp_hook=ConnectTrader(send_pack,Exchangeip2,Exchangeport2)
    return Done,Send_sp_hook
    
def getHookListUpperBound(stk_code,VOrderList,HookList,HookList_upper_bound,data_all):
    enum=stk_code-1
        #print("now the stk_code is",stk_code)
    old_hook_id=HookList_upper_bound[enum]+1
    HookList_lower_bound[enum]=old_hook_id
    Done=False
    Done,Send_sp_hook=check_hooks(stk_code,HookList,HookList_upper_bound)
        #print(Done,Send_sp_hook)
    while Done:
        if Send_sp_hook:
            add_one_VOrderList(stk_code,VOrderList,HookList_upper_bound[enum],data_all)
        old_hook_id=HookList_upper_bound[enum]+1
        #print("old_hook_id",old_hook_id)
         #print(HookList[enum][HookList[enum].self_order_id ==old_hook_id].index.tolist())
            #print(HookList[enum])
        new_hook_idx=HookList[enum][HookList[enum].self_order_id ==old_hook_id].index.tolist()[0] +1
        HookList_upper_bound[enum]= HookList[enum].loc[new_hook_idx].self_order_id-1
        HookList_lower_bound[enum]=old_hook_id+1
            #print(HookList_lower_bound,HookList_upper_bound)
        add_VOrderList(stk_code,VOrderList,data_all,HookList_lower_bound,HookList_upper_bound)
        Done,Send_sp_hook=check_hooks(stk_code,HookList,HookList_upper_bound)
    return VOrderList,HookList_lower_bound,HookList_upper_bound


def init_trader(type,filepath="/data/100x1000x1000/"):
    global VOrderList,HookList_lower_bound,HookList_upper_bound,data_all,hook,HookList
    if type==1:
        gap=0
    elif type==2:
        gap=5
    else:
        return print("Type Error! For Traders with label 1 or 2")
    data_all,hook=read_file(filepath)
    VOrderList=[[] for i in range(10)]
    header=["direction","type","price","volume"]
    HookList=Initial_hooks(hook)
    HookList_upper_bound=[]
    HookList_lower_bound=[0 for i in range(10)]
    for hooks in HookList:
        HookList_upper_bound.append(hooks["self_order_id"].min()-1)
    for stk_code in range(1,11):
        VOrderList=add_VOrderList(stk_code,VOrderList,data_all,HookList_lower_bound,HookList_upper_bound)
    VOrderList,HookList_lower_bound,HookList_upper_bound=getHookListUpperBound(stk_code,VOrderList,HookList,HookList_upper_bound,data_all)
    print()
    for i in range(10):
        print("初始的stk_code:",i+1,"有",len(VOrderList[i]),"个元素")









async def send_data(clientsocket,stk_code, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound):
    print(len(VOrderList[int(stk_code) - 1]))
    if len(VOrderList[int(stk_code) - 1]) > 0:
        index = 0
        res = max(VOrderList[int(stk_code) - 1], key=len, default='')
        tail = 1024 // (max(len(res), 1) + 1) - 1
        while index <= len(VOrderList[int(stk_code) - 1]) - tail:
            data = '$'.join(VOrderList[int(stk_code) - 1][index:index + tail])
            send_pack = "!$" + str(len(data)) + "$$" + data
            clientsocket.sendall(send_pack.encode('utf-8'))
            server_data = clientsocket.recv(1024).decode()
            index += tail
            print(len(send_pack))
            print(send_pack)
            receive_data(server_data)
        data = '$'.join(VOrderList[stk_code - 1][index:len(VOrderList[stk_code - 1])])
        send_pack = "!$" + str(len(data)) + "$$" + data
        clientsocket.sendall(send_pack.encode('utf-8'))
        print(len(send_pack))
        print(send_pack)
        server_data = clientsocket.recv(1024).decode()
        receive_data(server_data)
        # 关闭打开的文件
        VOrderList[stk_code - 1] = []
    else:
        server_data = clientsocket.recv(1024).decode()
        receive_data(server_data)
        time.sleep(3)
        await check_VOrderList(stk_code, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)
    print("over" + str(stk_code))
    return 1
    #


def receive_data(server_data):
    # server_data = clientsocket.recv(1024).decode()
    print('receive server data :', server_data)
    if "T" in server_data:
        if len(Trader_list) > 0:
            Trader_list[0] = Trader_list[0] + server_data
        else:
            Trader_list.append(server_data)
        bg = Trader_list[0].index("!T")
        ed = Trader_list[0].index("TT")
        if bg + 2 < ed:
            length = int(Trader_list[0][bg + 2:ed])
        else:
            length = 0
        Trader_data = Trader_list[0][:bg] + Trader_list[0][ed + 2:ed + 2 + length]
        Traders = Trader_data.split("T")
        print(Traders)
        for Trader in Traders:
            if len(Trader) > 0:
                stk_code = Trader[0]
                WriteTraders(stk_code, Trader[1:],Tradersavepath)
        Trader_list[0] = Trader_list[0][ed + 2 + length:]


async def check_VOrderList(stk_code, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound):
    VOrderList, HookList_lower_bound, HookList_upper_bound = getHookListUpperBound(stk_code, VOrderList, HookList,
                                                                                   HookList_upper_bound, data_all)
    print(HookList_lower_bound[stk_code - 1], HookList_upper_bound[stk_code - 1])
    VOrderList = add_VOrderList(stk_code, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)
    if len(VOrderList[stk_code - 1]) > 0:
        await send_data(clientsocket, stk_code, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)


async def main_work(VOrderList,host="10.216.68.190",port=31611):
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_address = (host, port)
    clientsocket.connect(server_address)
    signal(SIGPIPE, SIG_IGN)
    send_1 = send_data(clientsocket, 1, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)
    send_2 = send_data(clientsocket, 2, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)
    send_3 = send_data(clientsocket, 3, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)
    send_4 = send_data(clientsocket, 4, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)
    send_5 = send_data(clientsocket, 5, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)
    send_6 = send_data(clientsocket, 6, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)
    send_7 = send_data(clientsocket, 7, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)
    send_8 = send_data(clientsocket, 8, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)
    send_9 = send_data(clientsocket, 9, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)
    send_10 = send_data(clientsocket, 10, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)

    tasks = [
        asyncio.ensure_future(send_1),
        asyncio.ensure_future(send_2),
        asyncio.ensure_future(send_3),
        asyncio.ensure_future(send_4),
        asyncio.ensure_future(send_5),
        asyncio.ensure_future(send_6),
        asyncio.ensure_future(send_7),
        asyncio.ensure_future(send_8),
        asyncio.ensure_future(send_9),
        asyncio.ensure_future(send_10),
    ]
    dones, pendings = await asyncio.wait(tasks)

    for task in dones:
        print("The task's result is : {}".format(task.result()))

def main():
    import configparser
    global Trader_list,Tradeport2,Tradeip2,Exchangeip1,Exchangeport1,Exchangeip2,Exchangeport2,Tradersavepath
    cf = configparser.ConfigParser()
    cf.read("/data/team-16/config.ini", encoding='GB18030')  # 读取配置文件，如果写文件的绝对路径，就可以不用os模块
    secs = cf.sections()  # 获取文件中所有的section(一个配置文件中可以有多个配置，如数据库相关的配置，邮箱相关的配置，
    Tradeip1=cf.get("ip-config","Tradeip1")
    Tradeport1=int(cf.get("ip-config","Tradeport1"))
    Tradeip2=cf.get("ip-config","Tradeip2")
    Tradeport2=int(cf.get("ip-config","Tradeport2"))
    Exchangeip1=cf.get("ip-config","Exchangeip1")
    Exchangeport1=int(cf.get("ip-config","Exchangeport1"))
    Exchangeip2=cf.get("ip-config","Exchangeip2")
    Exchangeport2=int(cf.get("ip-config","Exchangeport2"))
    filepath=cf.get("path","filepath")
    Tradersavepath=cf.get("path","Tradersavepath")
    print(secs)
    print('当前主机名称为 : ' + socket.gethostname())
    print('当前主机的IP为: ' + socket.gethostbyname(socket.gethostname()))
    #asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    loop = asyncio.get_event_loop()
    nest_asyncio.apply()
    Trader_list = []
    clientsocket_1 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket_2 = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clean_stk_txt(Tradersavepath)
    print("开始和Trade2通信,ip:",Tradeip2,"记得打开Trade2的端口,现在的端口是",Tradeport2)
    init_trader(1,filepath)
    #host = input("请输入Exchange1上的ip地址:")
    #Exchangeport1 = int(input("请输入Exchange1上的通信端口:"))
    print("开始和Exchange1通信,ip:",Exchangeip1,"记得打开Exchange1的端口,现在的端口是",Exchangeport1)
    #host = input("请输入Exchange2上的ip地址:")
    #Exchangeport2 = int(input("请输入Exchange2上的通信端口:"))
    print("开始和Exchange1通信,ip:",Exchangeip2,"记得打开Exchange2的端口,现在的端口是",Exchangeport2)
    #Tradeip=input("请输入Trade2上的ip地址:")
    #Tradeport=int(input("请输入Trade2上的通信端口:"))
    loop.run_until_complete(main_work(VOrderList,Exchangeip1, Exchangeport1))
    # send_data(host,port,VOrderList,stk_code)


print(data_all["order_id"].shape)


def Vfind(arr, pos_min, pos_max):
    # pos_min = arr>=min
    # pos_max =  arr<max
    pos_rst = pos_min & pos_max
    return np.where(pos_rst == True)




def FilteringInvalidOrdersCurrent(prev_price, price):
    price = float(price)
    prev_price = float(prev_price)
    res = price[prev_price * 1.1 >= price and prev_price * 0.9 <= price]
    print(res.shape)
    return res

def add_VOrderList(stk_code, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound):
    pos = Vfind(data_all["order_id"], HookList_lower_bound[enum], HookList_upper_bound[enum]).shape
    print(pos.shape)
    enum = stk_code - 1
    prev_price = data_all["prev_price"][pos]
    price = data_all["price"][pos]
    
    Invalid = FilteringInvalidOrdersCurrent(prev_price, price)
    if not Invalid:
        order_id = data_all["order_id"][coordinate[0]][coordinate[1]][coordinate[2]]
        direction = data_all["direction"][coordinate[0]][coordinate[1]][coordinate[2]]
        type = data_all["type"][coordinate[0]][coordinate[1]][coordinate[2]]
        volume = data_all["volume"][coordinate[0]][coordinate[1]][coordinate[2]]
        stk_code = coordinate[0] % 10 + 1
        #            OrderList[stk_code-1][order_id-1]=
        # ValidOrderList[stk_code-1].loc[order_id-1]=[str(direction),str(type),price,str(volume)]
        VOrderList[enum].append(
            ' '.join([str(stk_code - 1) + str(order_id), str(direction) + str(type) + str(price), str(volume)]))
    return VOrderList
    return VOrderList


for stk_code in range(1, 11):
    VOrderList = add_VOrderList(stk_code, VOrderList, data_all, HookList_lower_bound, HookList_upper_bound)

main()
