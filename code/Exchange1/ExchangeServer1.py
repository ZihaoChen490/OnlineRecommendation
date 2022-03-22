from socket import *
from select import *
import pandas as pd
def InitialStockList():
    key_words=["order_id","direction","type","price","volume"]
    StockList=[pd.DataFrame([],columns=key_words) for i in range(10)]
    return StockList

def Renewable(StockList,data):
    print(len(data))
    print(data)
    if len(data)==0:
        return StockList
    if isinstance(data,str):
        order_list=data.split('$')
        for order in order_list:
            if len(order)>4:
                item_list=order.split(' ')
                print(item_list)
                stk_code=int(item_list[0][0])+1
                order_id=int(item_list[0][1:])
                if item_list[1][0]=='-':
                    direction=item_list[1][:2]
                    type=item_list[1][2]
                    price=float(item_list[1][3:])
                    volume=int(item_list[2])
                else:
                    direction=item_list[1][0]
                    type=item_list[1][1]
                    price=float(item_list[1][2:])
                    volume=int(item_list[2])
                StockList[stk_code-1].loc[order_id]=[order_id,direction,type,price,volume] 
    return StockList


def deal(BuyOrder,SellOrder,TransactionAuction,StockList,stk_code,Traders):
    bid_id_list=BuyOrder[BuyOrder["price"]>=TransactionAuction]
    ask_id_list=SellOrder[SellOrder["price"]<=TransactionAuction]
    Trader=[]
    while len(bid_id_list)>0 and len(ask_id_list)>0:
        if bid_id_list.iloc[0]["volume"]<ask_id_list.iloc[0]["volume"]:
            Trader=" ".join([str(stk_code-1),str(bid_id_list.iloc[0]["order_id"]),str(ask_id_list.iloc[0]["order_id"]),str(TransactionAuction),str(bid_id_list.iloc[0]["volume"])])
            SellOrder.loc[0,"volume"]=ask_id_list.iloc[0]["volume"]-bid_id_list.iloc[0]["volume"] 
            BuyOrder.drop(index=bid_id_list.iloc[0]["order_id"],inplace = True)
            StockList[stk_code-1].drop(index=bid_id_list.iloc[0]["order_id"],inplace = True)
            StockList[stk_code-1].loc[StockList[stk_code-1].order_id==ask_id_list.iloc[0]["order_id"],"volume"]=ask_id_list.iloc[0]["volume"]-bid_id_list.iloc[0]["volume"]
        elif bid_id_list.iloc[0]["volume"]>ask_id_list.iloc[0]["volume"]:
            Trader=" ".join([str(stk_code-1),str(bid_id_list.iloc[0]["order_id"]),str(ask_id_list.iloc[0]["order_id"]),str(TransactionAuction),str(ask_id_list.iloc[0]["volume"])])
            BuyOrder.loc[0,"volume"]=bid_id_list.iloc[0]["volume"]-ask_id_list.iloc[0]["volume"] 
            SellOrder.drop(index=ask_id_list.iloc[0]["order_id"],inplace = True)
            #print(SellOrder[ask_id_list.iloc[0]["order_id"]])
            StockList[stk_code-1].drop(index=ask_id_list.iloc[0]["order_id"],inplace = True)
            StockList[stk_code-1].loc[StockList[stk_code-1].order_id==bid_id_list.iloc[0]["order_id"],"volume"]=bid_id_list.iloc[0]["volume"]-ask_id_list.iloc[0]["volume"]
        else:
            Trader=" ".join([str(stk_code-1),str(bid_id_list.iloc[0]["order_id"]),str(ask_id_list.iloc[0]["order_id"]),str(TransactionAuction),str(ask_id_list.iloc[0]["volume"])])
            BuyOrder.drop(index=bid_id_list.iloc[0]["order_id"],inplace = True)
            SellOrder.drop(index=ask_id_list.iloc[0]["order_id"],inplace = True)
            StockList[stk_code-1].drop(index=bid_id_list.iloc[0]["order_id"],inplace = True)
            StockList[stk_code-1].drop(index=ask_id_list.iloc[0]["order_id"],inplace = True)
        Traders.append(Trader)   
        bid_id_list=BuyOrder[BuyOrder["price"]>=TransactionAuction]
        ask_id_list=SellOrder[SellOrder["price"]<=TransactionAuction]
    #print(StockList[0][StockList[0]["order_id"]==147])
    #print(StockList[0][StockList[0]["order_id"]==384])
    #Traders=WriteTraders()
    if len(Trader)==0:
        return Traders,StockList
    else:
        Trader,StockList=ContinuousAuctionMorning(StockList,Traders,stk_code)
        return Traders+Trader,StockList
    
def ContinuousAuctionMorning(StockList,Traders,stk_code):
    print(Traders)
    Stock=StockList[stk_code-1]
    if len(Stock)==0:
        continue
    BuyOrder=Stock[Stock["direction"]=='1'].sort_values(['price','order_id'],ascending=(False,True))
    SellOrder=Stock[Stock["direction"]=='-1'].sort_values(['price','order_id'],ascending=(True,True))
            #最高买入申报和最低卖出申报价格相同，以该价格成交
    if BuyOrder.price.max()==SellOrder.price.max():
        TransactionAuction=BuyOrder.price.max()
        Trader,StockList=deal(BuyOrder,SellOrder,TransactionAuction,StockList,stk_code,Traders)
        Traders=Trader+Traders
        #申买价高于即时揭示最低申卖价，以最低申卖价成交；
    if BuyOrder.price.any()<SellOrder.price.min():
        TransactionAuction=SellOrder.price.min()
        Trader,StockList=deal(BuyOrder,SellOrder,TransactionAuction,StockList,stk_code,Traders)
        Traders=Trader+Traders
         #申卖价低于最高申买价，以最高申买价成交。
    if SellOrder.price.any()>BuyOrder.price.max():
        Trader,StockList=deal(BuyOrder,SellOrder,TransactionAuction,StockList,stk_code,Traders)
        Traders=Trader+Traders
        TransactionAuction=SellOrder.price.max()
        #两个委托如果不能全部成交，剩余的继续留在单上，等待下次成交。
    return Traders,StockList


def ConnectTrader1(send_pack,Tradeip1="10.216.68.192",Tradeport1=31622):
    global Trader_list
    host = Tradeip1
    server_address=(host,Tradeport)
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(server_address)
    signal(SIGPIPE, SIG_IGN)
    clientsocket.sendall(send_pack.encode("utf-8"))
    recv_data = clientsocket.recv(1024).decode()
    socket().close(1)
    if recv_data!="yes":
        ConnectTrader1(send_pack)

def ConnectTrader2(send_pack,Tradeip2="10.216.68.192",Tradeport2=31622):
    global Trader_list
    host = Tradeip
    server_address=(host,Tradeport)
    clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    clientsocket.connect(server_address)
    signal(SIGPIPE, SIG_IGN)
    clientsocket.sendall(send_pack.encode("utf-8"))
    recv_data = clientsocket.recv(1024).decode()
    socket().close(1)
    if recv_data!="yes":
        ConnectTrader2(send_pack)

from signal import signal, SIGPIPE, SIG_DFL, SIG_IGN
def send_Traders(Traders,stk_code,fdmap,fd):
    if len(Traders)>0:
        index=0
        res = max(Traders, key=len, default='')
        tail=1024//(max(len(res),1)+1)-1
        while index<=len(Traders)-tail:
            data = 'T'.join(Traders[index:index+tail])
            send_pack="!T"+str(len(data))+"TT"+data
            fdmap[fd].sendall(send_pack.encode('utf-8'))
            index+=tail
            print(len(send_pack))
            print(send_pack)
        data = 'T'.join(Traders[index:len(Traders)])
        send_pack="!T"+str(len(data))+"TT"+data
        if stk_code<=5:
            ConnectTrader1(send_pack)
        else:
            ConnectTrader2(send_pack)
        print(len(send_pack))
        print(send_pack)
    
def Order_data(data,StockList,data_list,fdmap,fd):
    Traders=[]
    if data=="!$$$":
        Traders,StockList=ContinuousAuctionMorning(StockList,Traders,stk_code)
        if len(Traders)>0:
            send_Traders(Traders,stk_code,fdmap,fd)
            return StockList,data_list,Traders
        else:
            fdmap[fd].send(("over").encode("utf-8"))
            return StockList,data_list,Traders
    if len(data_list)>0:
        data_list[0]=data_list[0]+data
    else:
        data_list.append(data)
    print(data_list[0])
    bg=data_list[0].index("!$")
    ed=data_list[0].index("$$")
    if bg+2<ed:
        length=int(data_list[0][bg+2:ed])
    else:
        length=0
    Renewable(StockList,data_list[0][:bg]+data_list[0][ed+2:ed+2+length])
    for i in range(1,11):
        Traders,StockList=ContinuousAuctionMorning(StockList,Traders,stk_code)
        send_Traders(Traders,stk_code,fdmap,fd)
    data_list[0]=data_list[0][ed+2+length:]
    return StockList,data_list,Traders

def Hook_data(data,hook_list,fdmap,fd):
    key_words=["self_order_id", "target_stk_code", "target_trade_idx", "arg"]
    HookList=[pd.DataFrame([],columns=key_words) for i in range(10)]
    if 'begin' in data:
        hook_list=[""]
        #fdmap[fd].send(("Hook_begin").encode("utf-8"))
    elif 'over' in data:
        fdmap[fd].send(("Hook_over").encode("utf-8"))
        hook_list[0]=hook_list[0]+data[:data.index("Hooks_over")+1]
        #print(hook_list[0])
        hooks=hook_list[0].split("H")
        #print(hooks)
        for hook in hooks:
            if len(hook)>2:
                hook_item=hook.split(" ")
                #print(hook_item)
                HookList[int(hook_item[0])-1].loc[int(hook_item[1])]=hook_item[1:]
    else:
        hook_list[0]=hook_list[0]+data
    return HookList,hook_list

def main():
    import configparser
    global Trader_list,Tradeport2,Tradeip2,Exchangeip1,Exchangeport1,Exchangeip2,Exchangeport2
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
    global StockList,HookList
    StockList=InitialStockList()
    host = gethostname()#ip-10-216-68-189#"10.216.68.190" #socket.gethostname()
    print('当前主机名称为 : ' + gethostname())# 查看当前主机名
    print('当前主机的IP为: ' + gethostbyname(gethostname())) # 根据主机名称获取当前IP
    s = socket()
    s.setsockopt(SOL_SOCKET,SO_REUSEADDR,1)#创建套接字作为需要被关注的IO
    s.bind((Exchangeip1,Exchangeport1))
    s.listen(10)
    ep = epoll()#创建epoll()对象
    fdmap = {s.fileno():s}#创建查找字典，{文件描述符fileno:io_obj},将s监听套接字维护进入系统的监控字典中
    ep.register(s,EPOLLIN | EPOLLERR) #监控s,读IO或异常IO
    #循环监控IO的发生(不断会有请求传进来）
    data_list=[]
    hook_list=[]
    Traders=[]
    signal(SIGPIPE, SIG_IGN)
    while True:
        events = ep.poll()
        print(events)
        for fd,event in events:
            if fd == s.fileno():
                c,addr = fdmap[fd].accept()
                print('Connect from',addr)
                ep.register(c,EPOLLIN)
                fdmap[c.fileno()] = c
            elif event & EPOLLIN:
                data = fdmap[fd].recv(1024).decode()
                if '$' in data:
                    StockList,data_list,Trader=Order_data(data,StockList,data_list,fdmap,fd)
                #Traders+=Trader
                if 'H' in data:
                    HookList,hook_list=Hook_data(data,hook_list,fdmap,fd)
                if not data or data=="over":
                    print("客户端退出")
                    ep.unregister(fd)
                    fdmap[fd].close()
                    del fdmap[fd]
                    continue

main()