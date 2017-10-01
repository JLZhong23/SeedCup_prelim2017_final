# -*- coding:utf-8 -*-
'''文件作用：整理相关数据，获取训练数据集文件以及测试数据集文件
生成文件说明：‘Team_elo.csv‘队伍Elo天梯积分数据
            ’TeamSRS.csv’队伍平均胜率，得分，SRS等数据
            ‘TeamTotal.csv’队伍三分命中，出手，两分命中，出手等数据
            ‘TeamData_win.csv’训练集中队伍比赛胜负情况数据
            ‘TeamDataTrain.csv’最终训练使用数据
            ‘TeamDataTest.csv’需要预测比赛的相关数据
文件使用说明：直接python2 运行'''
import pandas as pd
import csv
import math
import random
import numpy as np



'''函数说明：根据‘matchDataTrain.csv’中的比赛情况计算相应队伍的Elo积分值
        1600为初始积分'''
def get_elo():
    elo=1600
    TeamElo={}
    for i in range(0,208):
        TeamElo[i]=elo
    Match=pd.read_csv('matchDataTrain.csv')
    #print Match
    sha_ma=Match.shape
    for r in range(0,sha_ma[0]):
        Team1=Match.loc[r]['客场队名']
        Team2=Match.loc[r]['主场队名']
        #print Match.loc[r]['比分（客场:主场）']
        score=Match.loc[r]['比分（客场:主场）'].split(':',1)
        rank1=TeamElo[Team1]
        rank2=TeamElo[Team2]+100
        if score[0]>score[1]:
            rankdiff1=rank2-rank1
            exp=rankdiff1/400
            pro=1/(1+math.pow(10,exp))
            if TeamElo[Team1] < 2100:
                k = 32
            elif TeamElo[Team1] >= 2100 and TeamElo[Team1] < 2400:
                k = 24
            else:
                k = 16
            TeamElo[Team1]=TeamElo[Team1]+k*(1-pro)
            TeamElo[Team2]=TeamElo[Team2]-k*(1-pro)
        else:
            rankdiff2 = rank1 - rank2
            exp2 = rankdiff2 / 400
            pro = 1 / (1 + math.pow(10, exp2))
            if TeamElo[Team2] < 2100:
                k = 32
            elif TeamElo[Team2] >= 2100 and TeamElo[Team2] < 2400:
                k = 24
            else:
                k = 16
            TeamElo[Team2] = TeamElo[Team2] + k * (1 - pro)
            TeamElo[Team1] = TeamElo[Team1] - k * (1 - pro)
    return TeamElo


'''函数说明：将获取的ELo积分写入文件'''
def Write_Elo_csv():
    Elo=get_elo()
    with open('Team_elo.csv','wb') as f:
        write=csv.writer(f)
        write.writerow(['队名','Elo'])
        for i in range(0,208):
            write.writerow([i,Elo[i]])

'''函数说明：根据‘teamData.csv’文件获取相应的队伍的失误，篮板等数据'''
def get_TeamTotal():
    Teamdata = pd.read_csv("teamData.csv")
    # print Teamdata

    dataname = Teamdata.columns.tolist()
    #for da in dataname:
     #   print da
    TeamTotle = []
    sha = Teamdata.shape
    Tp =Tpa =Tpb =Sp = 0
    Spa = Spb = 0
    Fp =Fpa = Fpb = 0
    TRB = ORB = DRB = AST = STL = BLK = TOV = PF = PTS = 0
    for r in range(0, sha[0] - 1):
        if Teamdata.loc[r]['队名'] == Teamdata.loc[r + 1]['队名']:
            Tp = Tp + Teamdata.loc[r]['投篮出手次数'] * Teamdata.loc[r]['出场次数'] / 82
            Tpa = Tpa + Teamdata.loc[r]['投篮命中次数'] * Teamdata.loc[r]['出场次数'] / 82
            Sp = Sp + Teamdata.loc[r]['三分出手次数'] * Teamdata.loc[r]['出场次数'] / 82
            Spa = Spa + Teamdata.loc[r]['三分命中次数'] * Teamdata.loc[r]['出场次数'] / 82
            #Spb = Spa / Sp
            Fp = Fp + Teamdata.loc[r]['罚球出手次数'] * Teamdata.loc[r]['出场次数'] / 82
            Fpa = Fpa + Teamdata.loc[r]['罚球命中次数'] * Teamdata.loc[r]['出场次数'] / 82
            #Fpb = Fpa / Fp
            TRB = TRB + Teamdata.loc[r]['篮板总数'] * Teamdata.loc[r]['出场次数'] / 82
            ORB = ORB + Teamdata.loc[r]['前场篮板'] * Teamdata.loc[r]['出场次数'] / 82
            DRB = DRB + Teamdata.loc[r]['后场篮板'] * Teamdata.loc[r]['出场次数'] / 82
            AST = AST + Teamdata.loc[r]['助攻'] * Teamdata.loc[r]['出场次数'] / 82
            STL = STL + Teamdata.loc[r]['抢断'] * Teamdata.loc[r]['出场次数'] / 82
            BLK = BLK + Teamdata.loc[r]['盖帽'] * Teamdata.loc[r]['出场次数'] / 82
            TOV = TOV + Teamdata.loc[r]['失误'] * Teamdata.loc[r]['出场次数'] / 82
            PF = PF + Teamdata.loc[r]['犯规'] * Teamdata.loc[r]['出场次数'] / 82
            PTS = PTS + Teamdata.loc[r]['得分'] * Teamdata.loc[r]['出场次数'] / 82
        else:
            Tp = Tp + Teamdata.loc[r]['投篮出手次数'] * Teamdata.loc[r]['出场次数'] / 82
            Tpa = Tpa + Teamdata.loc[r]['投篮命中次数'] * Teamdata.loc[r]['出场次数'] / 82
            Tpb = Tpa/Tp
            Sp = Sp + Teamdata.loc[r]['三分出手次数'] * Teamdata.loc[r]['出场次数'] / 82
            Spa = Spa + Teamdata.loc[r]['三分命中次数'] * Teamdata.loc[r]['出场次数'] / 82
            Spb = Spa/Sp
            Fp = Fp + Teamdata.loc[r]['罚球出手次数'] * Teamdata.loc[r]['出场次数'] / 82
            Fpa = Fpa + Teamdata.loc[r]['罚球命中次数'] * Teamdata.loc[r]['出场次数'] / 82
            Fpb = Fpa/Fp
            TRB = TRB + Teamdata.loc[r]['篮板总数'] * Teamdata.loc[r]['出场次数'] / 82
            ORB = ORB + Teamdata.loc[r]['前场篮板'] * Teamdata.loc[r]['出场次数'] / 82
            DRB = DRB + Teamdata.loc[r]['后场篮板'] * Teamdata.loc[r]['出场次数'] / 82
            AST = AST + Teamdata.loc[r]['助攻'] * Teamdata.loc[r]['出场次数'] / 82
            STL = STL + Teamdata.loc[r]['抢断'] * Teamdata.loc[r]['出场次数'] / 82
            BLK = BLK + Teamdata.loc[r]['盖帽'] * Teamdata.loc[r]['出场次数'] / 82
            TOV = TOV + Teamdata.loc[r]['失误'] * Teamdata.loc[r]['出场次数'] / 82
            PF = PF + Teamdata.loc[r]['犯规'] * Teamdata.loc[r]['出场次数'] / 82
            PTS = PTS + Teamdata.loc[r]['得分'] * Teamdata.loc[r]['出场次数'] / 82
            TeamNumber = Teamdata.loc[r]['队名']
            TeamTotle.append([TeamNumber, Tp, Tpa, Tpb, Sp, Spa, Spb, Fp, Fpa, Fpb, TRB, ORB
                                 , DRB, AST, STL, BLK, TOV, PF, PTS, ])
            Tp = Tpa = Tpb = Sp = 0
            Spa = Spb = 0
            Fp = Fpa = Fpb = 0
            TRB = ORB = DRB = AST = STL = BLK = TOV = PF = PTS = 0

    Tp = Tp + Teamdata.loc[sha[0] - 1]['投篮出手次数'] * Teamdata.loc[r]['出场次数'] / 82
    Tpa = Tpa + Teamdata.loc[sha[0] - 1]['投篮命中次数'] * Teamdata.loc[r]['出场次数'] / 82
    Tpb = Tpa / Tp
    Sp = Sp + Teamdata.loc[sha[0] - 1]['三分出手次数'] * Teamdata.loc[r]['出场次数'] / 82
    Spa = Spa + Teamdata.loc[sha[0] - 1]['三分命中次数'] * Teamdata.loc[r]['出场次数'] / 82
    Spb = Spa / Sp
    Fp = Fp + Teamdata.loc[sha[0] - 1]['罚球出手次数'] * Teamdata.loc[r]['出场次数'] / 82
    Fpa = Fpa + Teamdata.loc[sha[0] - 1]['罚球命中次数'] * Teamdata.loc[r]['出场次数'] / 82
    Fpb = Fpa / Fp
    TRB = TRB + Teamdata.loc[sha[0] - 1]['篮板总数'] * Teamdata.loc[r]['出场次数'] / 82
    ORB = ORB + Teamdata.loc[sha[0] - 1]['前场篮板'] * Teamdata.loc[r]['出场次数'] / 82
    DRB = DRB + Teamdata.loc[sha[0] - 1]['后场篮板'] * Teamdata.loc[r]['出场次数'] / 82
    AST = AST + Teamdata.loc[sha[0] - 1]['助攻'] * Teamdata.loc[r]['出场次数'] / 82
    STL = STL + Teamdata.loc[sha[0] - 1]['抢断'] * Teamdata.loc[r]['出场次数'] / 82
    BLK = BLK + Teamdata.loc[sha[0] - 1]['盖帽'] * Teamdata.loc[r]['出场次数'] / 82
    TOV = TOV + Teamdata.loc[sha[0] - 1]['失误'] * Teamdata.loc[r]['出场次数'] / 82
    PF = PF + Teamdata.loc[sha[0] - 1]['犯规'] * Teamdata.loc[r]['出场次数'] / 82
    PTS = PTS + Teamdata.loc[sha[0] - 1]['得分'] * Teamdata.loc[r]['出场次数'] / 82
    TeamNumber = Teamdata.loc[sha[0] - 1]['队名']
    TeamTotle.append([TeamNumber, Tp, Tpa, Tpb, Sp, Spa, Spb, Fp, Fpa, Fpb, TRB, ORB
                         , DRB, AST, STL, BLK, TOV, PF, PTS, ])
    return TeamTotle


'''函数说明：将队伍基本信息写入文件'''
def Write_TeamTotal_csv():
    TeamTotal=get_TeamTotal()
    with open('TeamTotal.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['队名', '投篮出手次数', '投篮命中次数', '投篮命中率', '三分出手次数', '三分命中次数', '三分命中率',
                         '罚球出手次数', '罚球命中次数', '罚球命中率', '篮板总数', '前场篮板', '后场篮板', '助攻', '抢断',
                         '盖帽', '失误', '犯规', '得分'])
        writer.writerows(TeamTotal)

'''函数说明：根据‘matchDataTrain.csv’中的比赛情况获取相应的得分，SRS等数据'''
def get_Teamsrs():
    Teamdata = pd.read_csv("matchDataTrain.csv")
    TeamTotle = []
    len = Teamdata.shape[0]                  # 数据集长度
    #print len
    tow = [0.0 for x in range(0, 208)]       # 总得分
    asw = [0.0 for x in range(0, 208)]       # 平均得分
    tol = [0.0 for x in range(0, 208)]       # 总失分
    asl = [0.0 for x in range(0, 208)]       # 平均失分
    awl = [0.0 for x in range(0, 208)]       # 场均净胜分
    Srs = [0.0 for x in range(0, 208)]       # srs
    tm = [0.0 for x in range(0, 208)]        # 总场数
    tw = [0.0 for x in range(0, 208)]        # 总胜场
    tl = [0.0 for x in range(0, 208)]        # 总负场
    wtol = [0.0 for x in range(0, 208)]      # 胜率
    for r in range(0, len):
        m = int(Teamdata.loc[r]['客场队名'])
        n = int(Teamdata.loc[r]['主场队名'])
        L = Teamdata.loc[r]['比分（客场:主场）'].split(':')
        kc = int(L[0])
        #print kc
        zc = int(L[1])
        #print zc
        tow[m] = tow[m] + kc            # 总得分
        tow[n] = tow[n] + zc
        tol[m] = tol[m] + zc            # 总失分
        tol[n] = tol[n] + kc
        tm[m] = tm[m] + 1               # 总场数
        tm[n] = tm[n] + 1
        if kc > zc:                     # 总胜&负场
            tw[m] = tw[m] + 1
            tl[n] = tl[n] + 1
        else:
            tw[n] = tw[n] + 1
            tl[m] = tl[m] + 1
    x = 0
    for s in range(0, 208):
        x = x + awl[s]
    for r in range(0, 208):
        if tm[r] != 0:
            asw[r] = tow[r]*1.0/tm[r]
            asl[r] = tol[r]*1.0/tm[r]
            wtol[r] = tw[r]*1.0/tm[r]
            awl[r] = (tow[r]-tol[r])*1.0/tm[r]
            Srs[r] = awl[r]*207.0/208
    for t in range(0, 208):
        TeamTotle.append([t, tow[t], asw[t], tol[t], asl[t], awl[t], Srs[t], tm[t], tw[t], tl[t], wtol[t], ])
    return TeamTotle

'''函数说明：将srs信息写入文件'''
def write_srs_csv():
    TeamTotal = get_Teamsrs()
    with open('TeamSRS.csv', 'wb') as f:
        writer = csv.writer(f)
        writer.writerow(['队名','总得分','平均得分','总失分','平均失分','场均净胜分','srs','总场数','胜场数','负场数','胜率'])
        writer.writerows(TeamTotal)

'''函数说明：整合训练集或则测试集队伍比赛信息并写入文件'''
def getnewteamvsdata(matchfile,writefile):
    MatchData = pd.read_csv(matchfile)
    teamdata = pd.read_csv('TeamSRS.csv')
    Data1 = MatchData.values
    Data2 = teamdata.values
    Data3 = pd.read_csv('Team_elo.csv', index_col='队名').values
    Data4=pd.read_csv('TeamTotal.csv',index_col='队名').values
    f = open(writefile, 'wb')
    a=[]
    for i in range(60):
        a.append(i)
    writer = csv.writer(f)
    writer.writerow(a)
    for r in range(0, MatchData.shape[0]):
        Team1 = MatchData.loc[r]['客场队名']
        Team2 = MatchData.loc[r]['主场队名']
        x=Data2[Team1].tolist()+Data3[Team1].tolist()+Data2[Team2].tolist()+Data3[Team2].tolist()+Data4[Team1].tolist()+Data4[Team2].tolist()
        writer.writerow(x)
    f.close()

'''函数说明：获取训练集的队伍比赛胜负情况'''
def getwin():
    MatchData = pd.read_csv('matchDataTrain.csv')
    f = open('TeamData_win.csv', 'wb')
    a=['客场','主场']
    writer = csv.writer(f)
    writer.writerow(a)
    for r in range(MatchData.shape[0]):
        y=[]
        score = MatchData.loc[r]['比分（客场:主场）'].split(':')
        #print score
        if int(score[0]) > int(score[1]):
            y.append(1)#Visiteam win
            y.append(0)
        else:
            y.append(0)#hometeam win
            y.append(1)

        writer.writerow(y)
    f.close()

Write_TeamTotal_csv()
Write_Elo_csv()
write_srs_csv()
getwin()
getnewteamvsdata('matchDataTrain.csv','TeamDataTrain.csv')
getnewteamvsdata('matchDataTest.csv','TeamDataTest.csv')

