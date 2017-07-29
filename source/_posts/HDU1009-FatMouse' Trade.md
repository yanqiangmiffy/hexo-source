---
title: HDU1009-FatMouse' Trade
date: 2017-07-24 14:54:03
---
<Excerpt in index | 首页摘要> 

HDU 1009 FatMouse’ Trade 题解报告
<!-- more -->

# 问题描述
```
FatMouse prepared M pounds of cat food, ready to trade with the cats guarding the warehouse containing his favorite food, JavaBean.

The warehouse has N rooms. The i-th room contains J[i] pounds of JavaBeans and requires F[i] pounds of cat food. FatMouse does not have to trade for all the JavaBeans in the room, instead, he may get J[i]* a% pounds of JavaBeans if he pays F[i]* a% pounds of cat food. Here a is a real number. Now he is assigning this homework to you: tell him the maximum amount of JavaBeans he can obtain.

```
# 输入
```
The input consists of multiple test cases. Each test case begins with a line containing two non-negative integers M and N. Then N lines follow, each contains two non-negative integers J[i] and F[i] respectively. The last test case is followed by two -1's. All integers are not greater than 1000.

```
# 输出
```
For each test case, print in a single line a real number accurate up to 3 decimal places, which is the maximum amount of JavaBeans that FatMouse can obtain.

```
# 样例输入
```
5 3
7 2
4 3
5 2
20 3
25 18
24 15
15 10
-1 -1

```
# 样例输出

```
13.333
31.500
```
# 解题报告
大意：一只老鼠有M磅的猫粮，另外有一只猫控制了老鼠的N个房间，这些房间里面放了老鼠爱吃的绿豆，给出每个房间的绿豆数量，和这个房间的绿豆所需要的猫粮数，现在要求老鼠用这M磅的猫粮最多能换到多少它爱吃的绿豆？

贪心题，由于所有的绿豆都是一样的，所以如果老鼠想要换到最多的绿豆，便可以换猫控制的房间里面最便宜的绿豆，也就是说先换取单位数量的绿豆所需要最少的猫粮的房间里的绿豆，这样就可以保证换到的绿豆是最多的。具体实现可以用一个结构体，里面保存每个房间里面有的绿豆的数量和换取这个房间的绿豆时所需要的猫粮的数量和换取这个房间的 单位重量的绿豆所需要的猫粮数（以下简称单价），然后再按照单价升序给这些结构体排一次序，这时就可以从最便宜的绿豆开始换了。

# 代码
```
#include<iostream>
#include<algorithm>
#include<stdio.h>
using namespace std;
struct stu
{
    int a;
    int b;
    double rate;
}s[1005];
bool cmp(stu x,stu y)
{
    if(x.rate!=y.rate)
    return x.rate>y.rate;
    else
    return x.a<y.a;
}
int main()
{   //freopen("1.txt","r",stdin);
    int m,n,i;
    double cnt;
    while(scanf("%d %d",&m,&n)!=EOF)
    {   if(m==-1&&n==-1)break;
        cnt=0;
        for(i=0;i<n;i++)
        {
            scanf("%d %d",&s[i].a,&s[i].b);
            s[i].rate=s[i].a*1.0/s[i].b;
        }
        sort(s,s+n,cmp);
        for(i=0;i<n;i++)
        {
            if(m>=s[i].b)
            {
                m-=s[i].b;
                cnt+=s[i].a;
            }
           else
           {
               cnt+=s[i].rate*m;break;
           }
        }
     printf("%.3lf\n",cnt);
    }
    return 0;
}
```