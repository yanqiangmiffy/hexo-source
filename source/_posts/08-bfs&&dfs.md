title: 图的广度优先搜索和图的深度优先搜索
date: 2017/08/30 15:29:25
tags: [dfs, bfs, 广度优先搜索,深度优先搜索]

---

<Excerpt in index | 首页摘要> 
图的邻接链表的表示方法、图的广度优先搜索和图的深度优先搜索
<!-- more -->


# 邻接链表
>邻接表表示法将图以邻接表（adjacency  lists）的形式存储在计算机中。所谓图的邻接表，也就是图的所有节点的邻接表的集合；而对每个节点，它的邻接表就是它的所有出弧。邻接表表示法就是对图的每个节点，用一个单向链表列出从该节点出发的所有弧，链表中每个单元对应于一条出弧。为了记录弧上的权，链表中每个单元除列出弧的另一个端点外，还可以包含弧上的权等作为数据域。图的整个邻接表可以用一个指针数组表示。例如下图所示，邻接表表示为

![邻接链表](http://upload-images.jianshu.io/upload_images/1531909-e2e11cfa815bf198.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

# 广度优先搜索
## 基本思路
1. 把根节点放到队列的末尾。
2. 每次从队列的头部取出一个元素，查看这个元素所有的下一级元素，把它们放到队列的末尾。并把这个元素记为它下一级元素的前驱。
3. 找到所要找的元素时结束程序。
4. 如果遍历整个树还没有找到，结束程序。
 ## 代码实现
```C
//http://www.geeksforgeeks.org/breadth-first-traversal-for-a-graph/
// Program to print BFS traversal from a given source vertex. BFS(int s)
// traverses vertices reachable from s.
#include<iostream>
#include <list>

using namespace std;

// This class represents a directed graph using adjacency list representation
class Graph
{
    int V;    // No. of vertices
    list<int> *adj;    // Pointer to an array containing adjacency lists
public:
    Graph(int V);  // Constructor
    void addEdge(int v, int w); // function to add an edge to graph
    void BFS(int s);  // prints BFS traversal from a given source s
};

Graph::Graph(int V)
{
    this->V = V;
    adj = new list<int>[V];
}

void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); // Add w to v’s list.
}

void Graph::BFS(int s)
{
    // Mark all the vertices as not visited
    bool *visited = new bool[V];
    for(int i = 0; i < V; i++)
        visited[i] = false;

    // Create a queue for BFS
    list<int> queue;

    // Mark the current node as visited and enqueue it
    visited[s] = true;
    queue.push_back(s);

    // 'i' will be used to get all adjacent vertices of a vertex
    list<int>::iterator i;

    while(!queue.empty())
    {
        // Dequeue a vertex from queue and print it
        s = queue.front();
        cout << s << " ";
        queue.pop_front();

        // Get all adjacent vertices of the dequeued vertex s
        // If a adjacent has not been visited, then mark it visited
        // and enqueue it
        for(i = adj[s].begin(); i != adj[s].end(); ++i)
        {
            if(!visited[*i])
            {
                visited[*i] = true;
                queue.push_back(*i);
            }
        }
    }
}

// Driver program to test methods of graph class
int main()
{
    // Create a graph given in the above diagram
    Graph g(4);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 2);
    g.addEdge(2, 0);
    g.addEdge(2, 3);
    g.addEdge(3, 3);

    cout << "Following is Breadth First Traversal "
         << "(starting from vertex 2) n:";
    g.BFS(2);

    return 0;
}

```
# 深度优先搜索
## 基本思路
1. 访问顶点v；
2. 依次从v的未被访问的邻接点出发，对图进行深度优先遍历；直至图中和v有路径相通的顶点都被访问；
3. 若此时图中尚有顶点未被访问，则从一个未被访问的顶点出发，重新进行深度优先遍历
## 代码实现
```C
//http://www.geeksforgeeks.org/depth-first-traversal-for-a-graph/
// C++ program to print DFS traversal from a given vertex in a  given graph
#include<iostream>
#include<list>

using namespace std;

// Graph class represents a directed graph using adjacency list representation
class Graph
{
    int V;    // No. of vertices
    list<int> *adj;    // Pointer to an array containing adjacency lists
    void DFSUtil(int v, bool visited[]);  // A function used by DFS
public:
    Graph(int V);   // Constructor
    void addEdge(int v, int w);   // function to add an edge to graph
    void DFS(int v);    // DFS traversal of the vertices reachable from v
};

Graph::Graph(int V)
{
    this->V = V;
    adj = new list<int>[V];
}

void Graph::addEdge(int v, int w)
{
    adj[v].push_back(w); // Add w to v’s list.
}

void Graph::DFSUtil(int v, bool visited[])
{
    // Mark the current node as visited and print it
    visited[v] = true;
    cout << v << " ";

    // Recur for all the vertices adjacent to this vertex
    list<int>::iterator i;
    for (i = adj[v].begin(); i != adj[v].end(); ++i)
        if (!visited[*i])
            DFSUtil(*i, visited);
}

// DFS traversal of the vertices reachable from v.
// It uses recursive DFSUtil()
void Graph::DFS(int v)
{
    // Mark all the vertices as not visited
    bool *visited = new bool[V];
    for (int i = 0; i < V; i++)
        visited[i] = false;

    // Call the recursive helper function to print DFS traversal
    DFSUtil(v, visited);
}

int main()
{
    // Create a graph given in the above diagram
    Graph g(4);
    g.addEdge(0, 1);
    g.addEdge(0, 2);
    g.addEdge(1, 2);
    g.addEdge(2, 0);
    g.addEdge(2, 3);
    g.addEdge(3, 3);

    cout << "Following is Depth First Traversal (starting from vertex 2) n:";
    g.DFS(2);

    return 0;
}

```
# 运行结果

![输入](http://upload-images.jianshu.io/upload_images/1531909-df3bc225970ef1bf.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![广度优先搜索](http://upload-images.jianshu.io/upload_images/1531909-c28e0034fe262c31.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

![深度优先搜索](http://upload-images.jianshu.io/upload_images/1531909-0ac37d322b53006b.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

也可以试试从其他定点（0,1,3）开始遍历☺
*参考*
[初识图，图的存储（邻接矩阵，邻接链表）和深搜遍历](http://blog.csdn.net/dextrad_ihacker/article/details/50132129)
[算法与数据结构（2）——图的表示法与常用的转化算法](http://www.cnblogs.com/liushang0419/archive/2011/05/06/2039386.html)