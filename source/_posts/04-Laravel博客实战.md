
title: Laravel博客实战
date: 2017/08/04 17:31:25
---

<Excerpt in index | 首页摘要> 
用了5天时间，参考后盾网老师的Laravel实战视频做的博客
<!-- more -->

>最近在学习Laravel，参考的课程是后盾网地Laravel5.2博客项目实战，地址是
[Laravel 5.2开发实](http://bbs.houdunwang.com/forum-247-1.html)

下面整个项目的开发过程：

# laravel-blog
基于laravel5.2的博客
## day1（7月31）：

1. 后台模板引入
2. 验证码
3. 表单验证
4. 后台权限和密码更改
5. 文章分类

## day2（8月01）:
1. 文章多级分类以及父分类
2. ajax修改排序
3. 文章分类添加
4. 文章分类编辑
5. 文章分类ajax异步删除




## day3（8月02）:
1. 文章添加以及百度编辑器Ueditor嵌入
2. 文章缩略图上传之uploadify（HTML5版本）的引入
3. 文章分页列表
4. 文章编辑
5. 文章删除

## day4（8月03）:
1. 数据库迁移以及数据填充
2. 友情链接增删改查
3. 自定义导航
4. 前台文章首页、列表页、文章模板
5. 前台模板数据共享

## day5（8月04）
1. 配置项模块的创建
2. 最新文章以及点击排行
3. 公共侧边栏模板继承
4. 文章页面信息以及详情
5. 文章上一篇下一篇以及相关文章

[项目地址](https://github.com/yanqiangmiffy/laravel-blog)

---
# 最终的效果

![前台.png](http://upload-images.jianshu.io/upload_images/1531909-c088728dbc0aeef0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![文章详情页.png](http://upload-images.jianshu.io/upload_images/1531909-5cf281d1d80cacd5.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
![管理页面.png](http://upload-images.jianshu.io/upload_images/1531909-4c956a5d9193e149.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)


---
# 踩的坑
 > ## 关于session

Laravel采用了另一套session机制，默认情况下session没有被打开，而有些情况下，我们引入的类需要开启session。比如引入验证码之后，需要把验证码字符存入session。

![error.png](http://upload-images.jianshu.io/upload_images/1531909-6a2c57ce7d3d74e0.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

此时可以在入口文件index.php打开session即可

![session.png](http://upload-images.jianshu.io/upload_images/1531909-b55f04107b59b2cc.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> ## csrf验证

在使用Laravel框架开发网站的时候，我们最好从头到底按照框架规范进行设计
![image.png](http://upload-images.jianshu.io/upload_images/1531909-4922d1a661748114.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)
在进行表单验证时，需要加上csrf token

![image.png](http://upload-images.jianshu.io/upload_images/1531909-f85e842836ad3b1c.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> return->back()->with()

return back()->with('msg','验证码错误');重定向至前一个页面，但传入的值用session('msg')无法取到

`
项目路由配置时，所有路由是配置在一个总的路由分组中，对这个分组添加了web中间件。删掉这个中间件或者去掉这个路由分组，问题得到解决
`

> ## 时区设置

默认时区采用的是UTC，需要手动改成东八区。PRC在config下的app.php文件里：
![时区.png](http://upload-images.jianshu.io/upload_images/1531909-e3fc42505ac364ed.png?imageMogr2/auto-orient/strip%7CimageView2/2/w/1240)

> ## 5.this与静态函数
```
 /* public static function tree()
     {
         $category=Category::all();
         return (new Category)->getTree($category,'cate_name','cate_id','cate_pid');
     }*/
```
```
  public function tree()
    {
        $category = $this->orderBy('cate_order','asc')->get();
        return $this->getTree($category, 'cate_name', 'cate_id', 'cate_pid');
    }
```