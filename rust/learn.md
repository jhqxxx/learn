<!--
 * @Author: jhq
 * @Date: 2024-01-28 17:23:30
 * @LastEditTime: 2025-02-10 15:07:30
 * @Description: 
-->
###### tips
* cargo build
* cargo run
* cargo check
* cargo huild --release

###### 常量与变量
* const MAX_POINTS:u32 = 100_000;

###### Shadowing(隐藏)
    let x = 5;
    let x = x + 1;
    let x = x + 2;
    shadow和变量标记为mut是不一样的
    多用于变量类型转换


###### 数据类型
1. 标量
    a. 整数 有符号/无符号
    b. 浮点类型
    c. 数值操作，加减乘除
2. 字符类型
3. 复合类型
    a. 元组 tuple, 类型可不同
    b. 数组

###### 函数
1. 语句：
2. 表达式：对应一个值
3. 返回值


###### 所有权
rust通过所有权系统来管理内存
栈-stack LIFO, 数据大小已知，入栈出栈
堆-heap，分配
* 每个值都有一个变量，这个变量是该值的所有者
* 每个值同时只能有一个所有者
* 当所有者超出作用域时，该值被删除
1. 复制 copy trait，简单标量的组合类型，直接入栈，不需要分配内存或某种资源
    整数
    bool
    char 
    浮点
    Tuple(如果字段是可复制的)
2. 移动
    let s1 = string::from("hello");
    let s2 = s1;      s1的所有权移动到了s2, s1不可调用
3. 克隆：深度拷贝
4. 引用 &：允许你引用某些值但不取得其所有权
5. 借用：把引用作为函数参数的操作
6. 可变引用
7. 悬空引用
8. 切片：不持有所有权


##### struct
1. struct
2. tuple struct
* 类单元结构体，一个没有任何字段的结构体，常用在想要在某个类型上实现trait但不需要在类型中存储数据时。
3. 方法 impl块定义
4. 关联函数：在impl块里定义不把self作为第一个参数的函数，叫关联函数，常用于构造器
* 加外部属性#[derive(Debug)]


##### 枚举 enum
enum Message {  
    Quit, 
    Move { x: i32, y: i32 }, 
    Write(String), 
    ChangeColor(i32, i32, i32), 
}

Option<T>:一个值要么有值要么没值
enum Option<T> {  
    None,
    Some(T),
}

##### match
处理枚举的控制流结构
other分支：匹配所有情况，必须放在最后
_:可以匹配任意值，但不绑定到该值

* if let:处理只匹配一个模式的值，而忽略其他模式的情况
* if let和else:else块中的代码与match表达式中的_分支中的代码相同

###### 模块
Package:Cargo的一个功能，它允许你创建、测试和分享crate
Crates:一个模块的树形结构，它形成了库或二进制项目
Module和use:允许你控制作用域和路径私有性
Path：一个命名如结构体、函数或模块等项的方式

##### 集合
Vector v


shift+alt+f: 格式化rust代码

###### String
+: 连接字符串：使用了类似这个签名的方法 fn add(self, s:&str) -> String{...}
format!: 连接多个字符串，返回字符串，不会获得参数的所有权
字节
标量值
字型簇 

##### HashMap<K,V>
new()
insert()
同构：所有的key是同一个类型，所有的value是同一个类型
get()
for
entry()
or_insert()

###### 错误处理
不可恢复的错误与panic
传播错误
？运算符 

###### 泛型

###### 生命周期

vscode快捷键：
shift+alt+f: 格式化rust代码