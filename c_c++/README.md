#### 容易混淆
* const
* static
* this指针
* inline内联函数
* 宏
* volatile关键字
* assert()
* sizeof()
* #pragma pack(n)
* 位域
* extern "C"
* typedef
* union
* explicit
* friend
* using
* decltype关键字
* 虚函数 virtual int A(); 虚函数是是实现的，哪怕是空实现，它的作用就是能让这个函数在子类里面可以被覆盖。虚函数在子类里可以不重写。
* 纯虚函数：virtual int A() = 0; 是一种特殊的虚函数，在基类中不定义，在派生类中必须定义，否则会报错。纯虚函数只是一个函数的声明，留到子类里去实现。带纯虚函数的类叫抽象类，这种类不能直接生成对象，只能被继承，并重写其虚函数后，才能使用。

* 智能指针的行为类似于常规指针，重要的区别时它负责自动释放所指向的对象
* shared_ptr：允许多个指针指向同一个对象，使用引用计数来记录指向该对象的指针数量，当计数为0时，释放所指向的堆内存。
    - reset()：参数为空时，会释放shared_ptr指向的对象，参数不为空时会先释放原来拥有的对象，再获取新对象的所有权。
* unique_ptr：独占所指向的对象。
* std::mutex- 互斥锁，用来保护共享资源，防止多个线程同时访问共享资源。
    - 定义锁：std::mutex mutex;
    - 获取锁：mutex.lock();
    - 释放锁：mutex.unlock();
    - std::lock_guard<std::mutex> lock(mutex); lock_guard简化了lock()和unlock()的调用，lock_guard在构造时自动锁定互斥量，而在退出作用域时，lock_guard对象销毁，会自动释放锁。

* std::remove_pointer<T>::type()用于去除指针类型，如果T是float*,则返回结果是float。
    - std::remove_pointer<T>::type() 等价于 std::remove_pointer_t<T>

* typename： typename用于声明模板类型参数，当编译器遇到依赖于模板参数的嵌套类型时，它无法自动推断出该名称是一个类型还是一个静态成员变量或函数，typename关键字明确告诉编译器，这个名称是一个类型。
