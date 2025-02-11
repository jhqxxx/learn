<!--
 * @Author: jhq
 * @Date: 2025-02-08 14:15:50
 * @LastEditTime: 2025-02-11 15:18:13
 * @Description: 
-->
* list
    - append: 添加一个元素到列表的末尾，如果添加的是另一个列表，则该列表作为一个整体被添加为单个元素
    - extend：将一个可迭代对象(如列表)中的所有元素逐一添加到源列表中，而不会创建嵌套列表

###### union类型
3.10版本引入的新类型，联合类型是一种特殊的数据类型，它可以存储多种类型的值，但这个类型的变量在特定的时刻只能是其中一种类型。

###### ThreadPoolExecutor
* 使用 ThreadPoolExecutor 来实例化线程池对象
* 使用 submit 函数来提交线程需要执行的任务（函数名和参数）到线程池中，并返回该任务的句柄（类似于文件、画图），注意 submit() 不是阻塞的，而是立即返回。通过 submit 函数返回的任务句柄，能够使用 done() 方法判断任务是否结束
* 使用 cancel() 方法可以取消提交的任务，如果任务已经在线程池中运行了，就取消不了
* 使用 result() 方法可以获取任务的返回值，这个方法是阻塞的
* as_completed() 方法，上面虽然提供了判断任务是否结束的方法，但是不能在主线程中一直判断，有时候我们是得知某个任务结束了，就去获取结果，而不是一直判断每个任务有没有结束。这是就可以使用 as_completed() 方法一次取出所有任务的结果。as_completed() 方法是一个生成器，在没有任务完成的时候，会阻塞，在有某个任务完成的时候，会 yield这个任务，就能执行for循环下面的语句，然后继续阻塞住，循环到所有的任务结束
* 示例
    with ThreadPoolExecutor() as executor:         
        futures = [executor.submit(_run_countgd, prompt) for prompt in prompts]         
        for future in as_completed(futures):             
            bboxes.extend(future.result())
