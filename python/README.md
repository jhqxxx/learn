<!--
 * @Author: jhq
 * @Date: 2025-02-08 14:15:50
 * @LastEditTime: 2025-04-10 15:22:07
 * @Description:
-->

- list
  - append: 添加一个元素到列表的末尾，如果添加的是另一个列表，则该列表作为一个整体被添加为单个元素
  - extend：将一个可迭代对象(如列表)中的所有元素逐一添加到源列表中，而不会创建嵌套列表

###### union 类型

3.10 版本引入的新类型，联合类型是一种特殊的数据类型，它可以存储多种类型的值，但这个类型的变量在特定的时刻只能是其中一种类型。

###### ThreadPoolExecutor

- 使用 ThreadPoolExecutor 来实例化线程池对象
- 使用 submit 函数来提交线程需要执行的任务（函数名和参数）到线程池中，并返回该任务的句柄（类似于文件、画图），注意 submit() 不是阻塞的，而是立即返回。通过 submit 函数返回的任务句柄，能够使用 done() 方法判断任务是否结束
- 使用 cancel() 方法可以取消提交的任务，如果任务已经在线程池中运行了，就取消不了
- 使用 result() 方法可以获取任务的返回值，这个方法是阻塞的
- as_completed() 方法，上面虽然提供了判断任务是否结束的方法，但是不能在主线程中一直判断，有时候我们是得知某个任务结束了，就去获取结果，而不是一直判断每个任务有没有结束。这是就可以使用 as_completed() 方法一次取出所有任务的结果。as_completed() 方法是一个生成器，在没有任务完成的时候，会阻塞，在有某个任务完成的时候，会 yield 这个任务，就能执行 for 循环下面的语句，然后继续阻塞住，循环到所有的任务结束
- 示例
  with ThreadPoolExecutor() as executor:  
   futures = [executor.submit(_run_countgd, prompt) for prompt in prompts]  
   for future in as_completed(futures):  
   bboxes.extend(future.result())

- async:声明一个函数是异步函数，异步函数能在函数执行过程中被挂起，去执行其他异步函数，等挂起条件消失后再回来执行
- await:声明程序挂起，其后面只能跟异步程序或有__await__属性的对象

- yeild:声明一个函数是生成器函数，生成器函数在函数执行过程中，每次遇到yield关键字，就返回一个值，并挂起，等下次调用next()方法时，再从上次挂起的位置继续执行，直到遇到下一个yield关键字，如此反复，直到函数执行完毕，返回StopIteration异常

- np.memmap():能将大文件分段读写，而不是一次性将整个文件读入内存，从而减少内存占用