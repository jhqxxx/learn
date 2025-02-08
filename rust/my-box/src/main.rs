/*
 * @Author: jhq
 * @Date: 2024-03-31 19:11:09
 * @LastEditTime: 2024-03-31 19:36:16
 * @Description: 
 */

use crate::List::{Cons, Nil};
use std::ops::Deref;

struct MyBox<T>(T);

impl<T> MyBox<T> {
    fn new(x:T)->MyBox<T> {
        MyBox(x)
    }
}

impl<T> Deref for MyBox<T> {
    type Target = T;

    fn deref(&self) -> &T {
        &self.0
    }
}

fn hello(name: &str) {
    println!("Hello, {}", name);
}

fn main() {
    let m = MyBox::new(String::from("Rust"));
    hello(&m);
    hello("rust");
    // let x=5;
    // let y = MyBox::new(x);

    // assert_eq!(5, x);
    // assert_eq!(5, *y);
    // let list = Cons(1, 
    //     Box::new(Cons(2, 
    //         Box::new(Cons(3, 
    //             Box::new(Nil))))));
}

enum  List {
    Cons(i32, Box<List>),
    Nil,
}