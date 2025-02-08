/*
 * @Author: jhq
 * @Date: 2024-02-16 17:24:00
 * @LastEditTime: 2024-02-16 22:54:11
 * @Description: 
 */

#[derive(Debug)]
struct Rectangle {
    width: u32,
    length: u32,
}

impl Rectangle {
    fn area(&self) -> u32 {
        self.length * self.width
    }

    fn can_hold(&self, other: &Rectangle) -> bool {
        self.width > other.width && self.length > other.length
    }

    fn square(size: u32) -> Rectangle {
        Rectangle {
            width: size,
            length: size,
        }
    }
}

enum IpAddrKind {
    V4(u8, u8, u8, u8),
    V6(String),
}

struct IpAddr {
    kind: IpAddrKind,
    address: String,
}

enum Message {
    Quit,
    Move { x: i32, y: i32 },
    Write(String),
    ChangeColor(i32, i32, i32),
}

impl Message {
    fn call(&self) {}
}

#[derive(Debug)]
enum UsState {
    Alabama,
    Alaska,
}

enum Coin {
    Penny,
    Nickel,
    Dime,
    Quarter(UsState),
}

fn value_in_cents(coin: Coin) -> u8 {
    match coin {
        Coin::Penny => 1,
        Coin::Nickel => 5,
        Coin::Dime => 10,
        Coin::Quarter(state) => {
            println!("State quarter from {:?}!", state);
            25
        },
    }
}

fn plus_one(x: Option<i32>) -> Option<i32> {
    match x {
        None => None,
        Some(i) => Some(i + 1),
        // _ => (),   // _通配符
    }
}

fn main() {

    let v = Some(0u8);
    match v {
        Some(3) => println!("three"),
        Some(4) => println!("four"),
        Some(5) => println!("five"),
        _ => (),
    }
    if let Some(3) = v {   // if let 处理只对应一个值的匹配
        println!("three");
    } else {
        println!("others");
    }

    let c = Coin::Quarter(UsState::Alaska);
    println!("{}", value_in_cents(c));
    let s = Rectangle::square(20);
    // let w = 30;
    // let l = 50;
    // println!("{}", area1(w, l));
    // let dim = (30, 50);
    // println!("{}", area2(dim));
    let rect = Rectangle {
        width: 30,
        length: 50,
    };
    let rect2 = Rectangle {         
        width: 20,         
        length: 40,     
    };

    // println!("{}", area(&rect));
    println!("{}", rect.area());
    println!("{}", rect.can_hold(&rect2));
    println!("{:#?}", rect);
}

fn area3(rect: &Rectangle) -> u32 {
    rect.width * rect.length
}

fn area2(dim: (u32, u32)) -> u32 {
    dim.0 * dim.1
}

fn area1(width: u32, length: u32) -> u32 {
    width * length
}
