/*
 * @Author: jhq
 * @Date: 2024-01-28 17:25:52
 * @LastEditTime: 2024-01-28 18:10:41
 * @Description: 
 */
use std::io;
use rand::Rng;
use std::cmp::Ordering;

fn main() {
    println!("猜数！");

    let secret_number = rand::thread_rng().gen_range(1, 101);

    loop {
        println!("猜测一个数！");

        let mut guess = String::new();
        io::stdin().read_line(&mut guess).expect("无法读取行");

        //shadow 
        let guess:u32 = match guess.trim().parse() {
            Ok(num) => num,
            Err(_) => continue,
        };
        println!("你猜测的数是：{}", guess);

        match guess.cmp(&secret_number) {
            Ordering::Less => println!("Too small!"),
            Ordering::Greater => println!("To big"),
            Ordering::Equal => {
                println!("You win!");
                break;
            }
        }
    }
    
}
