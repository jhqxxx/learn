/*
 * @Author: jhq
 * @Date: 2024-02-18 22:23:33
 * @LastEditTime: 2024-02-25 22:52:54
 * @Description: 
 */

use std::collections::HashMap;

enum SpreadsheetCell {
    Int(i32),
    Float(f64),
    Text(String),
}

fn main() {

    let teams = vec![String::from("Blue"), String::from("Yellow")];
    let init_scores = vec![10, 50];
    let scores: HashMap<_, _> = teams.iter().zip(init_scores.iter()).collect();
    // let mut scores = HashMap::new();
    // scores.insert(String::from("Blue"), 10);
    // let len = String::from("Hola").len();
    // // Unicode 标量值
    // println!("{}", len);
    // let mut s = String::from("initial contents");
    // let data = "initial contents";
    // let s2= data.to_string();

    // s.push_str(" add contents");   // 字符串添加字符串切片
    // s.push('a');  //字符串添加单个字符 
    // let s3 = String::from("hello ");
    // let s4 = String::from("world");

    // // let s5 = s3 + &s4;  // +:连接字符串
    // // println!("s3:{}", s3); 
    // let s6 = format!("{}-{}", s3, s4);
    // println!("s4:{}", s4);
    // // println!("s5:{}", s5);
    // println!("{}", s6)
    
    // let row = vec![
    //     SpreadsheetCell::Int(3),
    //     SpreadsheetCell::Text(String::from("blue")),
    //     SpreadsheetCell::Float(10.99),
    // ];

    // let mut v = vec![1, 2, 3];
    // for i in &mut v {
    //     println!("{}", i);
    //     *i += 50;
    //     println!("{}", i);
    // }

    // let v: Vec<i32> = Vec::new();
    // let v1 = vec![1, 2, 3];
    // let mut v2 = Vec::new();
    // v2.push(1);
    // let mut v = vec![1, 2, 3, 4];
    // let third = &v[2];
    // println!("The third element is {}", third);

    // match v.get(2) {
    //     Some(third) => println!("The third element is {}", third),
    //     None => println!("There is not third element!"),
    // }
    
    // error:
    // let first = &v[0];
    // v.push(6);
    // println!("The first element id {}", first);

}
