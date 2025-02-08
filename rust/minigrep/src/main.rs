/*
 * @Author: jhq
 * @Date: 2024-03-04 21:12:10
 * @LastEditTime: 2024-03-20 22:43:38
 * @Description:
 */

use minigrep::Config;
use std::env;
use std::process;


fn main() {
    // let args: Vec<String> = env::args().collect();
    // env::args()是一个迭代器，可以直接传入
    let config = Config::new(env::args()).unwrap_or_else(|err| {
        eprintln!("Problem parsing arguments {}", err);
        process::exit(1);
    });
    if let Err(e) = minigrep::run(config) {
        eprintln!("Application error:{}", e);
        process::exit(1);
    }
}
